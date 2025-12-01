import torch
from torch.utils.data import DataLoader
from data.dataset import MedPixDataset
from models.teacher import Teacher
from models.student import Student
import importlib
import inspect
from utils.logger import MetricsLogger
from utils.metrics import evaluate_detailed
import yaml
import os
from datetime import datetime
from transformers import AutoTokenizer

def train_teacher(model, loader, device, epochs, lr):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0.0
        nsteps = 0
        for batch in loader:
            pv = batch['pixel_values'].to(device)
            ids = batch['input_ids_teacher'].to(device)
            mask = batch['attention_mask_teacher'].to(device)
            y_mod = batch['modality'].to(device)
            y_loc = batch['location'].to(device)
            out = model(pv, ids, mask)
            loss = ce(out['logits_modality'], y_mod) + ce(out['logits_location'], y_loc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nsteps += 1
        avg = total / max(1, nsteps)
        print(f"Teacher Epoch {epoch+1}/{epochs} – loss {avg:.4f}")
    return model

def train_student(student, teacher, loader, device, epochs, lr, distill_fn):
    teacher.eval()
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        nsteps = 0
        for batch in loader:
            pv = batch['pixel_values'].to(device)
            ids_s = batch['input_ids_student'].to(device)
            mask_s = batch['attention_mask_student'].to(device)
            ids_t = batch['input_ids_teacher'].to(device)
            mask_t = batch['attention_mask_teacher'].to(device)
            y_mod = batch['modality'].to(device)
            y_loc = batch['location'].to(device)
            with torch.no_grad():
                t_out = teacher(pv, ids_t, mask_t)
            s_out = student(pv, ids_s, mask_s)
            loss = distill_fn(s_out, t_out, y_mod, y_loc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nsteps += 1
        avg = total / max(1, nsteps)
        print(f"Student Epoch {epoch+1}/{epochs} – loss {avg:.4f}")
    return student

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    student_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def make_dataset(split):
        return MedPixDataset(
            data_jsonl_file=os.path.join(cfg['data']['root'], f"splitted_dataset/data_{split}.jsonl"),
            desc_jsonl_file=os.path.join(cfg['data']['root'], f"splitted_dataset/descriptions_{split}.jsonl"),
            image_dir=os.path.join(cfg['data']['root'], "images"),
            tokenizer_teacher=teacher_tokenizer,
            tokenizer_student=student_tokenizer,
        )
    
    train_dataset = make_dataset("train")
    dev_dataset = make_dataset("dev")
    test_dataset = make_dataset("test")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    dev_loader = DataLoader(dev_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    
    teacher = Teacher(
        vision=cfg['teacher']['vision'],
        text=cfg['teacher']['text'],
        fusion_layers=cfg['teacher']['fusion_layers']
    ).to(device)
    
    student = Student(
        vision=cfg['student']['vision'],
        text=cfg['student']['text'],
        fusion_layers=cfg['student']['fusion_layers']
    ).to(device)
    
    logger = MetricsLogger(cfg['logging']['log_dir'])
    
    # Create a distillation/loss object from config
    def _make_loss_from_cfg(cfg):
        # mapping: config name -> (module, class)
        mapping = {
            'vanilla': ('losses.vanilla', 'DistillationLoss'),
            'combined': ('losses.combined', 'MedKDCombinedLoss'),
            'crd': ('losses.crd', 'CRDLoss'),
            'rkd': ('losses.rkd', 'RKDLoss'),
            'mmd': ('losses.mmd', 'MMDLoss'),
        }
        loss_type = cfg.get('loss', {}).get('type', 'vanilla')
        module_name, class_name = mapping.get(loss_type, mapping['vanilla'])
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception:
            # Fall back to vanilla DistillationLoss if anything goes wrong
            module = importlib.import_module('losses.vanilla')
            cls = getattr(module, 'DistillationLoss')

        # Build kwargs from cfg['training'] using only the parameters the class accepts
        training_cfg = cfg.get('training', {}) or {}
        kwargs = {}
        try:
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name == 'self' or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in training_cfg:
                    kwargs[name] = training_cfg[name]
        except Exception:
            # if introspection fails, pass nothing and rely on defaults
            kwargs = {}

        return cls(**kwargs)

    distill_fn = _make_loss_from_cfg(cfg)
    
    # Train teacher
    print("\n=== Training Teacher ===")
    teacher = train_teacher(teacher, train_loader, device, epochs=cfg['training']['teacher_epochs'], lr=cfg['training']['teacher_lr'])
    
    torch.cuda.empty_cache()
    
    print("\n=== Distilling to Student ===")
    best_dev_score = 0.0
    for epoch in range(1, cfg['training']['student_epochs'] + 1):
        student = train_student(student, teacher, train_loader, device, epochs=1, lr=cfg['training']['student_lr'], distill_fn=distill_fn)
        dev_metrics = evaluate_detailed(student, dev_loader, device, logger=logger, split="dev", token_type='student')
        dev_score = (dev_metrics['dev_mod_f1'] + dev_metrics['dev_loc_f1']) / 2
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_path = os.path.join(cfg['logging']['log_dir'], "student_best.pth")
            torch.save(student.state_dict(), best_path)
            print(f"  New best dev score: {dev_score:.4f}")
    
    # Test
    if best_dev_score > 0:
        best_path = os.path.join(cfg['logging']['log_dir'], "student_best.pth")
        student.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate_detailed(student, test_loader, device, logger=logger, split="test", token_type='student')
    
    final_path = os.path.join(cfg['logging']['log_dir'], "student_final.pth")
    torch.save(student.state_dict(), final_path)
    logger.save_csv()
    logger.save_json()
    print(f"All outputs saved in {cfg['logging']['log_dir']}")
    
if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
