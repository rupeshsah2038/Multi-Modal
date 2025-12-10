import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import time

@torch.no_grad()
def evaluate_detailed(model, loader, device, logger=None, split="dev", token_type='student', 
                      task1_label='modality', task2_label='location'):
    """
    Evaluate model with configurable task labels.
    
    Args:
        task1_label: Label for primary task (default: 'modality')
        task2_label: Label for secondary task (default: 'location')
    """
    model.eval()
    all_mod_pred, all_mod_true = [], []
    all_loc_pred, all_loc_true = [], []
    all_mod_prob, all_loc_prob = [], []
    start_time = time.time()

    input_ids_key = f'input_ids_{token_type}'
    attention_mask_key = f'attention_mask_{token_type}'

    for batch in loader:
        pv = batch['pixel_values'].to(device)
        ids = batch[input_ids_key].to(device)
        mask = batch[attention_mask_key].to(device)
        y_mod = batch['modality'].cpu().numpy()
        y_loc = batch['location'].cpu().numpy()

        out = model(pv, ids, mask)
        mod_logits = out['logits_modality']
        loc_logits = out['logits_location']

        mod_prob = F.softmax(mod_logits, dim=-1).cpu().numpy()
        loc_prob = F.softmax(loc_logits, dim=-1).cpu().numpy()
        mod_pred = mod_logits.argmax(dim=-1).cpu().numpy()
        loc_pred = loc_logits.argmax(dim=-1).cpu().numpy()

        all_mod_true.extend(y_mod)
        all_loc_true.extend(y_loc)
        all_mod_pred.extend(mod_pred)
        all_loc_pred.extend(loc_pred)
        all_mod_prob.extend(mod_prob)
        all_loc_prob.extend(loc_prob)

    infer_time = (time.time() - start_time) / max(1, len(loader.dataset)) * 1000

    mod_true = np.array(all_mod_true)
    loc_true = np.array(all_loc_true)
    mod_pred = np.array(all_mod_pred)
    loc_pred = np.array(all_loc_pred)
    mod_prob = np.array(all_mod_prob)
    loc_prob = np.array(all_loc_prob)

    mod_acc = accuracy_score(mod_true, mod_pred)
    loc_acc = accuracy_score(loc_true, loc_pred)
    mod_f1 = f1_score(mod_true, mod_pred, average='macro')
    loc_f1 = f1_score(loc_true, loc_pred, average='macro')
    mod_prec = precision_score(mod_true, mod_pred, average='macro')
    loc_prec = precision_score(loc_true, loc_pred, average='macro')
    mod_rec = recall_score(mod_true, mod_pred, average='macro')
    loc_rec = recall_score(loc_true, loc_pred, average='macro')

    try:
        mod_auc = roc_auc_score(mod_true, mod_prob[:, 1]) if mod_prob.shape[1] == 2 else roc_auc_score(mod_true, mod_prob, multi_class='ovr')
        loc_auc = roc_auc_score(loc_true, loc_prob, multi_class='ovr')
    except:
        mod_auc = loc_auc = 0.0

    if logger:
        logger.save_confusion(mod_true, mod_pred, task1_label, split)
        logger.save_confusion(loc_true, loc_pred, task2_label, split)

    metrics = {
        f"{split}_{task1_label}_acc": float(mod_acc),
        f"{split}_{task2_label}_acc": float(loc_acc),
        f"{split}_{task1_label}_f1": float(mod_f1),
        f"{split}_{task2_label}_f1": float(loc_f1),
        f"{split}_{task1_label}_prec": float(mod_prec),
        f"{split}_{task2_label}_prec": float(loc_prec),
        f"{split}_{task1_label}_rec": float(mod_rec),
        f"{split}_{task2_label}_rec": float(loc_rec),
        f"{split}_{task1_label}_auc": float(mod_auc),
        f"{split}_{task2_label}_auc": float(loc_auc),
        f"{split}_infer_ms": float(infer_time),
    }

    for k, v in metrics.items():
        if 'infer_ms' in k:
            print(f"{k}: {v:.2f}ms")
        else:
            print(f"{k}: {v:.4f}")

    return metrics
