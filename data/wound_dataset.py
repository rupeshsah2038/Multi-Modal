import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor

IMAGE_PROCESSOR = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    do_resize=True,
    size={"height": 224, "width": 224},
    do_center_crop=False,
    do_normalize=True,
)

class WoundDataset(Dataset):
    """
    Dataset for Wound-1-0 with structure:
      - images/: contains all images
      - metadata.csv: columns = file_path, type, severity, description
    
    Classification tasks:
      - Type classification (wound type)
      - Severity classification (wound severity level)
    """
    def __init__(self, csv_file, image_dir, tokenizer_teacher, tokenizer_student, 
                 max_length=256, type_column='type', severity_column='severity', 
                 description_column='description', filepath_column='file_path'):
        """
        Args:
            csv_file: Path to metadata.csv
            image_dir: Path to images/ directory
            tokenizer_teacher: Tokenizer for teacher model
            tokenizer_student: Tokenizer for student model
            max_length: Max token length for text encoding
            type_column: Column name for wound type
            severity_column: Column name for severity
            description_column: Column name for description
            filepath_column: Column name for image file path
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.max_length = max_length
        self.tokenizer_teacher = tokenizer_teacher
        self.tokenizer_student = tokenizer_student
        
        # Column names (configurable for flexibility)
        self.type_col = type_column
        self.severity_col = severity_column
        self.desc_col = description_column
        self.filepath_col = filepath_column
        
        # Build label mappings dynamically from data
        self.type_map = {label: idx for idx, label in enumerate(sorted(self.df[self.type_col].unique()))}
        self.severity_map = {label: idx for idx, label in enumerate(sorted(self.df[self.severity_col].unique()))}
        
        # Store reverse mappings for interpretability
        self.type_labels = {idx: label for label, idx in self.type_map.items()}
        self.severity_labels = {idx: label for label, idx in self.severity_map.items()}
        
        print(f"Wound dataset loaded: {len(self.df)} samples")
        print(f"Type classes ({len(self.type_map)}): {list(self.type_map.keys())}")
        print(f"Severity classes ({len(self.severity_map)}): {list(self.severity_map.keys())}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_filename = row[self.filepath_col]
        img_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        pixel_values = IMAGE_PROCESSOR(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Get text description
        text = str(row[self.desc_col]).strip() if pd.notna(row[self.desc_col]) else "No description available"
        
        # Tokenize for both teacher and student
        enc_t = self.tokenizer_teacher(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        enc_s = self.tokenizer_student(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        # Defensive checks: ensure token ids are within tokenizer vocab size
        def _vocab_size(tokenizer):
            vs = getattr(tokenizer, 'vocab_size', None)
            if vs is not None:
                return int(vs)
            try:
                gv = tokenizer.get_vocab()
                return len(gv)
            except Exception:
                return None
        
        vocab_t = _vocab_size(self.tokenizer_teacher)
        vocab_s = _vocab_size(self.tokenizer_student)
        
        max_id_t = int(enc_t['input_ids'].max().item())
        max_id_s = int(enc_s['input_ids'].max().item())
        
        if vocab_t is not None and max_id_t >= vocab_t:
            raise ValueError(
                f"Token id out of range for teacher tokenizer: "
                f"max id {max_id_t} >= vocab_size {vocab_t}.\n"
                "This likely means the tokenizer does not match the teacher model."
            )
        if vocab_s is not None and max_id_s >= vocab_s:
            raise ValueError(
                f"Token id out of range for student tokenizer: "
                f"max id {max_id_s} >= vocab_size {vocab_s}.\n"
                "This likely means the tokenizer does not match the student model."
            )
        
        # Get labels
        wound_type = self.type_map.get(row[self.type_col], 0)
        severity = self.severity_map.get(row[self.severity_col], 0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids_teacher': enc_t['input_ids'].squeeze(0),
            'attention_mask_teacher': enc_t['attention_mask'].squeeze(0),
            'input_ids_student': enc_s['input_ids'].squeeze(0),
            'attention_mask_student': enc_s['attention_mask'].squeeze(0),
            'modality': torch.tensor(wound_type, dtype=torch.long),  # Using 'modality' key for type
            'location': torch.tensor(severity, dtype=torch.long),    # Using 'location' key for severity
        }
    
    def get_num_classes(self):
        """Return number of classes for each task"""
        return {
            'type': len(self.type_map),
            'severity': len(self.severity_map)
        }
