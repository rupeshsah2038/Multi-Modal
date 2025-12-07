import jsonlines
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer

IMAGE_PROCESSOR = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    do_resize=True,
    size={"height": 224, "width": 224},
    do_center_crop=False,
    do_normalize=True,
)

def get_dataset(dataset_type, **kwargs):
    """
    Factory function to create the appropriate dataset based on type.
    
    Args:
        dataset_type: 'medpix' or 'wound'
        **kwargs: Dataset-specific arguments
    
    Returns:
        Dataset instance (MedPixDataset or WoundDataset)
    """
    if dataset_type == 'medpix':
        return MedPixDataset(**kwargs)
    elif dataset_type == 'wound':
        return WoundDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'medpix' or 'wound'.")

def get_num_classes(dataset_type, dataset_root=None, **kwargs):
    """
    Get the number of classes for each classification task based on dataset type.
    
    Args:
        dataset_type: 'medpix' or 'wound'
        dataset_root: Root directory of the dataset (required for wound dataset)
        **kwargs: Additional arguments for dataset inspection
    
    Returns:
        dict with 'modality' and 'location' class counts (or type/severity for wound)
    """
    if dataset_type == 'medpix':
        return {'modality': 2, 'location': 5}
    elif dataset_type == 'wound':
        # For wound dataset, we need to inspect the CSV to get dynamic class counts
        if dataset_root is None:
            raise ValueError("dataset_root is required for wound dataset class count inspection")
        csv_path = os.path.join(dataset_root, 'metadata.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"metadata.csv not found at {csv_path}")
        df = pd.read_csv(csv_path)
        type_col = kwargs.get('type_column', 'type')
        severity_col = kwargs.get('severity_column', 'severity')
        num_types = len(df[type_col].unique())
        num_severities = len(df[severity_col].unique())
        return {'modality': num_types, 'location': num_severities}
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

class MedPixDataset(Dataset):
    def __init__(self, data_jsonl_file, desc_jsonl_file, image_dir,
                 tokenizer_teacher, tokenizer_student, max_length=256):
        self.desc = []
        with jsonlines.open(desc_jsonl_file, 'r') as f:
            for obj in f:
                self.desc.append(obj)

        self.case_by_uid = {}
        with jsonlines.open(data_jsonl_file, 'r') as f:
            for obj in f:
                uid = obj.get('U_id')
                if uid:
                    self.case_by_uid[uid] = obj

        self.image_dir = image_dir
        self.max_length = max_length
        self.tokenizer_teacher = tokenizer_teacher
        self.tokenizer_student = tokenizer_student

        self.modality_map = {'CT': 0, 'MR': 1}
        self.location_map = {
            'Abdomen': 0,
            'Head': 1,
            'Reproductive and Urinary System': 2,
            'Thorax': 3,
            'Spine and Muscles': 4
        }

    def __len__(self):
        return len(self.desc)

    def __getitem__(self, idx):
        desc = self.desc[idx]

        img_path = os.path.join(self.image_dir, desc['image'] + '.png')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        pixel_values = IMAGE_PROCESSOR(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        case = self.case_by_uid.get(desc['U_id'])
        history = case['Case']['History'] if case and case.get('Case') and case['Case'].get('History') else ""
        caption = desc['Description']['Caption'] if desc.get('Description') and desc['Description'].get('Caption') else ""
        text = f"{caption}".strip() or "No description available"

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

        # Defensive checks: ensure token ids are within tokenizer vocab size.
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
                "Token id out of range for teacher tokenizer: "
                f"max id {max_id_t} >= vocab_size {vocab_t}.\n"
                "This likely means the tokenizer does not match the teacher model. "
                "Ensure your config text backbone maps to the correct pretrained tokenizer."
            )
        if vocab_s is not None and max_id_s >= vocab_s:
            raise ValueError(
                "Token id out of range for student tokenizer: "
                f"max id {max_id_s} >= vocab_size {vocab_s}.\n"
                "This likely means the tokenizer does not match the student model. "
                "Ensure your config text backbone maps to the correct pretrained tokenizer."
            )

        # Sanity check: attention mask and input ids have same shape
        if enc_t['input_ids'].shape != enc_t['attention_mask'].shape:
            raise ValueError('Teacher input_ids and attention_mask shape mismatch')
        if enc_s['input_ids'].shape != enc_s['attention_mask'].shape:
            raise ValueError('Student input_ids and attention_mask shape mismatch')

        modality = self.modality_map.get(desc['Type'], 0)
        location = self.location_map.get(desc['Location Category'], 0)

        return {
            'pixel_values': pixel_values,
            'input_ids_teacher': enc_t['input_ids'].squeeze(0),
            'attention_mask_teacher': enc_t['attention_mask'].squeeze(0),
            'input_ids_student': enc_s['input_ids'].squeeze(0),
            'attention_mask_student': enc_s['attention_mask'].squeeze(0),
            'modality': torch.tensor(modality, dtype=torch.long),
            'location': torch.tensor(location, dtype=torch.long),
        }


class WoundDataset(Dataset):
    """
    Dataset for Wound-1-0 with structure:
      - images/: contains all images
      - metadata.csv: columns = file_path, type, severity, description
    
    Classification tasks:
      - Type classification (wound type) -> mapped to 'modality' 
      - Severity classification (wound severity level) -> mapped to 'location'
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
        
        # Get labels (map to 'modality' and 'location' keys for compatibility)
        wound_type = self.type_map.get(row[self.type_col], 0)
        severity = self.severity_map.get(row[self.severity_col], 0)
        
        return {
            'pixel_values': pixel_values,
            'input_ids_teacher': enc_t['input_ids'].squeeze(0),
            'attention_mask_teacher': enc_t['attention_mask'].squeeze(0),
            'input_ids_student': enc_s['input_ids'].squeeze(0),
            'attention_mask_student': enc_s['attention_mask'].squeeze(0),
            'modality': torch.tensor(wound_type, dtype=torch.long),  # Type classification
            'location': torch.tensor(severity, dtype=torch.long),    # Severity classification
        }
    
    def get_num_classes(self):
        """Return number of classes for each task"""
        return {
            'modality': len(self.type_map),
            'location': len(self.severity_map)
        }
