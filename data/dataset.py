import jsonlines
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
