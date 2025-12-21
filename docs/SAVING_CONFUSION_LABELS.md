# Example: Saving Class Labels for Confusion Matrices

This example shows how to save class labels for confusion matrices using the `MetricsLogger.save_labels()` method for both MedPix and Wound datasets.

## MedPix Dataset
```python
from utils.logger import MetricsLogger

# MedPix modality and location labels
modality_labels = ['CT', 'MR']
location_labels = ['Abdomen', 'Head', 'Reproductive and Urinary System', 'Thorax', 'Spine and Muscles']

logger = MetricsLogger(run_dir="logs/example_run")
logger.save_labels(modality_labels, task_name="modality")
logger.save_labels(location_labels, task_name="location")
```

## Wound Dataset
```python
from utils.logger import MetricsLogger
from data.wound_dataset import WoundDataset

# Load wound dataset
wound_ds = WoundDataset(
    csv_file="datasets/Wound-1-0/metadata.csv",
    image_dir="datasets/Wound-1-0/images",
    tokenizer_teacher=...,  # your teacher tokenizer
    tokenizer_student=...,  # your student tokenizer
)

# Get type and severity labels
wound_type_labels = [wound_ds.type_labels[i] for i in range(len(wound_ds.type_labels))]
wound_severity_labels = [wound_ds.severity_labels[i] for i in range(len(wound_ds.severity_labels))]

logger = MetricsLogger(run_dir="logs/example_run")
logger.save_labels(wound_type_labels, task_name="type")
logger.save_labels(wound_severity_labels, task_name="severity")
```

## Usage in Plotting
When plotting confusion matrices, load the labels from the corresponding `.npy` file:
```python
import numpy as np
labels = np.load("logs/example_run/labels_modality.npy", allow_pickle=True)
# Use labels for axis ticks in your confusion matrix plot
```
