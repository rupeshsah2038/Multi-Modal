import os
import json
import csv
from collections import defaultdict
from datetime import datetime

class MetricsLogger:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.history = defaultdict(list)

    def log_epoch(self, epoch, train_loss, all_metrics):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        for k, v in all_metrics.items():
            if k == 'train_loss':
                continue
            self.history[k].append(v)

    def save_csv(self, filename="metrics.csv"):
        path = os.path.join(self.run_dir, filename)
        keys = list(self.history.keys())
        n = len(self.history['epoch'])
        rows = []
        for i in range(n):
            row = {k: self.history[k][i] if i < len(self.history[k]) else None for k in keys}
            rows.append(row)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics saved to {path}")

    def save_json(self, filename="metrics.json"):
        path = os.path.join(self.run_dir, filename)
        serial = {k: list(v) for k, v in self.history.items()}
        with open(path, 'w') as f:
            json.dump(serial, f, indent=2)
        print(f"Metrics JSON saved to {path}")

    def save_confusion(self, y_true, y_pred, task_name, split):
        from sklearn.metrics import confusion_matrix
        import numpy as np
        cm = confusion_matrix(y_true, y_pred)
        path = os.path.join(self.run_dir, f"cm_{task_name}_{split}.npy")
        np.save(path, cm)
        print(f"Confusion matrix saved: {path}")
