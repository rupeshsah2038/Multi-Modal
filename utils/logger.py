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
        self.test_metrics = {}

    def log_epoch(self, epoch, train_loss, all_metrics):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        for k, v in all_metrics.items():
            if k == 'train_loss':
                continue
            self.history[k].append(v)

    def log_test(self, test_metrics):
        if test_metrics and isinstance(test_metrics, dict):
            self.test_metrics.update(test_metrics)

    def save_csv(self, filename="metrics.csv", test_metrics=None):
        if test_metrics and isinstance(test_metrics, dict):
            self.test_metrics.update(test_metrics)

        path = os.path.join(self.run_dir, filename)
        if self.test_metrics:
            keys = list(self.test_metrics.keys())
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerow(self.test_metrics)
            print(f"Test metrics saved to {path}")
        else:
            with open(path, 'w', newline='') as f:
                pass
            print(f"Metrics saved to {path} (empty: no test metrics recorded)")

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

    def save_labels(self, labels, task_name):
        import numpy as np
        path = os.path.join(self.run_dir, f"labels_{task_name}.npy")
        np.save(path, np.array(labels))
        print(f"Class labels saved: {path}")
