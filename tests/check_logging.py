import os
import json
from utils.logger import MetricsLogger

out_dir = 'logs/test_metrics'
if os.path.exists(out_dir):
    # clear files
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

logger = MetricsLogger(out_dir)
logger.log_epoch(1, 0.123, {'dev_mod_f1': 0.5, 'dev_loc_f1': 0.6})
logger.log_epoch(2, 0.111, {'dev_mod_f1': 0.55, 'dev_loc_f1': 0.65})
logger.save_csv()
logger.save_json()

csv_path = os.path.join(out_dir, 'metrics.csv')
json_path = os.path.join(out_dir, 'metrics.json')
print('CSV exists:', os.path.exists(csv_path), 'size:', os.path.getsize(csv_path) if os.path.exists(csv_path) else 0)
print('JSON exists:', os.path.exists(json_path), 'size:', os.path.getsize(json_path) if os.path.exists(json_path) else 0)

with open(json_path, 'r') as f:
    data = json.load(f)
print('History keys in JSON:', list(data.keys()))
