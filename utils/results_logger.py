import os
import json
from datetime import datetime

class ResultsLogger:
    """Log experiment results including config, metrics, and model details."""
    
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.results_file = os.path.join(log_dir, 'results.json')
        self.results = {}
    
    def log_experiment(self, config, metrics, dev_metrics=None, test_metrics=None):
        """
        Log a complete experiment run.
        
        Args:
            config: full config dict
            metrics: dict with train/val metrics
            dev_metrics: optional dev set metrics
            test_metrics: optional test set metrics
        """
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': {
                'train': metrics.get('train', {}),
                'dev': dev_metrics or {},
                'test': test_metrics or {},
            },
            'models': {
                'teacher': {
                    'vision': config.get('teacher', {}).get('vision'),
                    'text': config.get('teacher', {}).get('text'),
                    'fusion_layers': config.get('teacher', {}).get('fusion_layers'),
                },
                'student': {
                    'vision': config.get('student', {}).get('vision'),
                    'text': config.get('student', {}).get('text'),
                    'fusion_layers': config.get('student', {}).get('fusion_layers'),
                },
            },
            'training': {
                'teacher_epochs': config.get('training', {}).get('teacher_epochs'),
                'student_epochs': config.get('training', {}).get('student_epochs'),
                'teacher_lr': config.get('training', {}).get('teacher_lr'),
                'student_lr': config.get('training', {}).get('student_lr'),
                'alpha': config.get('training', {}).get('alpha'),
                'beta': config.get('training', {}).get('beta'),
                'gamma': config.get('training', {}).get('gamma'),
                'T': config.get('training', {}).get('T'),
            },
            'data': {
                'root': config.get('data', {}).get('root'),
                'batch_size': config.get('data', {}).get('batch_size'),
                'num_workers': config.get('data', {}).get('num_workers'),
            },
            'fusion': {
                'type': config.get('fusion', {}).get('type', 'simple'),
            },
            'loss': {
                'type': config.get('loss', {}).get('type', 'vanilla'),
            },
        }
        self.save()
    
    def save(self):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Experiment results saved to {self.results_file}")
