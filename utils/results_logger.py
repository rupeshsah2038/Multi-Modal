import os
import json
from datetime import datetime

class ResultsLogger:
    """Log experiment results including config, metrics, and model details."""
    
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.results_file = os.path.join(log_dir, 'results.json')
        self.teacher_results_file = os.path.join(log_dir, 'teacher.json')
        self.results = {}
        self.teacher_results = {}

    def _save_json(self, path: str, payload: dict):
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
    
    def log_experiment(self, config, metrics, dev_metrics=None, test_metrics=None,
                      teacher_dev_metrics=None, teacher_test_metrics=None,
                      teacher_params=None, student_params=None):
        """
        Log a complete experiment run.
        
        Args:
            config: full config dict
            metrics: dict with train/val metrics
            dev_metrics: optional dev set metrics
            test_metrics: optional test set metrics
            teacher_dev_metrics: optional teacher dev set metrics
            teacher_test_metrics: optional teacher test set metrics
            teacher_params: optional dict with teacher parameter counts
            student_params: optional dict with student parameter counts
        """
        timestamp = datetime.now().isoformat()

        self.results = {
            'timestamp': timestamp,
            'config': config,
            'metrics': {
                'train': metrics.get('train', {}),
                'dev': dev_metrics or {},
                'test': test_metrics or {},
                'teacher': {
                    'dev': teacher_dev_metrics or {},
                    'test': teacher_test_metrics or {},
                },
            },
            'models': {
                'teacher': {
                    'vision': config.get('teacher', {}).get('vision'),
                    'text': config.get('teacher', {}).get('text'),
                    'fusion_layers': config.get('teacher', {}).get('fusion_layers'),
                    'total_params': teacher_params.get('total_params') if teacher_params else None,
                    'params_millions': teacher_params.get('params_millions') if teacher_params else None,
                },
                'student': {
                    'vision': config.get('student', {}).get('vision'),
                    'text': config.get('student', {}).get('text'),
                    'fusion_layers': config.get('student', {}).get('fusion_layers'),
                    'total_params': student_params.get('total_params') if student_params else None,
                    'params_millions': student_params.get('params_millions') if student_params else None,
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

        # Also persist teacher metrics separately for easier downstream analysis.
        tdev = teacher_dev_metrics or {}
        ttest = teacher_test_metrics or {}
        if (len(tdev) > 0) or (len(ttest) > 0):
            self.teacher_results = {
                'timestamp': timestamp,
                'config': {
                    'teacher': config.get('teacher', {}) or {},
                    'training': {
                        'teacher_epochs': config.get('training', {}).get('teacher_epochs'),
                        'teacher_lr': config.get('training', {}).get('teacher_lr'),
                    },
                    'data': config.get('data', {}) or {},
                    'evaluation': config.get('evaluation', {}) or {},
                    'fusion': config.get('fusion', {}) or {},
                    'loss': config.get('loss', {}) or {},
                },
                'metrics': {
                    'dev': tdev,
                    'test': ttest,
                },
                'model': {
                    'vision': config.get('teacher', {}).get('vision'),
                    'text': config.get('teacher', {}).get('text'),
                    'fusion_layers': config.get('teacher', {}).get('fusion_layers'),
                    'total_params': teacher_params.get('total_params') if teacher_params else None,
                    'params_millions': teacher_params.get('params_millions') if teacher_params else None,
                },
            }
        self.save()
    
    def save(self):
        """Save results to JSON file."""
        self._save_json(self.results_file, self.results)
        print(f"Experiment results saved to {self.results_file}")

        if self.teacher_results:
            self._save_json(self.teacher_results_file, self.teacher_results)
            print(f"Teacher metrics saved to {self.teacher_results_file}")
