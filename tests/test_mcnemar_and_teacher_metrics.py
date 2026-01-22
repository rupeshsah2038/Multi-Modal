import os
import json
import shutil
import numpy as np

from utils.logger import MetricsLogger
from utils.results_logger import ResultsLogger
from utils.metrics import mcnemar_test


def _reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def test_mcnemar_exact_known_value():
    # Construct a case with b=10, c=0 => n=10
    # Exact two-sided p-value = 2 * (C(10,0)/2^10) = 2/1024 = 0.001953125
    y_true = np.zeros(20, dtype=int)
    pred_a = y_true.copy()
    pred_b = y_true.copy()

    # Make 10 discordant pairs: A correct, B wrong
    pred_b[:10] = 1

    out = mcnemar_test(y_true, pred_a, pred_b, exact=True)
    assert out['b'] == 10, out
    assert out['c'] == 0, out
    assert abs(out['pvalue'] - 0.001953125) < 1e-12, out


def test_results_logger_writes_teacher_metrics():
    out_dir = 'logs/_mock_test_results_logger'
    _reset_dir(out_dir)

    cfg = {
        'teacher': {'vision': 'vit-large', 'text': 'bio-clinical-bert', 'fusion_layers': 2},
        'student': {'vision': 'vit-base', 'text': 'distilbert', 'fusion_layers': 1},
        'training': {},
        'data': {},
        'fusion': {},
        'loss': {},
        'evaluation': {'save_teacher_metrics': True},
    }

    teacher_dev = {'teacher_dev_modality_acc': 0.91}
    teacher_test = {'teacher_test_modality_acc': 0.89}
    dev = {'dev_modality_acc': 0.75}
    test = {'test_modality_acc': 0.78}

    rl = ResultsLogger(out_dir)
    rl.log_experiment(
        cfg,
        metrics={'train': {'final_loss': 1.23}},
        dev_metrics=dev,
        test_metrics=test,
        teacher_dev_metrics=teacher_dev,
        teacher_test_metrics=teacher_test,
        teacher_params={'total_params': 123, 'params_millions': 0.000123},
        student_params={'total_params': 45, 'params_millions': 0.000045},
    )

    results_path = os.path.join(out_dir, 'results.json')
    assert os.path.exists(results_path), 'results.json not created'

    j = json.load(open(results_path))
    assert 'metrics' in j, 'metrics missing'
    assert 'teacher' in j['metrics'], 'metrics.teacher missing'
    assert 'dev' in j['metrics']['teacher'], 'metrics.teacher.dev missing'
    assert 'test' in j['metrics']['teacher'], 'metrics.teacher.test missing'
    assert j['metrics']['teacher']['dev'] == teacher_dev, j['metrics']['teacher']['dev']
    assert j['metrics']['teacher']['test'] == teacher_test, j['metrics']['teacher']['test']


def test_metrics_logger_saves_teacher_confusions():
    out_dir = 'logs/_mock_test_metrics_logger'
    _reset_dir(out_dir)

    logger = MetricsLogger(out_dir)
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    logger.save_confusion(y_true, y_pred, task_name='modality', split='teacher_test')
    cm_path = os.path.join(out_dir, 'cm_modality_teacher_test.npy')
    assert os.path.exists(cm_path), 'teacher confusion matrix not saved'


if __name__ == '__main__':
    # Simple runner (no pytest dependency)
    test_mcnemar_exact_known_value()
    test_results_logger_writes_teacher_metrics()
    test_metrics_logger_saves_teacher_confusions()
    print('OK: mock tests passed')
