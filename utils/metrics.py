import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import time
import math


def mcnemar_test(y_true: np.ndarray,
                 pred_a: np.ndarray,
                 pred_b: np.ndarray,
                 *,
                 exact: bool = True,
                 correction: bool = True) -> dict:
    """McNemar's test for paired nominal data (two classifiers on same examples).

    Returns a dict with the 2x2 discordant counts (b, c) and a p-value.

    Table definition (correctness vs correctness):
      - b: A correct, B wrong
      - c: A wrong, B correct

    Args:
        y_true: ground truth labels, shape (N,)
        pred_a: predictions from model A, shape (N,)
        pred_b: predictions from model B, shape (N,)
        exact: if True, uses exact binomial McNemar p-value
        correction: if exact=False, applies continuity correction for chi-square approx
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if y_true.shape != pred_a.shape or y_true.shape != pred_b.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, pred_a={pred_a.shape}, pred_b={pred_b.shape}")

    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    b = int(np.sum(a_correct & (~b_correct)))
    c = int(np.sum((~a_correct) & b_correct))
    n = b + c

    if n == 0:
        # Identical correctness on all samples => no evidence of difference.
        return {
            'b': b,
            'c': c,
            'n': n,
            'statistic': 0.0,
            'pvalue': 1.0,
            'method': 'degenerate',
        }

    if exact:
        # Exact two-sided binomial test with p=0.5 over discordant pairs.
        k = min(b, c)
        # p = 2 * sum_{i=0..k} C(n, i) / 2^n
        tail = 0.0
        denom = 2.0 ** n
        for i in range(0, k + 1):
            tail += math.comb(n, i) / denom
        pvalue = min(1.0, 2.0 * tail)
        # Often reported stat for exact is min(b,c); keep a chi-square style stat too.
        statistic = float(min(b, c))
        method = 'exact-binomial'
        return {
            'b': b,
            'c': c,
            'n': n,
            'statistic': statistic,
            'pvalue': float(pvalue),
            'method': method,
        }

    # Chi-square approximation with 1 dof.
    diff = abs(b - c)
    if correction:
        diff = max(0.0, diff - 1.0)
    statistic = (diff * diff) / n
    # For df=1, survival function is erfc(sqrt(x/2))
    pvalue = math.erfc(math.sqrt(statistic / 2.0))
    return {
        'b': b,
        'c': c,
        'n': n,
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'method': 'chi2-approx' + ('-cc' if correction else ''),
    }

@torch.no_grad()
def evaluate_detailed(model, loader, device, logger=None, split="dev", token_type='student', 
                      task1_label='modality', task2_label='location', return_raw: bool = False):
    """
    Evaluate model with configurable task labels.
    
    Args:
        task1_label: Label for primary task (default: 'modality')
        task2_label: Label for secondary task (default: 'location')
    """
    model.eval()
    all_mod_pred, all_mod_true = [], []
    all_loc_pred, all_loc_true = [], []
    all_mod_prob, all_loc_prob = [], []
    start_time = time.time()

    input_ids_key = f'input_ids_{token_type}'
    attention_mask_key = f'attention_mask_{token_type}'

    for batch in loader:
        pv = batch['pixel_values'].to(device)
        ids = batch[input_ids_key].to(device)
        mask = batch[attention_mask_key].to(device)
        y_mod = batch['modality'].cpu().numpy()
        y_loc = batch['location'].cpu().numpy()

        out = model(pv, ids, mask)
        mod_logits = out['logits_modality']
        loc_logits = out['logits_location']

        mod_prob = F.softmax(mod_logits, dim=-1).cpu().numpy()
        loc_prob = F.softmax(loc_logits, dim=-1).cpu().numpy()
        mod_pred = mod_logits.argmax(dim=-1).cpu().numpy()
        loc_pred = loc_logits.argmax(dim=-1).cpu().numpy()

        all_mod_true.extend(y_mod)
        all_loc_true.extend(y_loc)
        all_mod_pred.extend(mod_pred)
        all_loc_pred.extend(loc_pred)
        all_mod_prob.extend(mod_prob)
        all_loc_prob.extend(loc_prob)

    infer_time = (time.time() - start_time) / max(1, len(loader.dataset)) * 1000

    mod_true = np.array(all_mod_true)
    loc_true = np.array(all_loc_true)
    mod_pred = np.array(all_mod_pred)
    loc_pred = np.array(all_loc_pred)
    mod_prob = np.array(all_mod_prob)
    loc_prob = np.array(all_loc_prob)

    mod_acc = accuracy_score(mod_true, mod_pred)
    loc_acc = accuracy_score(loc_true, loc_pred)
    mod_f1 = f1_score(mod_true, mod_pred, average='macro')
    loc_f1 = f1_score(loc_true, loc_pred, average='macro')
    mod_prec = precision_score(mod_true, mod_pred, average='macro')
    loc_prec = precision_score(loc_true, loc_pred, average='macro')
    mod_rec = recall_score(mod_true, mod_pred, average='macro')
    loc_rec = recall_score(loc_true, loc_pred, average='macro')

    try:
        mod_auc = roc_auc_score(mod_true, mod_prob[:, 1]) if mod_prob.shape[1] == 2 else roc_auc_score(mod_true, mod_prob, multi_class='ovr')
        loc_auc = roc_auc_score(loc_true, loc_prob, multi_class='ovr')
    except:
        mod_auc = loc_auc = 0.0

    if logger:
        logger.save_confusion(mod_true, mod_pred, task1_label, split)
        logger.save_confusion(loc_true, loc_pred, task2_label, split)

    metrics = {
        f"{split}_{task1_label}_acc": float(mod_acc),
        f"{split}_{task2_label}_acc": float(loc_acc),
        f"{split}_{task1_label}_f1": float(mod_f1),
        f"{split}_{task2_label}_f1": float(loc_f1),
        f"{split}_{task1_label}_prec": float(mod_prec),
        f"{split}_{task2_label}_prec": float(loc_prec),
        f"{split}_{task1_label}_rec": float(mod_rec),
        f"{split}_{task2_label}_rec": float(loc_rec),
        f"{split}_{task1_label}_auc": float(mod_auc),
        f"{split}_{task2_label}_auc": float(loc_auc),
        f"{split}_infer_ms": float(infer_time),
    }

    for k, v in metrics.items():
        if 'infer_ms' in k:
            print(f"{k}: {v:.2f}ms")
        else:
            print(f"{k}: {v:.4f}")

    if return_raw:
        raw = {
            'task1_label': task1_label,
            'task2_label': task2_label,
            'y_true_task1': mod_true,
            'y_pred_task1': mod_pred,
            'y_true_task2': loc_true,
            'y_pred_task2': loc_pred,
        }
        return metrics, raw

    return metrics
