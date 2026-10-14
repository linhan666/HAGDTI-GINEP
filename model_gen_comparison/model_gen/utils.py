import os
import warnings
import numpy as np
import torch
import re
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, precision_recall_curve, auc,
                             precision_score, recall_score)

# ==============================================================================
# 1. Tokenizer (保持不变，用于 SMILES 生成)
# ==============================================================================
class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    # Fixed-length reserved token block; trailing slots reserved for future use.
    SPECIAL_TOKENS = (
        '<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>'
    ) + tuple(f'<t_{i}>' for i in range(6, NUM_RESERVED_TOKENS))

    PATTEN = re.compile(r'\[[^\]]+\]'
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()
        for a in smiles_set:
            vocabs.update(re.findall(Tokenizer.PATTEN, a))
        return vocabs

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles):
        if not smiles:  # handles None and empty string
            return [self.s2i['<sos>'], self.s2i['<eos>']]
        l = []
        # <sos> is added at start, <eos> at end
        for s in ('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>'):
            if s not in self.s2i:
                a = self.s2i['<unk>']
            else:
                a = self.s2i[s]
            l.append(a)
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s.get(i, '<unk>')
                if c == '<eos>':
                    break
                if c == '<sos>' or c == '<pad>':
                    continue
                s.append(c)
            smiles.append(''.join(s))
        return smiles

# ==============================================================================
# 2. 评价指标 (适配 DTI 分类任务)
# ==============================================================================

def get_auc(y_true, y_prob):
    """计算 ROC-AUC"""
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().detach().numpy()
    if isinstance(y_prob, torch.Tensor): y_prob = y_prob.cpu().detach().numpy()

    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError as e:
        # Single-class labels: AUC undefined; warn and return NaN for transparency.
        warnings.warn(f"get_auc: roc_auc_score failed ({e}), returning NaN")
        return float('nan')

def get_aupr(y_true, y_prob):
    """计算 AUPR (Area Under Precision-Recall Curve)"""
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().detach().numpy()
    if isinstance(y_prob, torch.Tensor): y_prob = y_prob.cpu().detach().numpy()

    try:
        return average_precision_score(y_true, y_prob)
    except ValueError as e:
        warnings.warn(f"get_aupr: average_precision_score failed ({e}), returning NaN")
        return float('nan')

def get_metrics(y_true, y_prob, threshold=0.5):
    """一次性返回所有分类指标字典"""
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().detach().numpy()
    if isinstance(y_prob, torch.Tensor): y_prob = y_prob.cpu().detach().numpy()

    y_pred = (y_prob > threshold).astype(int)

    return {
        'AUC': get_auc(y_true, y_prob),
        'AUPR': get_aupr(y_true, y_prob),
        'F1': f1_score(y_true, y_pred),
        'ACC': accuracy_score(y_true, y_pred)
    }


def get_ternary_metrics(y_true, y_pred_probs):
    """
    Three-class DTI classification metrics for v4 ternary activity labels.

    Args:
        y_true:       array-like [N], int values in {0=inactive, 1=agonist, 2=inhibitor}
        y_pred_probs: array-like [N, 3], softmax probabilities for the three classes

    Returns:
        dict with keys: ACC3, macro_F1,
                        F1_inactive/agonist/inhibitor,
                        precision_inactive/agonist/inhibitor,
                        recall_inactive/agonist/inhibitor
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.cpu().detach().numpy()

    y_true       = np.asarray(y_true, dtype=int).ravel()
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)

    if y_pred_probs.ndim != 2 or y_pred_probs.shape[1] != 3:
        raise ValueError(f"y_pred_probs must be [N, 3], got shape {y_pred_probs.shape}")
    if len(y_true) != y_pred_probs.shape[0]:
        raise ValueError("y_true and y_pred_probs must have the same number of samples")
    if np.isnan(y_pred_probs).any() or np.isinf(y_pred_probs).any():
        raise ValueError("y_pred_probs contains NaN or Inf values")
    if not (np.all(y_pred_probs >= 0) and np.all(y_pred_probs <= 1)):
        raise ValueError("y_pred_probs values must be in [0, 1]; pass softmax probabilities, not logits")
    row_sums = y_pred_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("y_pred_probs rows must sum to 1.0 (±1e-5); apply softmax before calling")

    y_pred = y_pred_probs.argmax(axis=1)
    labels = [0, 1, 2]
    class_names = ['inactive', 'agonist', 'inhibitor']

    per_f1  = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    result = {
        'ACC3':     float(accuracy_score(y_true, y_pred)),
        'macro_F1': float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)),
    }
    for i, name in enumerate(class_names):
        result[f'F1_{name}']        = float(per_f1[i])
        result[f'precision_{name}'] = float(per_pre[i])
        result[f'recall_{name}']    = float(per_rec[i])

    return result

# ==============================================================================
# 3. 日志工具
# ==============================================================================

def logging(msg, log_dir):
    """简单的日志记录"""
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    fpath = os.path.join(log_dir, "log.txt")
    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print(msg)