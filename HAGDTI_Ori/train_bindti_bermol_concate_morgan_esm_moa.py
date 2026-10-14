# ==============================================================================
# File: train_bindti_bermol_concate_morgan_esm_moa.py
# Train WideDeepHMCNFModel with the feature encoders used by the HAGDTI pipeline.
# The model returns probabilities ordered as [Active, Agonist, Antagonist].
# ==============================================================================

import argparse
import builtins
import hashlib
import json
import math
import os
import pickle
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

if __package__:
    from .model_bindti_concate_bermol_morgan_esm_hmcn import WideDeepHMCNFModel
else:
    from model_bindti_concate_bermol_morgan_esm_hmcn import WideDeepHMCNFModel

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as exc:
    raise RuntimeError('RDKit is required for SMILES cleaning and Morgan fingerprints.') from exc

try:
    from transformers import AutoTokenizer, EsmModel
except ImportError as exc:
    raise RuntimeError('transformers is required for ESM feature extraction.') from exc


def _load_bermol_tokenizer(model_dir):
    """Import BerMol from the configured code directory, not a host-specific path."""
    if model_dir and model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    try:
        from bermol.tokenizer import BerMolTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f'Could not import bermol.tokenizer from configured path: {model_dir}'
        ) from exc
    return BerMolTokenizer

# ==============================================================================
# 1. Hierarchical probability loss
# ==============================================================================
class RecursiveHierarchicalLoss(torch.nn.Module):
    """Weighted BCE over HMCN probabilities plus parent-child consistency penalty."""

    def __init__(self, penalty=1e-5, pos_weight=None):
        super().__init__()
        self.penalty = penalty
        self.register_buffer('pos_weight', pos_weight)
        self.hierarchy = ((0, 1), (0, 2))  # Active is the parent of both MoA labels.

    def forward(self, probabilities, targets):
        if probabilities.shape != targets.shape:
            raise ValueError(
                f'Prediction/target shape mismatch: {tuple(probabilities.shape)} '
                f'vs {tuple(targets.shape)}'
            )
        if probabilities.ndim != 2 or probabilities.size(1) != 3:
            raise ValueError('Expected [batch, 3] probabilities: Active, Agonist, Antagonist.')

        eps = torch.finfo(probabilities.dtype).eps
        probs = probabilities.clamp(min=eps, max=1.0 - eps)
        targets = targets.to(dtype=probs.dtype)
        if self.pos_weight is None:
            cls_loss = F.binary_cross_entropy(probs, targets)
        else:
            pos_weight = self.pos_weight.to(dtype=probs.dtype).view(1, -1)
            cls_loss = -(
                pos_weight * targets * torch.log(probs)
                + (1.0 - targets) * torch.log1p(-probs)
            ).mean()

        consistency_loss = probs.new_zeros(())
        for parent_idx, child_idx in self.hierarchy:
            violation = torch.clamp(probs[:, child_idx] - probs[:, parent_idx], min=0.0)
            consistency_loss = consistency_loss + violation.square().sum()
        return cls_loss + self.penalty * consistency_loss

# ==============================================================================
# 2. 特征提取工具
# ==============================================================================
def load_bermol_safe(model_dir):
    possible_names = ["bermol_fixed.pt", "BerMolModel.pkl", "BerMolModel_base.pkl"]
    target_path = None
    for name in possible_names:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            target_path = p; break
    if target_path is None: raise FileNotFoundError(f"BerMol not found in {model_dir}")
    sys.modules['__builtin__'] = builtins
    try:
        predictor = torch.load(target_path, map_location='cpu', weights_only=False)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        try:
            predictor = torch.load(target_path, map_location='cpu')
        except Exception:
            predictor = torch.load(target_path, map_location='cpu', encoding='latin1')
    except Exception:
        predictor = torch.load(target_path, map_location='cpu', encoding='latin1', weights_only=False)
    return predictor


@lru_cache(maxsize=16)
def _path_fingerprint(path):
    """Identify a local model/code tree by canonical path and file content."""
    if not path:
        return None
    resolved_path = os.path.realpath(os.path.abspath(path))
    if not os.path.exists(resolved_path):
        return {'path': resolved_path, 'exists': False}

    entries = []
    if os.path.isfile(resolved_path):
        paths = [(os.path.basename(resolved_path), resolved_path)]
    else:
        paths = []
        for root, dirs, files in os.walk(resolved_path):
            dirs.sort()
            for filename in sorted(files):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, resolved_path)
                paths.append((relative_path, file_path))
    for relative_path, file_path in paths:
        stat = os.stat(file_path)
        file_hash = hashlib.sha256()
        with open(file_path, 'rb') as source_file:
            for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b''):
                file_hash.update(chunk)
        entries.append((relative_path, stat.st_size, stat.st_mtime_ns, file_hash.hexdigest()))
    fingerprint = hashlib.sha256(
        json.dumps(entries, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    ).hexdigest()
    return {'path': resolved_path, 'files': fingerprint}


def _feature_cache_key(df, args):
    columns = df[['SMILES', 'Protein', 'Y']]
    dataset_hash = hashlib.sha256(
        pd.util.hash_pandas_object(columns, index=False).values.tobytes()
    ).hexdigest()
    identity = {
        'schema': 2,
        'dataset': dataset_hash,
        'feature_extractor': _path_fingerprint(__file__),
        'bermol': _path_fingerprint(args.bermol_path),
        'wide_esm': _path_fingerprint(args.esm_path),
        'deep_esm': _path_fingerprint(args.esm2_path),
        'morgan_dim': args.morgan_dim,
        'wide_esm_dim': args.esm_dim,
        'deep_esm_dim': args.esm2_dim,
        'esm_max_length': 1024,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    ).hexdigest()[:20]


def _minimum_random_train_rows(row_count, k_folds):
    largest_test_fold = math.ceil(row_count / k_folds)
    fold_train_rows = row_count - largest_test_fold
    validation_rows = math.ceil(fold_train_rows * 0.1)
    return fold_train_rows - validation_rows

def extract_bermol_features(df, model_dir, cache_path, device):
    if os.path.exists(cache_path):
        print(f"[BerMol] 加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    print(f"[BerMol] 开始提取特征...")
    predictor = load_bermol_safe(model_dir)
    model = predictor.model
    tokenizer = _load_bermol_tokenizer(model_dir)(predictor.vocab)
    model.to(device).eval()
    smiles_list = df['SMILES'].unique().tolist()
    features = {'cls': {}, 'seq': {}}
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="[BerMol Extract]"):
            batch_smiles = smiles_list[i : i + batch_size]
            batch_token_ids, max_len, valid_lens, valid_batch_indices = [], 0, [], []
            for idx, smi in enumerate(batch_smiles):
                try:
                    tokens = tokenizer.encode(smi).squeeze(0).tolist()
                    batch_token_ids.append(tokens)
                    l = len(tokens); max_len = max(max_len, l); valid_lens.append(l)
                    valid_batch_indices.append(idx)
                except: continue
            if not batch_token_ids: continue
            padded_ids = np.zeros((len(batch_token_ids), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(batch_token_ids), max_len, max_len), dtype=np.float32) - 10000.0
            for idx, tokens in enumerate(batch_token_ids):
                l = len(tokens)
                padded_ids[idx, :l] = tokens
                attention_mask[idx, :, :l] = 0.0
            input_ids = torch.LongTensor(padded_ids).to(device)
            input_mask = torch.FloatTensor(attention_mask).to(device)
            seq_out, pooled_out = model.encoder(input_ids, input_mask)
            seq_cpu = seq_out.cpu().numpy()
            pooled_cpu = pooled_out.cpu().numpy()
            for k, original_idx in enumerate(valid_batch_indices):
                smi = batch_smiles[original_idx]
                l = valid_lens[k]
                features['seq'][smi] = seq_cpu[k, :l, :]
                features['cls'][smi] = pooled_cpu[k]
    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    return features

def extract_esm_cls_features(df, esm_model_path, cache_path, device):
    if os.path.exists(cache_path):
        print(f"[ESM-CLS] 加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    print(f"[ESM-CLS] 开始提取 CLS 特征...")
    tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
    model = EsmModel.from_pretrained(esm_model_path).to(device)
    model.eval()
    proteins = df['Protein'].unique().tolist()
    features = {}
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(proteins), batch_size), desc="[ESM CLS Extract]"):
            batch_prot = proteins[i : i + batch_size]
            inputs = tokenizer(batch_prot, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for prot, emb in zip(batch_prot, cls_embeddings):
                features[prot] = emb
    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    return features

def extract_esm2_seq_features(df, esm_path, cache_path, device):
    if os.path.exists(cache_path):
        print(f"[ESM2-Seq] 加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    print(f"[ESM2-Seq] Loading Model from {esm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    model = EsmModel.from_pretrained(esm_path).to(device)
    model.eval()
    proteins = df['Protein'].unique().tolist()
    features = {}
    proteins.sort(key=len)
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(proteins), batch_size), desc="[ESM2 Seq Extract]"):
            batch_prot = proteins[i : i + batch_size]
            inputs = tokenizer(batch_prot, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            last_hidden_cpu = last_hidden_state.cpu().numpy()
            mask_cpu = inputs['attention_mask'].cpu().numpy()
            for k, prot in enumerate(batch_prot):
                valid_len = int(mask_cpu[k].sum())
                if valid_len > 2:
                    emb = last_hidden_cpu[k, 1 : valid_len - 1, :]
                else:
                    emb = last_hidden_cpu[k, 0 : valid_len, :]
                features[prot] = emb
    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features

def extract_morgan_features(df, cache_path, n_bits=1024):
    if os.path.exists(cache_path):
        print(f"[Morgan] 加载缓存: {cache_path}")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    print(f"[Morgan] 开始计算指纹 (n_bits={n_bits})...")
    smiles_list = df['SMILES'].unique().tolist()
    features = {}
    for smi in tqdm(smiles_list, desc="[Morgan Extract]"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                features[smi] = np.array(fp)
        except: continue
    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    return features


# ==============================================================================
# 3. Dataset & Collator
# ==============================================================================
class FusionDataset(Dataset):
    def __init__(self, df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats):
        self.df = df.reset_index(drop=True)
        self.bermol_cls = bermol_feats['cls']
        self.bermol_seq = bermol_feats['seq']
        self.esm_cls_feats = esm_cls_feats
        self.morgan_feats = morgan_feats
        self.esm_seq_feats = esm_seq_feats

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi, prot, y_raw = row['SMILES'], row['Protein'], row['Y']

        if (smi not in self.bermol_cls or
            smi not in self.bermol_seq or
            prot not in self.esm_cls_feats or
            prot not in self.esm_seq_feats or
            smi not in self.morgan_feats):
            return None

        # Features
        b_cls_vec = torch.as_tensor(self.bermol_cls[smi], dtype=torch.float)
        e_cls_vec = torch.as_tensor(self.esm_cls_feats[prot], dtype=torch.float)
        morgan_vec = torch.as_tensor(self.morgan_feats[smi], dtype=torch.float)
        b_seq_vec = torch.as_tensor(self.bermol_seq[smi], dtype=torch.float)
        p_seq_vec = torch.as_tensor(self.esm_seq_feats[prot], dtype=torch.float)

        # --- Label Conversion (Hierarchical) ---
        # y_raw: 0=Inactive, 1=Agonist, 2=Antagonist
        # Target: [Active, Agonist, Antagonist]
        label_vec = np.zeros(3, dtype=np.float32)
        y_int = int(y_raw)

        if y_int == 1:
            label_vec[0] = 1.0 # Active
            label_vec[1] = 1.0 # Agonist
        elif y_int == 2:
            label_vec[0] = 1.0 # Active
            label_vec[2] = 1.0 # Antagonist

        return (b_cls_vec, e_cls_vec, morgan_vec, b_seq_vec, p_seq_vec, torch.tensor(label_vec))

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    b_cls, e_cls, morgans, b_seqs, p_seqs, ys = zip(*batch)

    # Pad Drug
    if any(s.ndim != 2 for s in b_seqs):
        raise ValueError("BerMol sequence features must have shape [tokens, channels].")
    drug_dim = b_seqs[0].size(1)
    if any(s.size(1) != drug_dim for s in b_seqs):
        raise ValueError("BerMol sequence feature widths differ within a batch.")
    max_d = max(s.size(0) for s in b_seqs)
    b_seq_pad = b_seqs[0].new_zeros((len(b_seqs), max_d, drug_dim))
    b_mask = torch.zeros(len(b_seqs), max_d, dtype=torch.float)
    for i, s in enumerate(b_seqs):
        l = s.size(0)
        b_seq_pad[i, :l] = s
        b_mask[i, :l] = 1.0

    # Pad Protein
    if any(p.ndim != 2 for p in p_seqs):
        raise ValueError("Protein sequence features must have shape [tokens, channels].")
    esm_dim = p_seqs[0].size(1)
    if any(p.size(1) != esm_dim for p in p_seqs):
        raise ValueError("Protein sequence feature widths differ within a batch.")
    max_p = max(p.size(0) for p in p_seqs)
    p_seq_pad = p_seqs[0].new_zeros((len(p_seqs), max_p, esm_dim))
    p_mask = torch.zeros(len(p_seqs), max_p, dtype=torch.float)
    for i, p in enumerate(p_seqs):
        l = p.size(0)
        p_seq_pad[i, :l, :] = p
        p_mask[i, :l] = 1.0

    return (torch.stack(b_cls), torch.stack(e_cls), torch.stack(morgans),
            b_seq_pad, b_mask,
            p_seq_pad, p_mask,
            torch.stack(ys))

# ==============================================================================
# 4. Evaluation (Comprehensive Metrics)
# ==============================================================================
def eval_model(model, loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if not batch: continue
            b_c, e_c, m_v, b_s, b_m, p_s, p_m, y = [x.to(device) for x in batch]
            probabilities = model(b_c, e_c, m_v, b_s, b_m, p_s, p_m)
            probs = probabilities.cpu().numpy()
            targets = y.cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

    if not all_probs:
        metrics = {}
        for label in ('Active', 'Agonist', 'Antagonist'):
            metrics[f'{label}_AUC'] = 0.0
            metrics[f'{label}_AUPRC'] = 0.0
            metrics[f'{label}_F1'] = 0.0
        metrics.update({'Mean_AUC': 0.0, 'Mean_AUPRC': 0.0, 'Mean_F1': 0.0, 'Mean_Acc': 0.0})
        return metrics

    all_probs = np.vstack(all_probs)   # [N, 3]
    all_targets = np.vstack(all_targets) # [N, 3]
    all_preds = (all_probs > threshold).astype(int)

    metrics = {}
    label_names = ['Active', 'Agonist', 'Antagonist']

    # Macro storage
    macro_scores = {'AUC': [], 'AUPRC': [], 'F1': [], 'Acc': [], 'Prec': [], 'Rec': []}

    for i in range(3):
        lbl = label_names[i]
        y_true = all_targets[:, i]
        y_score = all_probs[:, i]
        y_pred = all_preds[:, i]

        # Binary Metrics
        try: auc = roc_auc_score(y_true, y_score)
        except ValueError: auc = 0.0 # Handle single class batch

        try: auprc = average_precision_score(y_true, y_score)
        except ValueError: auprc = 0.0

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics[f'{lbl}_AUC'] = auc
        metrics[f'{lbl}_AUPRC'] = auprc
        metrics[f'{lbl}_F1'] = f1

        # Accumulate for Macro Average if class exists
        if len(np.unique(y_true)) > 1:
            macro_scores['AUC'].append(auc)
            macro_scores['AUPRC'].append(auprc)
        macro_scores['F1'].append(f1)
        macro_scores['Acc'].append(acc)

    metrics['Mean_AUC'] = np.mean(macro_scores['AUC']) if macro_scores['AUC'] else 0.0
    metrics['Mean_AUPRC'] = np.mean(macro_scores['AUPRC']) if macro_scores['AUPRC'] else 0.0
    metrics['Mean_F1'] = np.mean(macro_scores['F1'])
    metrics['Mean_Acc'] = np.mean(macro_scores['Acc'])

    return metrics

def run_kfold_experiment(args, strategy, df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats, device):
    print(f"\n### 启动层级实验: {strategy.upper()} Split ###")

    if strategy == 'target':
        subjects = df['Protein'].unique()
        splitter = KFold(n_splits=args.k_folds, shuffle=True, random_state=42).split(subjects)
    else:
        splitter = KFold(n_splits=args.k_folds, shuffle=True, random_state=42).split(df)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splitter):
        print(f"\n>>> [Fold {fold+1}/{args.k_folds}]")

        # Split Data
        if strategy == 'target':
            subjects = df['Protein'].unique()
            train_prots = subjects[train_idx]
            val_test_prots = subjects[test_idx]
            val_prots, test_prots = train_test_split(val_test_prots, test_size=0.5, random_state=42)
            train_df = df[df['Protein'].isin(train_prots)]
            val_df = df[df['Protein'].isin(val_prots)]
            test_df = df[df['Protein'].isin(test_prots)]
        else:
            fold_train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            train_df, val_df = train_test_split(fold_train_df, test_size=0.1, random_state=42)

        train_dataset = FusionDataset(train_df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats)
        if len(train_dataset) < 2:
            raise ValueError(f'{strategy} fold {fold + 1} has fewer than two training rows.')
        # Keep a final singleton out because BatchNorm1d needs at least two rows.
        drop_last = len(train_dataset) > args.batch_size and len(train_dataset) % args.batch_size == 1
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=drop_last, num_workers=args.num_workers)
        val_loader = DataLoader(FusionDataset(val_df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats),
                                batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
        test_loader = DataLoader(FusionDataset(test_df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats),
                                 batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

        # --- 计算 Class Weights (Pos_Weight) ---
        # 统计训练集中 Active, Agonist, Antagonist 的数量
        total_samples = len(train_df)
        n_active = len(train_df[train_df['Y'].isin([1, 2])])
        n_agonist = len(train_df[train_df['Y'] == 1])
        n_antagonist = len(train_df[train_df['Y'] == 2])

        counts = np.array([n_active, n_agonist, n_antagonist])
        # Pos_weight = (Total - Pos) / Pos
        weights = (total_samples - counts) / (counts + 1e-5)
        pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"    Class Weights (Pos_Weight): {weights}")

        # --- 初始化模型 ---
        model = WideDeepHMCNFModel(
            bermol_dim=768,
            esm_dim=args.esm_dim,
            morgan_dim=args.morgan_dim,
            esm2_seq_dim=args.esm2_dim,
            hidden_dim=256,
            interaction_dim=args.interaction_dim,
            interaction_layers=args.interaction_layers,
            num_heads=args.num_heads,
        ).to(device)

        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # 传入 Pos_Weight 处理类别不平衡
        crit = RecursiveHierarchicalLoss(penalty=1e-5, pos_weight=pos_weight).to(device)

        scheduler = ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5, verbose=True)

        best_auc, patience_cnt = -float('inf'), 0
        save_path = os.path.join(args.output_dir, f"best_{strategy}_f{fold+1}.pth")

        for ep in range(args.epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Ep {ep+1}", leave=False, ncols=80)
            for batch in loop:
                if not batch: continue
                b_c, e_c, m_v, b_s, b_m, p_s, p_m, y = [x.to(device) for x in batch]
                opt.zero_grad()
                probabilities = model(b_c, e_c, m_v, b_s, b_m, p_s, p_m)
                loss = crit(probabilities, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Validation
            res = eval_model(model, val_loader, device)
            val_mean_auc = res['Mean_AUC']

            scheduler.step(val_mean_auc)

            if val_mean_auc > best_auc:
                best_auc = val_mean_auc
                torch.save(model.state_dict(), save_path)
                patience_cnt = 0
                print(f"    [Saved] Mean AUC: {val_mean_auc:.4f}, Mean F1: {res['Mean_F1']:.4f}")
                print(f"       Details -> Active AUC: {res['Active_AUC']:.4f}, Ago AUC: {res['Agonist_AUC']:.4f}, Ant AUC: {res['Antagonist_AUC']:.4f}")
            else:
                patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"    [Early Stop] Best Val Mean AUC: {best_auc:.4f}")
                break

        # Testing
        if len(test_df) > 0 and os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))
            res = eval_model(model, test_loader, device)
            res['Fold'] = fold + 1
            fold_results.append(res)
            print(f"    [Test Result] Mean AUC: {res['Mean_AUC']:.4f}, Mean F1: {res['Mean_F1']:.4f}")

    if fold_results:
        df_res = pd.DataFrame(fold_results)
        df_res.to_csv(os.path.join(args.output_dir, f"results_{strategy}.csv"), index=False)
        print("\n=== Final Average Results ===")
        print(df_res.mean(numeric_only=True))

def _validate_feature_width(name, feature_map, expected_width):
    if not feature_map:
        raise ValueError(f"No {name} features were extracted or loaded.")
    sample = np.asarray(next(iter(feature_map.values())))
    actual_width = int(sample.shape[-1])
    if actual_width != expected_width:
        raise ValueError(
            f"{name} feature width is {actual_width}, but the model is configured for "
            f"{expected_width}. Check the ESM checkpoint, CLI dimension, and feature cache."
        )


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.k_folds < 2:
        raise ValueError('--k_folds must be at least 2.')
    if args.batch_size < 2:
        raise ValueError('--batch_size must be at least 2 because the wide branch uses BatchNorm1d.')
    if args.epochs < 1 or args.patience < 1 or args.num_workers < 0:
        raise ValueError('--epochs and --patience must be positive; --num_workers cannot be negative.')
    if args.num_heads <= 0 or args.interaction_dim <= 0 or args.interaction_dim % args.num_heads:
        raise ValueError('--interaction_dim must be positive and divisible by --num_heads.')
    if args.interaction_layers < 0:
        raise ValueError('--interaction_layers cannot be negative; use 0 for no cross-sequence layers.')

    print(f"Reading: {args.data_path}")
    try:
        df = pd.read_csv(args.data_path, sep='\t')
    except UnicodeDecodeError:
        df = pd.read_csv(args.data_path, sep='\t', encoding='utf-16')
    df.rename(columns={'Smiles': 'SMILES', 'Sequence': 'Protein', 'Activity_Label': 'Y'}, inplace=True)
    required_columns = {'SMILES', 'Protein', 'Y'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f'Missing required input columns: {sorted(missing_columns)}')

    # Cleaning
    def desalt_and_canonicalize(smi):
        if not isinstance(smi, str): return None
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return None
            if '.' in smi:
                frags = Chem.GetMolFrags(mol, asMols=True)
                mol = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)[0]
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except: return None

    tqdm.pandas(desc="Cleaning")
    df['SMILES'] = df['SMILES'].progress_apply(desalt_and_canonicalize)
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    valid_proteins = df['Protein'].map(lambda sequence: isinstance(sequence, str) and bool(sequence.strip()))
    if not valid_proteins.all():
        invalid_count = int((~valid_proteins).sum())
        raise ValueError(f'Found {invalid_count} rows with missing, non-text, or empty protein sequences.')
    df['Protein'] = df['Protein'].str.strip()
    df = df.dropna(subset=['SMILES', 'Protein', 'Y']).copy()
    pair_label_counts = df.groupby(['SMILES', 'Protein'], sort=False)['Y'].nunique()
    conflicting_pair_count = int((pair_label_counts > 1).sum())
    if conflicting_pair_count:
        raise ValueError(
            f'Found {conflicting_pair_count} SMILES/Protein pairs with conflicting labels; '
            'resolve them before training.'
        )
    df = df.drop_duplicates(subset=['SMILES', 'Protein']).copy()
    labels = df['Y'].to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Labels must be finite integer values in {0, 1, 2}.')
    df['Y'] = df['Y'].astype(int)
    invalid_labels = sorted(set(df['Y'].unique()) - {0, 1, 2})
    if invalid_labels:
        raise ValueError(f'Labels must be 0, 1, or 2; found: {invalid_labels}')
    if df.empty:
        raise ValueError('No valid rows remain after SMILES and label cleaning.')
    if len(df) < args.k_folds:
        raise ValueError('The cleaned dataset has fewer rows than --k_folds.')
    if df['Protein'].nunique() < 2 * args.k_folds:
        raise ValueError(
            'Target split requires at least two held-out proteins per fold; '
            'increase the number of proteins or reduce --k_folds.'
        )
    if _minimum_random_train_rows(len(df), args.k_folds) < 2:
        raise ValueError('The cleaned dataset is too small for the requested random K-fold train/validation split.')

    # Paths
    cache_key = _feature_cache_key(df, args)
    bermol_cache = os.path.join(args.output_dir, f"bermol_features_{cache_key}.pkl")
    esm_cls_cache = os.path.join(args.output_dir, f"esm_cls_features_{cache_key}.pkl")
    morgan_cache = os.path.join(args.output_dir, f"morgan_features_{args.morgan_dim}_{cache_key}.pkl")
    esm_seq_cache = os.path.join(args.output_dir, f"esm_seq_features_{args.esm2_dim}_{cache_key}.pkl")

    # Extract features and fail early on stale or mismatched caches.
    bermol_feats = extract_bermol_features(df, args.bermol_path, bermol_cache, device)
    esm_cls_feats = extract_esm_cls_features(df, args.esm_path, esm_cls_cache, device)
    morgan_feats = extract_morgan_features(df, morgan_cache, n_bits=args.morgan_dim)
    esm_seq_feats = extract_esm2_seq_features(df, args.esm2_path, esm_seq_cache, device)
    _validate_feature_width('BerMol pooled', bermol_feats['cls'], 768)
    _validate_feature_width('BerMol sequence', bermol_feats['seq'], 768)
    _validate_feature_width('wide ESM pooled', esm_cls_feats, args.esm_dim)
    _validate_feature_width('Morgan', morgan_feats, args.morgan_dim)
    _validate_feature_width('deep ESM sequence', esm_seq_feats, args.esm2_dim)

    # Filter
    valid_smiles = set(bermol_feats['cls']) & set(bermol_feats['seq']) & set(morgan_feats)
    valid_proteins = set(esm_cls_feats) & set(esm_seq_feats)
    df = df[df['SMILES'].isin(valid_smiles) & df['Protein'].isin(valid_proteins)].copy()
    if df.empty:
        raise ValueError('No rows remain after filtering for complete feature coverage.')
    print(f"Valid Data: {len(df)}")
    if len(df) < args.k_folds or df['Protein'].nunique() < 2 * args.k_folds:
        raise ValueError('Feature filtering left too few rows or proteins for the requested K-fold splits.')
    if _minimum_random_train_rows(len(df), args.k_folds) < 2:
        raise ValueError('Feature filtering left too few rows for the random K-fold train/validation split.')

    # Run
    run_kfold_experiment(args, 'target', df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats, device)
    run_kfold_experiment(args, 'random', df, bermol_feats, esm_cls_feats, morgan_feats, esm_seq_feats, device)

    print("\nAll Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--bermol_path", type=str, required=True, help="Directory containing the BerMol package and model weights.")
    parser.add_argument("--esm_path", type=str, required=True)
    parser.add_argument("--esm2_path", type=str, required=True)
    parser.add_argument("--esm_dim", type=int, default=640, help="Width of the pooled wide-branch ESM feature.")
    parser.add_argument("--esm2_dim", type=int, default=2560, help="Width of the deep-branch ESM token features.")

    parser.add_argument("--output_dir", type=str, default="./output_bindti_bermol_morgan_esm650_hierarchical")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--interaction_dim", type=int, default=128)
    parser.add_argument("--interaction_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--morgan_dim", type=int, default=1024)

    args = parser.parse_args()
    main(args)
