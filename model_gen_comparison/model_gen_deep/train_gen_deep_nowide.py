import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import pickle
import argparse
import builtins
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from transformers import AutoTokenizer, EsmModel
except ImportError:
    print("Warning: RDKit or Transformers not found. Feature extraction may fail.")
    pass


from model_gen_deep import FusionDeepDTAGen
from utils import Tokenizer, logging


def desalt_and_canonicalize(smi):
    if not isinstance(smi, str): return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        if '.' in smi:
            frags = Chem.GetMolFrags(mol, asMols=True)
            mol = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)[0]
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None

def load_bermol_safe(model_dir):
    possible_names = ["bermol_fixed.pt", "BerMolModel.pkl", "BerMolModel_base.pkl"]
    target_path = None
    for name in possible_names:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            target_path = p; break
    if target_path is None: raise FileNotFoundError(f"BerMol model not found in {model_dir}")
    sys.modules['__builtin__'] = builtins
    try: predictor = torch.load(target_path, map_location='cpu')
    except: predictor = torch.load(target_path, map_location='cpu', encoding='latin1')
    return predictor

def extract_bermol_features(df, model_dir, cache_path, device):
    """
    提取 BerMol 特征：仅保留 SEQ (Deep部分)，移除 CLS
    """
    if os.path.exists(cache_path):
        print(f"[BerMol] Loading cache from: {cache_path}")
        with open(cache_path, 'rb') as f: return pickle.load(f)

    print(f"[BerMol] Extracting features (Seq only)...")
    if model_dir not in sys.path: sys.path.append(model_dir)
    from bermol.tokenizer import BerMolTokenizer

    predictor = load_bermol_safe(model_dir)
    model = predictor.model
    tokenizer = BerMolTokenizer(predictor.vocab)
    model.to(device).eval()

    smiles_list = df['SMILES'].unique().tolist()
    features = {'seq': {}}
    batch_size = 64

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="[BerMol]"):
            batch_smiles = smiles_list[i : i + batch_size]
            batch_token_ids, valid_indices, max_len = [], [], 0

            for idx, smi in enumerate(batch_smiles):
                try:
                    tokens = tokenizer.encode(smi).squeeze(0).tolist()
                    if len(tokens) > 512: tokens = tokens[:512]
                    batch_token_ids.append(tokens)
                    max_len = max(max_len, len(tokens))
                    valid_indices.append(idx)
                except: continue

            if not batch_token_ids: continue

            padded_ids = np.zeros((len(batch_token_ids), max_len), dtype=np.int64)
            mask = np.zeros((len(batch_token_ids), max_len, max_len), dtype=np.float32) - 1e4
            for k, tokens in enumerate(batch_token_ids):
                l = len(tokens)
                padded_ids[k, :l] = tokens
                mask[k, :, :l] = 0.0

            input_ids = torch.LongTensor(padded_ids).to(device)
            input_mask = torch.FloatTensor(mask).to(device)


            seq_out, _ = model.encoder(input_ids, input_mask)

            seq_cpu = seq_out.cpu().numpy()

            for k, original_idx in enumerate(valid_indices):
                smi = batch_smiles[original_idx]
                features['seq'][smi] = seq_cpu[k, :len(batch_token_ids[k]), :]

    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    return features


def extract_esm_seq_features(df, model_path, cache_path, device):
    if os.path.exists(cache_path):
        print(f"[ESM-Seq] Loading cache from: {cache_path}")
        with open(cache_path, 'rb') as f: return pickle.load(f)
    print(f"[ESM-Seq] Extracting features...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path).to(device)
    model.eval()
    proteins = df['Protein'].unique().tolist()
    features = {}
    proteins.sort(key=len)
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(proteins), batch_size), desc="[ESM-Seq]"):
            batch_prot = proteins[i : i + batch_size]
            inputs = tokenizer(batch_prot, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state.cpu().numpy()
            mask = inputs['attention_mask'].cpu().numpy()
            for k, prot in enumerate(batch_prot):
                valid_len = int(mask[k].sum())

                if valid_len > 2: emb = last_hidden_state[k, 1 : valid_len - 1, :]
                else: emb = last_hidden_state[k, 0 : valid_len, :]
                features[prot] = emb
    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    return features


def ensure_tokenizer(smiles_list, tokenizer_path):
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'rb') as f: return pickle.load(f)
    print(f"Creating new tokenizer...")
    vocabs = Tokenizer.gen_vocabs(smiles_list)
    tokenizer = Tokenizer(vocabs)
    with open(tokenizer_path, 'wb') as f: pickle.dump(tokenizer, f)
    return tokenizer


class BatchData:
    pass

class HybridDataset(Dataset):
    def __init__(self, df, bermol_feats, esm_seq_feats, tokenizer):
        self.df = df.reset_index(drop=True)
        self.bermol_seq = bermol_feats['seq']

        self.esm_seq_feats = esm_seq_feats

        self.tokenizer = tokenizer

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smi, prot, label = row['SMILES'], row['Protein'], row['Y']

        try: affinity = float(row['Affinity'])
        except: affinity = 9.0 if label == 1 else 4.0


        if (smi not in self.bermol_seq or prot not in self.esm_seq_feats):
            return None


        b_seq = torch.tensor(self.bermol_seq[smi], dtype=torch.float)
        e_seq = torch.tensor(self.esm_seq_feats[prot], dtype=torch.float)
        token_ids = torch.tensor(self.tokenizer.parse(smi), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float)
        aff = torch.tensor(affinity, dtype=torch.float)

        return b_seq, e_seq, token_ids, y, aff

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None


    b_seqs, e_seqs, token_ids, ys, affs = zip(*batch)

    MAX_LEN = 768
    b_seqs = [s[:MAX_LEN, :] for s in b_seqs]
    token_ids = [t[:MAX_LEN] for t in token_ids]

    b_seq_pad = pad_sequence(b_seqs, batch_first=True, padding_value=0)
    b_mask = torch.zeros(b_seq_pad.shape[:2])
    for i, seq in enumerate(b_seqs): b_mask[i, :seq.shape[0]] = 1

    e_seq_pad = pad_sequence(e_seqs, batch_first=True, padding_value=0)
    e_mask = torch.zeros(e_seq_pad.shape[:2])
    for i, seq in enumerate(e_seqs): e_mask[i, :seq.shape[0]] = 1

    target_seq_pad = pad_sequence(token_ids, batch_first=True, padding_value=2)

    data = BatchData()

    data.b_seq = b_seq_pad
    data.b_mask = b_mask
    data.e_seq = e_seq_pad
    data.e_mask = e_mask
    data.target_seq = target_seq_pad
    data.y = torch.stack(ys)
    data.affinity = torch.stack(affs)

    return data


def validate(model, loader, device):
    model.eval()
    total_dti_loss, total_gen_loss = 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="Validating", leave=False):
            if data is None: continue


            data.b_seq = data.b_seq.to(device, non_blocking=True)
            data.b_mask = data.b_mask.to(device, non_blocking=True)
            data.e_seq = data.e_seq.to(device, non_blocking=True)
            data.e_mask = data.e_mask.to(device, non_blocking=True)
            data.target_seq = data.target_seq.to(device, non_blocking=True)
            data.y = data.y.to(device, non_blocking=True)
            data.affinity = data.affinity.to(device, non_blocking=True)

            dti_logits, _, dti_loss, lm_loss, _ = model(data, word_dropout=0.0)

            total_dti_loss += dti_loss.item()
            total_gen_loss += lm_loss.item()
            all_probs.extend(torch.sigmoid(dti_logits).cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_dti = total_dti_loss / len(loader)
    avg_gen = total_gen_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
    except: auc, acc = 0, 0
    return avg_dti, avg_gen, auc, acc

def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)


    print(f"Reading data from {args.data_path}...")
    try: df = pd.read_csv(args.data_path, sep='\t' if args.data_path.endswith('.txt') else ',')
    except: df = pd.read_csv(args.data_path, sep='\t', encoding='utf-16')

    col_map = {'Smiles': 'SMILES', 'Sequence': 'Protein', 'Activity_Label': 'Y', 'affinity': 'Affinity'}
    df.rename(columns=col_map, inplace=True)

    tqdm.pandas(desc="Cleaning Data")
    df['SMILES'] = df['SMILES'].progress_apply(desalt_and_canonicalize)
    df = df.dropna(subset=['SMILES', 'Protein', 'Y']).drop_duplicates(subset=['SMILES', 'Protein'])
    df['Y'] = df['Y'].astype(int)
    print(f"Cleaned Data Size: {len(df)}")


    bermol_cache = os.path.join(args.output_dir, "bermol_seq_features.pkl")
    esm_seq_cache = os.path.join(args.output_dir, "esm_seq_features.pkl")
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.pkl")


    tokenizer = ensure_tokenizer(df['SMILES'].tolist(), tokenizer_path)
    bermol_feats = extract_bermol_features(df, args.bermol_path, bermol_cache, device)
    esm_seq_feats = extract_esm_seq_features(df, args.esm_seq_path, esm_seq_cache, device)


    dataset = HybridDataset(df, bermol_feats, esm_seq_feats, tokenizer)
    train_size = int(0.9 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)


    sample_b_seq = next(iter(bermol_feats['seq'].values()))
    sample_e_s = next(iter(esm_seq_feats.values()))

    model = FusionDeepDTAGen(
        tokenizer,
        bermol_dim=sample_b_seq.shape[-1],
        esm_seq_dim=sample_e_s.shape[-1]

    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Start Training (AMP: {amp_dtype})...")

    best_val_auc = 0.0
    global_step = 0
    total_steps = args.epochs * len(train_loader)
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        train_dti, train_gen = 0, 0

        for batch_idx, data in enumerate(loop):
            if data is None: continue


            data.b_seq = data.b_seq.to(device, non_blocking=True)
            data.b_mask = data.b_mask.to(device, non_blocking=True)
            data.e_seq = data.e_seq.to(device, non_blocking=True)
            data.e_mask = data.e_mask.to(device, non_blocking=True)
            data.target_seq = data.target_seq.to(device, non_blocking=True)
            data.y = data.y.to(device, non_blocking=True)
            data.affinity = data.affinity.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            warmup_steps = int(0.2 * total_steps)
            kl_weight = min(0.001, 0.001 * (global_step / warmup_steps)) if global_step < warmup_steps else 0.001

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                dti_logits, _, dti_loss, lm_loss, kl_loss = model(data, word_dropout=args.word_dropout)
                total_loss = kl_loss * kl_weight + dti_loss + lm_loss

            if torch.isnan(total_loss):
                print(f"NaN Loss. Skipping.")
                optimizer.zero_grad()
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_dti += dti_loss.item()
            train_gen += lm_loss.item()
            global_step += 1
            if batch_idx % 10 == 0:
                loop.set_postfix(DTI=dti_loss.item(), Gen=lm_loss.item(), KL=f"{kl_loss.item():.1e}")

        val_dti, val_gen, val_auc, val_acc = validate(model, test_loader, device)
        print(f"\nEp {epoch+1}: Train DTI {train_dti/len(train_loader):.4f} | Val AUC {val_auc:.4f} | Val ACC {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print("  >>> Best Model Saved")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early Stopping!")
                break

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_fusion_final_deep_nowide")
    parser.add_argument("--bermol_path", type=str, required=True)
    parser.add_argument("--esm_seq_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--word_dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    return parser


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
