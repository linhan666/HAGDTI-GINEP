"""Original atom, bond and RW-PE featurization for GIN-EP only."""
from typing import Dict
import numpy as np
import torch

_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "OTHER"]
ATOM_TYPE_VOCAB = {a: i for i, a in enumerate(_ATOM_TYPES)}
ATOM_TYPE_DIM   = len(_ATOM_TYPES)               # 10

DEGREE_DIM        = 7   # 0..6
FORMAL_CHARGE_DIM = 5   # -2,-1,0,1,2
HYBRID_DIM        = 5   # SP, SP2, SP3, SP3D, SP3D2
H_COUNT_DIM       = 5   # 0..4
AROMATIC_DIM      = 1
IN_RING_DIM       = 1

ATOM_FEATURE_DIM = (
    ATOM_TYPE_DIM
    + DEGREE_DIM
    + FORMAL_CHARGE_DIM
    + HYBRID_DIM
    + H_COUNT_DIM
    + AROMATIC_DIM
    + IN_RING_DIM
)  # = 34

# ── 键（bond）特征与随机游走位置编码（轴3 _ep 编码器使用；备份 ligand_graph.py.20260618.bak）──
_BOND_TYPE_VOCAB = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
BOND_TYPE_DIM    = 4
BOND_STEREO_DIM  = 6
BOND_FEATURE_DIM = BOND_TYPE_DIM + 1 + 1 + BOND_STEREO_DIM   # = 12 (type + conjugated + in_ring + stereo)
RW_PE_DIM        = 20                                        # random-walk PE 步数维度

def _one_hot(idx: int, dim: int) -> np.ndarray:
    """安全 one-hot：越界写入 0 向量。"""
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v

def _atom_type_idx(symbol: str) -> int:
    return ATOM_TYPE_VOCAB.get(symbol, ATOM_TYPE_VOCAB["OTHER"])

def _atom_feature_vec(atom) -> np.ndarray:
    """RDKit Atom → 34 维特征向量。"""
    sym = atom.GetSymbol()
    feats = [
        _one_hot(_atom_type_idx(sym), ATOM_TYPE_DIM),
        _one_hot(min(atom.GetDegree(), DEGREE_DIM - 1), DEGREE_DIM),
        _one_hot(int(atom.GetFormalCharge()) + 2, FORMAL_CHARGE_DIM),  # -2..2 → 0..4
        _one_hot(int(atom.GetHybridization()) - 2, HYBRID_DIM),        # RDKit 枚举值偏移（粗略）
        _one_hot(min(atom.GetTotalNumHs(), H_COUNT_DIM - 1), H_COUNT_DIM),
        np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=np.float32),
        np.array([1.0 if atom.IsInRing()      else 0.0], dtype=np.float32),
    ]
    return np.concatenate(feats, axis=0)

def _bond_feature_vec(bond) -> np.ndarray:
    """RDKit Bond → 12 维键特征（type one-hot 4 + conjugated 1 + in_ring 1 + stereo one-hot 6）。"""
    bt_idx = _BOND_TYPE_VOCAB.get(str(bond.GetBondType()), -1)   # 未知类型 → 全零
    feats = [
        _one_hot(bt_idx, BOND_TYPE_DIM),
        np.array([1.0 if bond.GetIsConjugated() else 0.0], dtype=np.float32),
        np.array([1.0 if bond.IsInRing() else 0.0], dtype=np.float32),
        _one_hot(int(bond.GetStereo()), BOND_STEREO_DIM),
    ]
    return np.concatenate(feats, axis=0)

def _random_walk_pe(bond_edge_index: np.ndarray, n_atom: int,
                    walk_length: int = RW_PE_DIM) -> np.ndarray:
    """标准 RWPE（Dwivedi et al.）：每个节点 k=1..walk_length 步随机游走返回自身的概率。

    bond_edge_index: [2, E] 仅键边（双向、无 self-loop）。无边或单原子时返回全零 [N, walk_length]。
    """
    pe = np.zeros((n_atom, walk_length), dtype=np.float32)
    if n_atom == 0 or bond_edge_index.shape[1] == 0:
        return pe
    A = np.zeros((n_atom, n_atom), dtype=np.float64)
    A[bond_edge_index[0], bond_edge_index[1]] = 1.0
    deg  = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / deg, 0.0)
    P    = A * dinv[:, None]                 # 行归一化转移矩阵
    Pk   = P.copy()
    for k in range(walk_length):
        pe[:, k] = np.diagonal(Pk)
        Pk = Pk @ P
    return pe.astype(np.float32)

def _placeholder_atom_graph() -> Dict[str, torch.Tensor]:
    """SMILES 解析失败时返回的最小图（1 个原子，1 条 self-loop）。"""
    feat = np.zeros(ATOM_FEATURE_DIM, dtype=np.float32)
    feat[_atom_type_idx("C")] = 1.0           # 默认碳
    return {
        "atom_features":   torch.from_numpy(feat).unsqueeze(0),       # [1, F]
        "atom_edge_index": torch.tensor([[0], [0]], dtype=torch.long),# [2, 1]
        "atom_types":      torch.tensor([_atom_type_idx("C")], dtype=torch.long),
        "atom_edge_attr":  torch.zeros(1, BOND_FEATURE_DIM, dtype=torch.float32),  # self-loop 全零
        "atom_rw_pe":      torch.zeros(1, RW_PE_DIM, dtype=torch.float32),
    }

def mol_to_atom_graph(smiles: str) -> Dict[str, torch.Tensor]:
    """
    SMILES → 原子图特征。

    返回 dict：
      atom_features    [N_atom, ATOM_FEATURE_DIM]  float32
      atom_edge_index  [2, E_atom]                 long  (无向 + self-loop)
      atom_types       [N_atom]                    long  (原子类型 id，供 substructure 引用)
    """
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError("ligand_graph 需要 RDKit，请 `pip install rdkit-pypi`") from e

    if not isinstance(smiles, str) or not smiles:
        return _placeholder_atom_graph()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return _placeholder_atom_graph()

    n = mol.GetNumAtoms()
    feats = np.stack([_atom_feature_vec(a) for a in mol.GetAtoms()], axis=0)
    atom_types = np.array(
        [_atom_type_idx(a.GetSymbol()) for a in mol.GetAtoms()], dtype=np.int64
    )

    # 无向 edge_index + self-loop；同步构建键特征（self-loop 给全零）
    src, dst, edge_feats = [], [], []
    bond_src, bond_dst = [], []                      # 仅键边（供 RWPE，不含 self-loop）
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_feature_vec(bond)
        src.append(i); dst.append(j); edge_feats.append(bf)
        src.append(j); dst.append(i); edge_feats.append(bf)
        bond_src += [i, j]; bond_dst += [j, i]
    for i in range(n):
        src.append(i); dst.append(i)
        edge_feats.append(np.zeros(BOND_FEATURE_DIM, dtype=np.float32))

    edge_index = np.stack([np.array(src, dtype=np.int64),
                           np.array(dst, dtype=np.int64)], axis=0)
    edge_attr = (np.stack(edge_feats, axis=0)
                 if edge_feats else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32))

    bond_edge_index = (np.stack([np.array(bond_src, dtype=np.int64),
                                 np.array(bond_dst, dtype=np.int64)], axis=0)
                       if bond_src else np.zeros((2, 0), dtype=np.int64))
    rw_pe = _random_walk_pe(bond_edge_index, n, RW_PE_DIM)

    return {
        "atom_features":   torch.from_numpy(feats),
        "atom_edge_index": torch.from_numpy(edge_index),
        "atom_types":      torch.from_numpy(atom_types),
        "atom_edge_attr":  torch.from_numpy(edge_attr.astype(np.float32)),  # [E, 12]
        "atom_rw_pe":      torch.from_numpy(rw_pe),                          # [N, 20]
    }
