"""GIN-EP graph branch only: GINE + bond features + random-walk PE."""
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from tensor_utils import pad_by_batch
from ligand_graph import BOND_FEATURE_DIM

def _require_gine():
    try:
        from torch_geometric.nn import GINEConv
    except ImportError as e:
        raise ImportError("_ep 编码器需要 torch_geometric (GINEConv)。") from e
    return GINEConv

class _EPInput(nn.Module):
    """把 atom_features 与 RW-PE 融合成节点输入；PE 关闭时等价于仅 atom_features。"""

    def __init__(self, atom_in_dim: int, use_pe: bool, pe_walk: int, pe_dim: int):
        super().__init__()
        self.use_pe = use_pe
        if use_pe:
            self.pe_norm = nn.BatchNorm1d(pe_walk)
            self.pe_lin  = nn.Linear(pe_walk, pe_dim)
            self.out_dim = atom_in_dim + pe_dim
        else:
            self.out_dim = atom_in_dim

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["atom_features"]
        if not self.use_pe:
            return x
        pe = batch.get("atom_rw_pe")
        if pe is None:
            return torch.cat([x, x.new_zeros(x.size(0), self.pe_lin.out_features)], dim=1)
        return torch.cat([x, self.pe_lin(self.pe_norm(pe))], dim=1)

def _ep_switches(cfg):
    return (
        bool(getattr(cfg, "graph_use_edge_attr", True)),
        bool(getattr(cfg, "graph_use_rw_pe", True)),
        int(getattr(cfg, "graph_rw_pe_walk", 20)),
        int(getattr(cfg, "graph_rw_pe_dim", 8)),
    )

class GINEdgePEAtomEncoder(nn.Module):
    """gin_ep：输入投影 → 3×GINEConv（键感知）+ RW-PE。"""

    def __init__(self, atom_in_dim: int, cfg):
        super().__init__()
        GINEConv = _require_gine()
        dim = getattr(cfg, "graph_gin_dim", 32)
        use_edge, use_pe, pe_walk, pe_dim = _ep_switches(cfg)
        self.use_edge = use_edge
        self.inp = _EPInput(atom_in_dim, use_pe, pe_walk, pe_dim)
        self.node_proj = nn.Linear(self.inp.out_dim, dim)
        self.edge_emb  = nn.Linear(BOND_FEATURE_DIM, dim)
        self.conv1 = GINEConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))); self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = GINEConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))); self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = GINEConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))); self.bn3 = nn.BatchNorm1d(dim)
        self.out_dim = dim

    def _edge(self, batch: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        ei = batch["atom_edge_index"]
        ea = batch.get("atom_edge_attr") if self.use_edge else None
        if ea is not None:
            return self.edge_emb(ea)
        return x.new_zeros(ei.size(1), x.size(1))

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x  = self.node_proj(self.inp(batch))
        ei = batch["atom_edge_index"]
        e  = self._edge(batch, x)
        x = F.relu(self.conv1(x, ei, e)); x = self.bn1(x)
        x = F.relu(self.conv2(x, ei, e)); x = self.bn2(x)
        x = F.relu(self.conv3(x, ei, e)); x = self.bn3(x)
        return x

class GraphDeepEncoder(nn.Module):
    def __init__(self, cfg, atom_in_dim):
        super().__init__()
        if cfg.graph_deep_type != "gin_ep":
            raise ValueError("Only gin_ep is supported")
        self.encoder = GINEdgePEAtomEncoder(atom_in_dim, cfg)
        self.token_proj = nn.Linear(self.encoder.out_dim, cfg.fusion_dim)

    def forward(self, batch):
        node_h = self.token_proj(self.encoder(batch))
        return pad_by_batch(node_h, batch["batch_atom"], batch["lig_ids"].size(0))
