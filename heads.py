"""Flat and hierarchical classification heads.

Class labels: 0=inactive, 1=agonist, 2=inhibitor.
Hierarchy nodes: 0=active (parent), 1=agonist, 2=inhibitor (children).
Hierarchical labels: inactive=[0,0,0], agonist=[1,1,0], inhibitor=[1,0,1].
MLP returns class logits; hierarchical heads return node logits. Conditional
softmax returns an active logit and subtype logits. Use extract_3class_probs
for normalized class probabilities in label order.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


HIER_LABEL_MATRIX = torch.tensor([[0, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=torch.float32)


def get_hier_labels(class_labels: torch.Tensor) -> torch.Tensor:
    """将 3类标签 [B] 转换为层级二值标签 [B, 3]"""
    matrix = HIER_LABEL_MATRIX.to(class_labels.device)
    return matrix[class_labels]


class MLPHead(nn.Module):
    """Two-layer classifier returning mutually exclusive class logits."""
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:

        return {"head_type": "mlp", "logits": self.net(x)}


class HMCNFHead(nn.Module):
    """Combine global and level-specific logits for active, agonist and inhibitor nodes."""
    def __init__(self, input_dim: int, hierarchy: list = None,
                 beta: float = 0.5, dropout_rate: float = 0.2,
                 hidden_dim: int = 384):
        super().__init__()
        if hierarchy is None:
            hierarchy = [1, 2]
        self.hierarchy   = hierarchy
        self.beta        = beta
        self.total_nodes = sum(hierarchy)


        self.global_l1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.global_l2 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.global_consensus = nn.Sequential(
            nn.Linear(hidden_dim, self.total_nodes),
        )


        self.local_l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hierarchy[0]),
        )

        self.local_l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hierarchy[1]),
        )

    def forward(self, x: torch.Tensor) -> dict:

        g1 = self.global_l1(x)                            # [B, hidden_dim]
        g2 = self.global_l2(torch.cat([g1, x], dim=1))   # [B, hidden_dim]

        logits_glob = self.global_consensus(g2)                    # [B, 3]
        logits_l1   = self.local_l1(g1)                            # [B, 1]
        logits_l2   = self.local_l2(g2)                            # [B, 2]
        logits_loc  = torch.cat([logits_l1, logits_l2], dim=1)     # [B, 3]


        logits = (1 - self.beta) * logits_glob + self.beta * logits_loc  # [B, 3]
        return {"head_type": "hmcnf", "logits": logits}


class GraphConvolution(nn.Module):
    """基础图卷积层（仅权重乘，邻接矩阵在外部通过 einsum 作用）"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self._reset()

    def _reset(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.matmul(x, self.weight)


class HiAGMTPHead(nn.Module):
    """Project fused features onto three hierarchy nodes and propagate through the label graph."""
    def __init__(self, input_dim: int, node_dim: int = 128, device: str = "cuda"):
        super().__init__()
        self.num_nodes  = 3
        self.node_dim   = node_dim


        self.transformation  = nn.Linear(input_dim, self.num_nodes * node_dim)
        self.trans_dropout   = nn.Dropout(0.2)


        self.gcn    = GraphConvolution(node_dim, node_dim)

        self.linear = nn.Linear(node_dim, 1)


        _adj = torch.eye(self.num_nodes)
        _adj[0, 1] = 1.0
        _adj[0, 2] = 1.0
        row_sum    = _adj.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        D          = torch.diag(d_inv_sqrt)
        adj_hat    = D @ _adj @ D
        self.register_buffer("adj", adj_hat)

    def forward(self, x: torch.Tensor) -> dict:

        h = self.trans_dropout(self.transformation(x))        # [B, N*node_dim]
        h = h.view(h.size(0), self.num_nodes, self.node_dim)  # [B, N, node_dim]


        support = self.gcn(h)                                  # [B, N, node_dim]
        output  = torch.einsum("nm,bmd->bnd", self.adj, support)  # [B, N, node_dim]
        if self.gcn.bias is not None:
            output = output + self.gcn.bias
        output  = F.relu(output)

        logits  = self.linear(output).squeeze(-1)             # [B, 3]
        return {"head_type": "hiagm", "logits": logits}


def chmcnn_mcm(p: torch.Tensor) -> torch.Tensor:
    """Enforce P(active) >= P(agonist), P(inhibitor) on [B, 3] node probabilities."""
    mcm_active = torch.maximum(torch.maximum(p[:, 0], p[:, 1]), p[:, 2])
    return torch.stack([mcm_active, p[:, 1], p[:, 2]], dim=1)


def chmcnn_mc_loss(logits: torch.Tensor, hier_lbl: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Hierarchy loss using constrained probabilities for negatives and positive descendants for positives."""
    h = torch.sigmoid(logits)                       # [B, 3]
    mcm = chmcnn_mcm(h)
    pos = h * hier_lbl
    mcm_pos_active = pos[:, [0, 1, 2]].max(dim=1).values
    mcm_pos = torch.stack([mcm_pos_active, h[:, 1], h[:, 2]], dim=1)
    loss = -(hier_lbl * torch.log(mcm_pos.clamp(eps, 1.0))
             + (1.0 - hier_lbl) * torch.log((1.0 - mcm).clamp(eps, 1.0)))
    return loss.mean()


class CHMCNNHead(nn.Module):
    """Return hierarchy-node logits; use MCM for probabilities and chmcnn_mc_loss for training."""
    def __init__(self, input_dim: int, hidden_dim: int = 384, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {"head_type": "chmcnn", "logits": self.net(x)}


class CondSoftmaxHead(nn.Module):
    """Return active_logit [B] and conditional agonist/inhibitor logits [B, 2]."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.active_head = nn.Linear(hidden_dim, 1)
        self.cond_head   = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.shared(x)
        active_logit = self.active_head(h).squeeze(-1)        # [B]
        cond_logits  = self.cond_head(h)                      # [B, 2]
        logits = torch.cat([active_logit.unsqueeze(-1), cond_logits], dim=-1)  # [B, 3]
        return {
            "head_type":   "cond_softmax",
            "active_logit": active_logit,
            "cond_logits":  cond_logits,
            "logits":       logits,
        }


def extract_3class_probs(pred: dict) -> torch.Tensor:
    """Convert head-specific outputs to normalized [inactive, agonist, inhibitor] probabilities."""
    head = pred["head_type"]
    if head == "mlp":
        return F.softmax(pred["logits"].float(), dim=-1)  # [B, 3]

    if head == "cond_softmax":

        p_active = torch.sigmoid(pred["active_logit"].float())          # [B]
        cond     = F.softmax(pred["cond_logits"].float(), dim=-1)       # [B, 2] = [ago, inh]
        p_inactive  = 1.0 - p_active
        p_agonist   = p_active * cond[:, 0]
        p_inhibitor = p_active * cond[:, 1]
        probs3 = torch.stack([p_inactive, p_agonist, p_inhibitor], dim=1)
        return probs3 / (probs3.sum(dim=1, keepdim=True) + 1e-9)


    if head == "chmcnn":
        p = chmcnn_mcm(torch.sigmoid(pred["logits"].float()))  # [B, 3]
    else:
        p = torch.sigmoid(pred["logits"].float())  # [B, 3]

    p_active   = p[:, 0]                            # [B]
    p_ago      = p[:, 1]
    p_inh      = p[:, 2]

    denom      = p_ago + p_inh + 1e-9
    p_inactive  = 1.0 - p_active                    # [B]
    p_agonist   = p_active * p_ago / denom           # [B]
    p_inhibitor = p_active * p_inh / denom           # [B]

    probs3 = torch.stack([p_inactive, p_agonist, p_inhibitor], dim=1)  # [B, 3]

    probs3 = probs3 / (probs3.sum(dim=1, keepdim=True) + 1e-9)
    return probs3
