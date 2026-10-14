"""
heads.py — GIN-EP 分类头（MLP / HMCN-F / HiAGM-TP / C-HMCNN / conditional-softmax）

层级语义：
  · 标签定义：0=inactive, 1=agonist, 2=inhibitor
  · 层级关系：L1（Active = label>0）→ L2（Agonist=1 / Inhibitor=2）
  · 节点编号：节点0=Active, 节点1=Agonist, 节点2=Inhibitor

输出约定：
  · MLPHead      → logits [B, 3]，配合 CrossEntropyLoss
  · HMCNFHead    → logits [B, 3]（未sigmoid），配合 Hierarchical BCEWithLogitsLoss（autocast安全）
  · HiAGMTPHead  → logits [B, 3]（未sigmoid），配合 Hierarchical BCEWithLogitsLoss

层级标签转换（3类 → 3维二值向量）:
  0 (inactive)  → [0, 0, 0]  active=0, agonist=0, inhibitor=0
  1 (agonist)   → [1, 1, 0]  active=1, agonist=1, inhibitor=0
  2 (inhibitor) → [1, 0, 1]  active=1, agonist=0, inhibitor=1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 层级标签矩阵（行=类别，列=[active, agonist, inhibitor]）
HIER_LABEL_MATRIX = torch.tensor([[0, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=torch.float32)


def get_hier_labels(class_labels: torch.Tensor) -> torch.Tensor:
    """将 3类标签 [B] 转换为层级二值标签 [B, 3]"""
    matrix = HIER_LABEL_MATRIX.to(class_labels.device)
    return matrix[class_labels]


# ─────────────────────────────────────────────────────────────────────────────
# 1. MLP 线性分类头（v7 基线）
# ─────────────────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """
    标准两层 MLP 分类头。

    结构: Linear(input→hidden) → GELU → Dropout → Linear(hidden→num_classes)

    Fix12: 原实现仅为单个 nn.Linear，名不副实。
    现补充隐藏层与激活函数，使其符合标准多层感知机定义。
    hidden_dim 通过 Config.mlp_head_hidden_dim 控制（默认 128）。
    注：model.py 中的 fusion 模块已提供 2 层 MLP，此处为独立分类头。
    输出: logits [B, num_classes]，使用 CrossEntropyLoss。
    """
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x: [B, input_dim]
        return {"head_type": "mlp", "logits": self.net(x)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. HMCN-F 层级分类头
# ─────────────────────────────────────────────────────────────────────────────

class HMCNFHead(nn.Module):
    """
    HMCN-F (Hierarchical Multi-Label Classification Network - Feedforward)

    结构：
      · 全局流 (Global Flow): 带 skip-connection 的逐层 MLP
        x → g1 → [g1; x] → g2 → global_consensus(3节点)
      · 局部流 (Local Flow): 每层独立分支
        g1 → local_l1(1节点: Active)
        g2 → local_l2(2节点: Agonist, Inhibitor)
      · 最终输出: beta 加权融合后的节点 logits [B, 3]
        sigmoid 仅在概率解释或 extract_3class_probs 中使用

    输出: logits [B, 3]（未sigmoid，与 HiAGM 对齐）
    损失: BCEWithLogitsLoss(logits, hier_labels)  — autocast 安全
    """
    def __init__(self, input_dim: int, hierarchy: list = None,
                 beta: float = 0.5, dropout_rate: float = 0.2,
                 hidden_dim: int = 384):
        super().__init__()
        if hierarchy is None:
            hierarchy = [1, 2]  # Level1: Active(1节点), Level2: Agonist+Inhibitor(2节点)
        self.hierarchy   = hierarchy
        self.beta        = beta
        self.total_nodes = sum(hierarchy)  # = 3

        # ── 全局流 ──────────────────────────────────────────────────────
        self.global_l1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Skip connection: concat(g1, x) → g2
        self.global_l2 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # 全局共识：一次性预测所有节点（输出 logits，不加 Sigmoid）
        self.global_consensus = nn.Sequential(
            nn.Linear(hidden_dim, self.total_nodes),
        )

        # ── 局部流 ──────────────────────────────────────────────────────
        # Level 1: Active 节点 (1个)，输出 logits
        self.local_l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hierarchy[0]),
        )
        # Level 2: Agonist + Inhibitor 节点 (2个)，输出 logits
        self.local_l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hierarchy[1]),
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x: [B, input_dim]
        g1 = self.global_l1(x)                            # [B, hidden_dim]
        g2 = self.global_l2(torch.cat([g1, x], dim=1))   # [B, hidden_dim]

        logits_glob = self.global_consensus(g2)                    # [B, 3]
        logits_l1   = self.local_l1(g1)                            # [B, 1]
        logits_l2   = self.local_l2(g2)                            # [B, 2]
        logits_loc  = torch.cat([logits_l1, logits_l2], dim=1)     # [B, 3]

        # beta 加权融合：仍在 logit 空间做线性混合，与原论文等价
        logits = (1 - self.beta) * logits_glob + self.beta * logits_loc  # [B, 3]
        return {"head_type": "hmcnf", "logits": logits}


# ─────────────────────────────────────────────────────────────────────────────
# 3. HiAGM-TP 层级分类头
# ─────────────────────────────────────────────────────────────────────────────

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
        # x: [B, N, in_features] → [B, N, out_features]
        return torch.matmul(x, self.weight)


class HiAGMTPHead(nn.Module):
    """
    HiAGM-TP (Text Propagation Variant) 层级分类头

    结构：
      1. 投影层：将 DTI 融合特征映射到 num_nodes × node_dim 的图节点空间
      2. GCN：通过归一化邻接矩阵传播节点信息
         邻接关系：Active(0) → Agonist(1), Active(0) → Inhibitor(2)
      3. 线性分类器：每个节点输出1维 logit

    输出: logits [B, 3]（未经 sigmoid），使用 BCEWithLogitsLoss
    """
    def __init__(self, input_dim: int, node_dim: int = 128, device: str = "cuda"):
        super().__init__()
        self.num_nodes  = 3  # Active(0), Agonist(1), Inhibitor(2)
        self.node_dim   = node_dim

        # 投影层: [B, input_dim] → [B, num_nodes * node_dim]
        self.transformation  = nn.Linear(input_dim, self.num_nodes * node_dim)
        self.trans_dropout   = nn.Dropout(0.2)

        # GCN
        self.gcn    = GraphConvolution(node_dim, node_dim)
        # 分类器
        self.linear = nn.Linear(node_dim, 1)

        # 归一化邻接矩阵 A_hat = D^-0.5 · A · D^-0.5（含自环）
        # 层级: Active→Agonist, Active→Inhibitor
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
        # x: [B, input_dim]
        h = self.trans_dropout(self.transformation(x))        # [B, N*node_dim]
        h = h.view(h.size(0), self.num_nodes, self.node_dim)  # [B, N, node_dim]

        # GCN: support = h @ W; output = A_hat @ support
        support = self.gcn(h)                                  # [B, N, node_dim]
        output  = torch.einsum("nm,bmd->bnd", self.adj, support)  # [B, N, node_dim]
        if self.gcn.bias is not None:
            output = output + self.gcn.bias
        output  = F.relu(output)

        logits  = self.linear(output).squeeze(-1)             # [B, 3]
        return {"head_type": "hiagm", "logits": logits}


# ─────────────────────────────────────────────────────────────────────────────
# 4. C-HMCNN 层级分类头（轴2，备份 heads.py.20260618.bak）
# ─────────────────────────────────────────────────────────────────────────────

def chmcnn_mcm(p: torch.Tensor) -> torch.Tensor:
    """C-HMCNN Max Constraint Module。

    节点顺序 [Active(0), Agonist(1), Inhibitor(2)]，层级 Active→{Agonist, Inhibitor}。
    保证 P(parent) >= P(child)：父节点输出 = 其子树概率最大值（结构上层级相容）。
    """
    mcm_active = torch.maximum(torch.maximum(p[:, 0], p[:, 1]), p[:, 2])
    return torch.stack([mcm_active, p[:, 1], p[:, 2]], dim=1)


def chmcnn_mc_loss(logits: torch.Tensor, hier_lbl: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """C-HMCNN MCLoss（节点序 [Active, Agonist, Inhibitor]）。

    负样本项用 MCM 相容概率；正样本项对每个节点取其子树内「正标签」节点激活的最大值。
    """
    h = torch.sigmoid(logits)                       # [B, 3]
    mcm = chmcnn_mcm(h)                              # 负样本项用层级相容概率
    pos = h * hier_lbl                              # 非正标签置 0
    mcm_pos_active = pos[:, [0, 1, 2]].max(dim=1).values     # Active 子树 = {0,1,2}
    mcm_pos = torch.stack([mcm_pos_active, h[:, 1], h[:, 2]], dim=1)  # 叶子子树=自身
    loss = -(hier_lbl * torch.log(mcm_pos.clamp(eps, 1.0))
             + (1.0 - hier_lbl) * torch.log((1.0 - mcm).clamp(eps, 1.0)))
    return loss.mean()


class CHMCNNHead(nn.Module):
    """C-HMCNN (Coherent Hierarchical Multi-label Classification Network)。

    输出 raw 节点 logits [B, 3]；推理时经 sigmoid + MCM 保证层级相容。
    损失：train.py 中的 chmcnn_mc_loss（MCLoss）。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 384, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> dict:
        return {"head_type": "chmcnn", "logits": self.net(x)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. 条件 softmax 头（轴2）：P(class)=P(Active)·P(Ago/Inh|Active)
# ─────────────────────────────────────────────────────────────────────────────

class CondSoftmaxHead(nn.Module):
    """条件 softmax 头，与标签语义 inactive/agonist/inhibitor 完全吻合。

    输出:
      active_logit [B]    二分类 Active vs Inactive
      cond_logits  [B, 2] 给定 Active 时 [Agonist, Inhibitor] 的 softmax logits
      logits       [B, 3] 兼容字段 = [active_logit, cond_logits]
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：将任意 head 输出转换为 [B, 3] 归一化概率（用于统一评估）
# ─────────────────────────────────────────────────────────────────────────────

def extract_3class_probs(pred: dict) -> torch.Tensor:
    """
    将不同分类头的原始输出统一转换为 3类互斥概率 [B, 3]。

    · MLP:    softmax(logits)
    · HMCN-F: 条件概率分解
        p_inactive = 1 - P(active)
        p_agonist  = P(active) * P(agonist) / (P(agonist)+P(inhibitor)+ε)
        p_inhibitor= P(active) * P(inhibitor)/ (P(agonist)+P(inhibitor)+ε)
    · HiAGM:  sigmoid(logits) 后同 HMCN-F
    """
    head = pred["head_type"]
    if head == "mlp":
        return F.softmax(pred["logits"].float(), dim=-1)  # [B, 3]

    if head == "cond_softmax":
        # 条件分解：P(Active)·P(Ago/Inh|Active)
        p_active = torch.sigmoid(pred["active_logit"].float())          # [B]
        cond     = F.softmax(pred["cond_logits"].float(), dim=-1)       # [B, 2] = [ago, inh]
        p_inactive  = 1.0 - p_active
        p_agonist   = p_active * cond[:, 0]
        p_inhibitor = p_active * cond[:, 1]
        probs3 = torch.stack([p_inactive, p_agonist, p_inhibitor], dim=1)
        return probs3 / (probs3.sum(dim=1, keepdim=True) + 1e-9)

    # C-HMCNN：先经 MCM 保证层级相容；HMCN-F / HiAGM：直接 sigmoid
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
    # 归一化保证和为1（极端情况保护）
    probs3 = probs3 / (probs3.sum(dim=1, keepdim=True) + 1e-9)
    return probs3
