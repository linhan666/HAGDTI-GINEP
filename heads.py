"""MLP prediction head used by the original GIN-EP arm."""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def extract_3class_probs(pred):
    return F.softmax(pred["logits"].float(), dim=-1)
