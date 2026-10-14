from typing import Tuple
import torch

def pad_by_batch(
    packed: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 packed 表示 [N_total, D] 按 batch_idx 还原为 padded 表示 [B, Mmax, D]。

    要求 batch_idx 已按 batch 顺序排序（dataset collate 自然满足）。

    返回：
        padded [B, Mmax, D]
        mask   [B, Mmax]    bool, True = 真实 token
    """
    device = packed.device
    if batch_idx.numel() == 0:
        D = packed.size(-1) if packed.dim() > 1 else 1
        return (
            packed.new_zeros(batch_size, 0, D) if packed.dim() > 1
            else packed.new_zeros(batch_size, 0),
            torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        )

    counts  = torch.bincount(batch_idx, minlength=batch_size)            # [B]
    Mmax    = int(counts.max().item())
    # 起始偏移
    cum     = torch.cumsum(counts, dim=0)
    offsets = torch.cat([counts.new_zeros(1), cum[:-1]], dim=0)          # [B]
    pos     = torch.arange(packed.size(0), device=device) - offsets[batch_idx]  # 在所属 batch 内的位置

    if packed.dim() == 1:
        padded = packed.new_zeros(batch_size, Mmax)
    else:
        padded = packed.new_zeros(batch_size, Mmax, packed.size(-1))

    mask = torch.zeros(batch_size, Mmax, dtype=torch.bool, device=device)
    padded[batch_idx, pos] = packed
    mask[batch_idx, pos]   = True
    return padded, mask
