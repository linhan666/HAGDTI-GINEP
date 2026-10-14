from typing import Tuple
import torch

def pad_by_batch(
    packed: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded [B, M, D] values and a boolean mask; True marks valid entries.

batch_idx must be sorted by batch ID, with each graph occupying a contiguous
range in packed [N, D]. One-dimensional packed input returns [B, M] values.
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

    cum     = torch.cumsum(counts, dim=0)
    offsets = torch.cat([counts.new_zeros(1), cum[:-1]], dim=0)          # [B]
    pos     = torch.arange(packed.size(0), device=device) - offsets[batch_idx]

    if packed.dim() == 1:
        padded = packed.new_zeros(batch_size, Mmax)
    else:
        padded = packed.new_zeros(batch_size, Mmax, packed.size(-1))

    mask = torch.zeros(batch_size, Mmax, dtype=torch.bool, device=device)
    padded[batch_idx, pos] = packed
    mask[batch_idx, pos]   = True
    return padded, mask
