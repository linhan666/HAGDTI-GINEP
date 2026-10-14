"""Complete GIN-EP model; alternate architectures have been removed."""
import glob
import json
import logging
import os
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from data import PHARMACOPHORE_DIM, MOTIF_FEATURE_DIM
from heads import MLPHead
from ligand_graph import ATOM_FEATURE_DIM
from graph_deep_encoder import GraphDeepEncoder
logger = logging.getLogger(__name__)

class _ESMCOutput:
    """模拟 HuggingFace BaseModelOutput，提供 .last_hidden_state 属性。"""
    __slots__ = ('last_hidden_state',)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

class ESMCBackboneWrapper(nn.Module):
    """
    将 ESMC 模型（EvolutionaryScale esm 包）包装为 HuggingFace 风格接口。

    输入: input_ids [B, L]  (来自 EsmSequenceTokenizer)
          attention_mask [B, L]  (ESMC 内部通过 token ID 判断 pad，
                                  此参数保留以兼容调用方接口，不显式传入 ESMC)
    输出: _ESMCOutput，其 .last_hidden_state = [B, L, d_model]
    """

    def __init__(self, esmc_model: nn.Module):
        super().__init__()
        self.esmc = esmc_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor=None, **kwargs) -> _ESMCOutput:
        out = self.esmc(sequence_tokens=input_ids)
        embeddings = out.embeddings
        return _ESMCOutput(embeddings)

    @property
    def transformer(self):
        return getattr(self.esmc, 'transformer', None)

def _find_model_cfg_value(model_cfg: Dict, key: str):
    candidates = [model_cfg, model_cfg.get('model', {}) if isinstance(model_cfg.get('model'), dict) else {}, model_cfg.get('model_config', {}) if isinstance(model_cfg.get('model_config'), dict) else {}, model_cfg.get('architecture', {}) if isinstance(model_cfg.get('architecture'), dict) else {}]
    for cfg in candidates:
        if key in cfg:
            return cfg[key]
    return None

def _infer_esmc_architecture(model_cfg: Dict, local_path: str, pth_files) -> Dict[str, int]:
    path_hint = ' '.join([local_path, *pth_files]).lower()
    size_hint = str(model_cfg.get('model_size') or model_cfg.get('size') or model_cfg.get('name') or path_hint).lower()
    if '600' in size_hint:
        defaults = dict(d_model=1152, n_heads=18, n_layers=36)
    else:
        defaults = dict(d_model=960, n_heads=15, n_layers=30)
    return {'d_model': int(_find_model_cfg_value(model_cfg, 'd_model') or defaults['d_model']), 'n_heads': int(_find_model_cfg_value(model_cfg, 'n_heads') or defaults['n_heads']), 'n_layers': int(_find_model_cfg_value(model_cfg, 'n_layers') or defaults['n_layers'])}

def _validate_esmc_architecture(arch: Dict[str, int]) -> None:
    d_model = arch['d_model']
    n_heads = arch['n_heads']
    n_layers = arch['n_layers']
    if not 128 <= d_model <= 4096:
        raise ValueError(f'[ESMC] 非法 d_model: {d_model}')
    if not 1 <= n_heads <= 128:
        raise ValueError(f'[ESMC] 非法 n_heads: {n_heads}')
    if not 1 <= n_layers <= 256:
        raise ValueError(f'[ESMC] 非法 n_layers: {n_layers}')
    if d_model % n_heads != 0:
        raise ValueError(f'[ESMC] d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除')

def load_esmc_model(local_path: str) -> ESMCBackboneWrapper:
    """
    从本地目录加载 ESMC 模型并返回包装后的 ESMCBackboneWrapper。

    加载策略（按顺序尝试）：
      1. ESMC.from_pretrained(local_path)   — esm SDK 原生目录加载
      2. 手动读取 config.json + *.pth       — 备用方案

    参数:
        local_path: 包含 config.json 和 *.pth 文件的本地目录（如 ~/LH/PLM/ESMC_300M）
    """
    from esm.models.esmc import ESMC
    from esmc_tokenizer import load_esmc_tokenizer
    local_path = os.path.expanduser(local_path)
    try:
        model = ESMC.from_pretrained(local_path)
        logger.info('[ESMC] 从本地 registry 加载成功: %s', local_path)
        return ESMCBackboneWrapper(model)
    except Exception as e:
        logger.warning('[ESMC] from_pretrained(%s) 失败: %s，尝试手动加载…', local_path, e)
    config_path = os.path.join(local_path, 'config.json')
    pth_files = glob.glob(os.path.join(local_path, '*.pth'))
    if not pth_files:
        raise FileNotFoundError(f'[ESMC] 在 {local_path} 中未找到 .pth 权重文件。请确认路径正确，或检查 esm 包是否支持该目录格式。')
    with open(config_path) as f:
        model_cfg = json.load(f)
    arch = _infer_esmc_architecture(model_cfg, local_path, pth_files)
    _validate_esmc_architecture(arch)
    model = ESMC(d_model=arch['d_model'], n_heads=arch['n_heads'], n_layers=arch['n_layers'], tokenizer=load_esmc_tokenizer())
    state = torch.load(pth_files[0], map_location='cpu', weights_only=True)
    for key in ('model', 'state_dict', 'module'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=True)
    logger.info('[ESMC] 手动加载成功: %s', pth_files[0])
    return ESMCBackboneWrapper(model)

def freeze_encoder(model: nn.Module, n: int):
    """
    冻结编码器全部参数，然后解冻最后 n 层（n=0 则全部冻结）。

    支持的层路径：
      · model.layers                            — ESM2 (HuggingFace)
      · model.encoder.layer                     — BERT 风格
      · model.esmc.transformer.blocks           — ESMC (ESMCBackboneWrapper)
      · model.transformer.blocks                — ESMC 直接访问
    """
    for p in model.parameters():
        p.requires_grad = False
    if n <= 0:
        return
    layers = None
    if hasattr(model, 'layers'):
        layers = list(model.layers)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = list(model.encoder.layer)
    elif hasattr(model, 'esmc'):
        trans = getattr(model.esmc, 'transformer', None)
        if trans is not None and hasattr(trans, 'blocks'):
            layers = list(trans.blocks)
    elif hasattr(model, 'transformer'):
        trans = model.transformer
        if hasattr(trans, 'blocks'):
            layers = list(trans.blocks)
    if layers:
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif n > 0:
        logger.warning('freeze_encoder: 未找到可解冻的 transformer 层，unfreeze_last_n_layers=%d', n)
    norm_attrs = ['emb_layer_norm_after', 'layer_norm', 'final_layer_norm', 'norm']
    search_roots = [model]
    if hasattr(model, 'esmc'):
        search_roots.append(model.esmc)
        if hasattr(model.esmc, 'transformer'):
            search_roots.append(model.esmc.transformer)
    for root in search_roots:
        for attr in norm_attrs:
            if hasattr(root, attr):
                for p in getattr(root, attr).parameters():
                    p.requires_grad = True

class TMSegmentEncoder(nn.Module):

    def __init__(self, backbone: nn.Module, cfg: Config):
        super().__init__()
        self.backbone = backbone
        self.seg_proj = nn.Linear(cfg.protein_hidden, cfg.fusion_dim)
        self.tm_pos_bias = nn.Embedding(7, cfg.fusion_dim)
        self.tok_proj = nn.Linear(cfg.protein_hidden, cfg.fusion_dim)
        bias = torch.tensor(cfg.TM_INIT_BIAS, dtype=torch.float32)
        with torch.no_grad():
            norms = self.tm_pos_bias.weight.norm(dim=1, keepdim=True)
            scale = (bias * 7).unsqueeze(1)
            self.tm_pos_bias.weight.data = self.tm_pos_bias.weight.data / (norms + 1e-08) * scale * norms.mean()
        self.norm = nn.LayerNorm(cfg.fusion_dim)
        self.tok_norm = nn.LayerNorm(cfg.fusion_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def project_cls(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """Project precomputed TM CLS states while preserving the TM prior."""
        B, N, _ = cls_hidden.shape
        dev = cls_hidden.device
        proj = self.seg_proj(cls_hidden)
        pos = self.tm_pos_bias(torch.arange(N, device=dev).unsqueeze(0))
        return self.drop(self.norm(proj + pos))

    def forward(self, tm_ids: torch.Tensor, tm_masks: torch.Tensor, precomp_hidden: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if precomp_hidden is not None:
            hidden_view = precomp_hidden
            B, N, L, _ = hidden_view.shape
            dev = hidden_view.device
        else:
            B, N, L = tm_ids.shape
            dev = tm_ids.device
            raw = self.backbone(input_ids=tm_ids.view(B * N, L), attention_mask=tm_masks.view(B * N, L)).last_hidden_state
            hidden_view = raw.view(B, N, L, -1)
        cls = hidden_view[:, :, 0, :]
        cls_repr = self.project_cls(cls)
        token_seqs = self.tok_norm(self.tok_proj(hidden_view))
        return (cls_repr, token_seqs)

class ProteinEncoder(nn.Module):
    """完整蛋白编码器：三路 ESMC + BW 基序门控融合"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.protein_token_source = getattr(cfg, 'protein_token_source', 'tm7')
        if self.protein_token_source != 'tm7':
            raise ValueError(f"protein_token_source 必须是 'tm7' 或 'full_sequence'，实际为 {self.protein_token_source!r}")
        self.backbone = load_esmc_model(cfg.esmc_local_path)
        freeze_encoder(self.backbone, cfg.unfreeze_last_n_layers)
        self.tm_enc = TMSegmentEncoder(self.backbone, cfg)
        self.full_proj = nn.Linear(cfg.protein_hidden, cfg.fusion_dim)
        self.pocket_proj = nn.Linear(cfg.protein_hidden, cfg.fusion_dim)
        self._emb_cache: Optional[Dict] = None
        self.motif_enc = nn.Sequential(nn.Linear(MOTIF_FEATURE_DIM, 32), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(32, cfg.fusion_dim), nn.LayerNorm(cfg.fusion_dim))
        self.motif_gate = nn.Sequential(nn.Linear(cfg.fusion_dim * 2, cfg.fusion_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(cfg.fusion_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def set_embedding_cache(self, embeds: Dict, device: str) -> None:
        """Register frozen ESMC outputs on CPU for batch-wise device transfer.

        tm7 keys: full_cls, pocket_cls, tm_hidden.
        full_sequence keys: full_cls, full_hidden, pocket_cls, tm_cls_hidden.
        """
        required = {'full_cls', 'pocket_cls', 'tm_hidden'}
        missing = required.difference(embeds)
        if missing:
            raise KeyError(f'ESMC cache 缺少 {self.protein_token_source} 所需字段: {sorted(missing)}')
        del device
        self._emb_cache = {k: v.detach().cpu() for k, v in embeds.items()}

    def clear_embedding_cache(self) -> None:
        self._emb_cache = None

    @staticmethod
    def _cached_batch(cache_tensor: torch.Tensor, prot_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        cpu_idx = prot_idx.detach().to(device='cpu', dtype=torch.long)
        batch = cache_tensor.index_select(0, cpu_idx)
        return batch.to(device, non_blocking=device.type == 'cuda')

    def forward(self, full_ids, full_mask, pocket_ids, pocket_mask, tm_ids, tm_masks, motif_feat: torch.Tensor, prot_idx: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._emb_cache is not None and prot_idx is not None:
            dev = full_ids.device
            full_cls = self._cached_batch(self._emb_cache['full_cls'], prot_idx, dev)
            pocket_cls = self._cached_batch(self._emb_cache['pocket_cls'], prot_idx, dev)
            tm_hidden = self._cached_batch(self._emb_cache['tm_hidden'], prot_idx, dev)
            cls_repr, interaction_tokens = self.tm_enc(tm_ids, tm_masks, precomp_hidden=tm_hidden)
            interaction_masks = tm_masks
        else:
            full_hidden = self.backbone(input_ids=full_ids, attention_mask=full_mask).last_hidden_state
            full_cls = full_hidden[:, 0, :]
            pocket_cls = self.backbone(input_ids=pocket_ids, attention_mask=pocket_mask).last_hidden_state[:, 0, :]
            cls_repr, tm_tokens = self.tm_enc(tm_ids, tm_masks)
            interaction_tokens = tm_tokens
            interaction_masks = tm_masks
        tm_agg = cls_repr.mean(dim=1)
        seq_repr = self.norm(self.full_proj(full_cls) + 1.5 * self.pocket_proj(pocket_cls) + tm_agg)
        motif_r = self.motif_enc(motif_feat)
        gate = self.motif_gate(torch.cat([seq_repr, motif_r], dim=-1))
        prot = self.drop(seq_repr + gate * motif_r)
        return (prot, cls_repr, interaction_tokens, interaction_masks)

class LigandEncoder(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(cfg.ligand_encoder)
        freeze_encoder(self.backbone, cfg.unfreeze_last_n_layers)
        self.smiles_proj = nn.Linear(cfg.ligand_hidden, cfg.fusion_dim)
        self.tok_proj = nn.Linear(cfg.ligand_hidden, cfg.fusion_dim)
        self.pharm_enc = nn.Sequential(nn.Linear(PHARMACOPHORE_DIM, 64), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(64, cfg.fusion_dim), nn.LayerNorm(cfg.fusion_dim))
        self.fuse = nn.Sequential(nn.Linear(cfg.fusion_dim * 2, cfg.fusion_dim), nn.GELU(), nn.LayerNorm(cfg.fusion_dim))
        self.tok_norm = nn.LayerNorm(cfg.fusion_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, lig_ids, lig_mask, pharm_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(input_ids=lig_ids, attention_mask=lig_mask).last_hidden_state
        cls = hidden[:, 0, :]
        lig_cls = self.drop(self.fuse(torch.cat([self.smiles_proj(cls), self.pharm_enc(pharm_feat)], dim=-1)))
        lig_toks = self.tok_norm(self.tok_proj(hidden))
        return (lig_cls, lig_toks)

class MultiHeadFinegrainedCrossAttention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        fd = cfg.fusion_dim
        heads = cfg.cross_attn_heads
        self.heads = heads
        self.d_head = fd // heads
        self.scale = self.d_head ** (-0.5)
        self.W_q = nn.Linear(fd, fd, bias=False)
        self.W_k = nn.Linear(fd, fd, bias=False)
        self.W_v = nn.Linear(fd, fd, bias=False)
        self.W_o = nn.Linear(fd, fd)
        self.attn_drop = nn.Dropout(cfg.cross_attn_dropout)
        self.norm = nn.LayerNorm(fd)
        self.drop = nn.Dropout(cfg.dropout)
        self.pool_proj = nn.Sequential(nn.Linear(fd * 2, fd), nn.GELU(), nn.LayerNorm(fd))
        self.use_gated = bool(getattr(cfg, 'graph_xattn_gated', False))
        self.head_gate = nn.Linear(fd, heads) if self.use_gated else None

    @staticmethod
    def _make_kv_mask(protein_masks: torch.Tensor) -> torch.Tensor:
        B, N, L = protein_masks.shape
        return protein_masks.view(B, N * L) == 0

    def forward(self, lig_toks: torch.Tensor, token_seqs: torch.Tensor, protein_masks: torch.Tensor, lig_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_smi, fd = lig_toks.shape
        N, L_tm = (token_seqs.shape[1], token_seqs.shape[2])
        L_kv = N * L_tm
        kv_seq = token_seqs.view(B, L_kv, fd)
        Q = self.W_q(lig_toks)
        K = self.W_k(kv_seq)
        V = self.W_v(kv_seq)

        def split_heads(x, s):
            return x.view(B, s, self.heads, self.d_head).transpose(1, 2)
        Q = split_heads(Q, L_smi)
        K = split_heads(K, L_kv)
        V = split_heads(V, L_kv)
        scores = Q @ K.transpose(-2, -1) * self.scale
        kv_pad = self._make_kv_mask(protein_masks)
        scores = scores.masked_fill(kv_pad[:, None, None, :], float('-inf'))
        attn_map = F.softmax(scores, dim=-1)
        attn_map = torch.nan_to_num(attn_map, nan=0.0)
        attn_map = self.attn_drop(attn_map)
        out = attn_map @ V
        out = out.transpose(1, 2).contiguous().view(B, L_smi, fd)
        out = self.drop(self.W_o(out))
        lig_real = lig_mask.bool()
        out_masked = out * lig_real.unsqueeze(-1).float()
        ctx_max = out_masked.masked_fill(~lig_real.unsqueeze(-1), float('-inf')).max(dim=1).values
        ctx_max = torch.nan_to_num(ctx_max, nan=0.0)
        n_real = lig_real.float().sum(dim=1, keepdim=True).clamp(min=1)
        ctx_mean = out_masked.sum(dim=1) / n_real
        ctx = self.pool_proj(torch.cat([ctx_max, ctx_mean], dim=-1))
        ctx = self.norm(ctx)
        return (ctx, attn_map)

def collapse_attn_to_tm7(attn, n_tm=7, L_tm=None):
    import numpy as np
    if attn.ndim == 1:
        return attn
    heads, L_smi, L_kv = attn.shape
    if L_tm is None:
        L_tm = L_kv // n_tm
    step1 = attn.mean(axis=0)
    step2 = step1.mean(axis=0)
    step3 = step2.reshape(n_tm, L_tm)
    return step3.mean(axis=1)


class ClassAGPCRDTIModel(nn.Module):
    """Complete original GIN-EP arm: ESMC/GPCR + wide ligand + graph attention + MLP."""
    def __init__(self, cfg):
        super().__init__()
        if (cfg.graph_deep_type, cfg.head_type, cfg.fusion_mode, cfg.protein_token_source) != ("gin_ep", "mlp", "flat", "tm7"):
            raise ValueError("Only the original gin_ep/mlp/flat/tm7 architecture is supported")
        self.cfg = cfg
        fd = cfg.fusion_dim
        self.prot_enc = ProteinEncoder(cfg)
        self.lig_enc = LigandEncoder(cfg)
        self.graph_deep_enc = GraphDeepEncoder(cfg, atom_in_dim=ATOM_FEATURE_DIM)
        self.cross_attn = MultiHeadFinegrainedCrossAttention(cfg)
        self.fusion = nn.Sequential(
            nn.Linear(fd * 3, fd * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(fd * 2, fd), nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.head = MLPHead(fd, cfg.num_classes, cfg.mlp_head_hidden_dim)
        self.binary_head = nn.Linear(fd, 2) if cfg.use_binary_aux else None

    def forward(self, batch):
        prot, _, tm_tokens, tm_masks = self.prot_enc(
            batch["full_ids"], batch["full_mask"], batch["pocket_ids"], batch["pocket_mask"],
            batch["tm_ids"], batch["tm_masks"], batch["motif_feat"], prot_idx=batch.get("prot_idx"),
        )
        lig, _ = self.lig_enc(batch["lig_ids"], batch["lig_mask"], batch["pharm_feat"])
        graph_tokens, graph_mask = self.graph_deep_enc(batch)
        ctx, attn = self.cross_attn(graph_tokens, tm_tokens, tm_masks, graph_mask)
        h = self.fusion(torch.cat([prot, lig, ctx], dim=-1))
        out = {**self.head(h), "repr": h, "protein_attn": attn, "tm_attn": attn}
        if self.binary_head is not None:
            out["binary_logits"] = self.binary_head(h)
        return out
