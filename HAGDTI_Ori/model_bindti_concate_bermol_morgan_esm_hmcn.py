# ==============================================================================
# 文件名: model_bindti_hmcnf.py
# 描述: Wide & Deep + ESM2 + HMCN-F (层级分类头)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 基础组件 (Intention, BiIntention, FeatureFusion) - 保持不变
# ------------------------------------------------------------------------------
class Intention(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super(Intention, self).__init__()
        if num_heads <= 0 or dim <= 0 or dim % num_heads:
            raise ValueError('dim must be positive and divisible by num_heads.')
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.raw_alpha = nn.Parameter(torch.zeros(1))
        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)
        self.softmax = nn.Softmax(dim=-2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, query=None, key_mask=None, query_mask=None):
        if query is None: query = x
        q_out, k_out, v_out = self.wq(query), self.wk(x), self.wv(x)
        b, n, c = x.shape
        b_q, n_q, c_q = query.shape
        key = k_out.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        value = v_out.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        query_reshaped = q_out.reshape(b_q, n_q, self.head, self.head_dim).permute(0, 2, 1, 3)
        if key_mask is not None:
            if key_mask.shape != (b, n):
                raise ValueError(f'key_mask must have shape {(b, n)}.')
            valid_keys = key_mask.to(device=x.device).bool()[:, None, :, None]
            key = key * valid_keys.to(dtype=key.dtype)
            value = value * valid_keys.to(dtype=value.dtype)
        if query_mask is not None and query_mask.shape != (b_q, n_q):
            raise ValueError(f'query_mask must have shape {(b_q, n_q)}.')
        key_t = key.transpose(-2, -1)
        kk = key_t @ key
        identity = torch.eye(kk.shape[-1], device=x.device, dtype=kk.dtype)
        regularized_kk = kk + F.softplus(self.raw_alpha).to(dtype=kk.dtype) * identity
        solved_keys = torch.linalg.solve(regularized_kk, key_t)
        attn_map = self.softmax(solved_keys @ value)
        out = (query_reshaped @ attn_map).permute(0, 2, 1, 3).reshape(b_q, n_q, c_q)
        out = self.out(out)
        if query_mask is not None:
            out = out * query_mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)
        return out

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=dropout))
        self.wk = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=dropout))
        self.wv = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=dropout))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, mask=None):
        query, key, value = self.wq(x), self.wk(x), self.wv(x)
        valid_mask = None
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(f'mask must have shape {tuple(x.shape[:2])}.')
            valid_mask = mask.to(device=x.device).bool()
        att, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=None if valid_mask is None else ~valid_mask,
            need_weights=False,
        )
        output = att + x
        if valid_mask is not None:
            output = output * valid_mask.to(dtype=output.dtype).unsqueeze(-1)
        return output

class IntentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Intention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias, device=device)

    def forward(self, x, q, x_mask=None, q_mask=None):
        x = self.norm_layer(x)
        att = self.attn(x, q, key_mask=x_mask, query_mask=q_mask)
        return att + q

class BiIntention(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(BiIntention, self).__init__()
        if layer < 0:
            raise ValueError('interaction layer count cannot be negative.')
        self.layer = layer
        self.drug_intention = nn.ModuleList([IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, drug, protein, drug_mask=None, protein_mask=None):
        if drug_mask is None:
            drug_mask = torch.ones(drug.shape[:2], device=drug.device, dtype=torch.bool)
        else:
            drug_mask = drug_mask.to(device=drug.device).bool()
        if protein_mask is None:
            protein_mask = torch.ones(protein.shape[:2], device=protein.device, dtype=torch.bool)
        else:
            protein_mask = protein_mask.to(device=protein.device).bool()
        if drug_mask.shape != drug.shape[:2] or protein_mask.shape != protein.shape[:2]:
            raise ValueError('Drug and protein masks must match their sequence dimensions.')
        if torch.any(drug_mask.sum(dim=1) == 0) or torch.any(protein_mask.sum(dim=1) == 0):
            raise ValueError('Each drug and protein sequence must contain at least one valid token.')

        drug = self.attn_drug(drug, drug_mask)
        protein = self.attn_protein(protein, protein_mask)
        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein, drug_mask, protein_mask)
            v_d = self.protein_intention[i](protein, drug, protein_mask, drug_mask)
            drug, protein = v_d, v_p
        drug = drug.masked_fill(~drug_mask.unsqueeze(-1), torch.finfo(drug.dtype).min)
        protein = protein.masked_fill(~protein_mask.unsqueeze(-1), torch.finfo(protein.dtype).min)
        v_d_pool = torch.max(drug, dim=1)[0]
        v_p_pool = torch.max(protein, dim=1)[0]
        f = torch.cat((v_d_pool, v_p_pool), dim=1)
        return f

class DrugFeatureFusion(nn.Module):
    def __init__(self, bermol_dim=768, morgan_dim=1024, out_dim=768, dropout=0.1):
        super(DrugFeatureFusion, self).__init__()
        self.morgan_proj = nn.Sequential(
            nn.Linear(morgan_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        self.bermol_proj = nn.Identity() if bermol_dim == out_dim else nn.Linear(bermol_dim, out_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, bermol, morgan):
        m_feat = self.morgan_proj(morgan)
        b_feat = self.bermol_proj(bermol)
        combined = torch.cat([b_feat, m_feat], dim=1)
        z = self.gate_net(combined)
        fused = z * b_feat + (1 - z) * m_feat
        return self.layer_norm(fused)

# ------------------------------------------------------------------------------
# 核心组件: HMCN-F 分类头
# ------------------------------------------------------------------------------
class HMCNFHead(nn.Module):
    def __init__(self, input_dim, hierarchy=[1, 2], beta=0.5, dropout_rate=0.2, hidden_dim=384):
        """
        HMCN-F (Hierarchical Multi-Label Classification Network - Feedforward)

        Args:
            input_dim: 输入特征维度 (WideDeep 输出)
            hierarchy: [Level 1 节点数, Level 2 节点数] -> 默认为 [1, 2] (Active; Ago/Ant)
            beta: 平衡 Local 和 Global 预测的系数 (0.5 表示各占一半)
        """
        super().__init__()
        self.hierarchy = hierarchy
        self.beta = beta
        self.total_labels = sum(hierarchy) # 1 + 2 = 3

        # --- 全局流 (Global Flow) ---
        # Level 1 Global: 输入 -> Hidden
        self.global_l1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Level 2 Global: [Level 1 Hidden, 原始输入] -> Hidden
        # 这种 Skip Connection 保证深层也能看到原始特征
        self.global_l2 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 全局共识 (Global Consensus): 尝试一次性预测所有 3 个标签
        self.global_consensus = nn.Sequential(
            nn.Linear(hidden_dim, self.total_labels),
            nn.Sigmoid()
        )

        # --- 局部流 (Local Flow) ---
        # Level 1 Local: 预测是否活性 (1个节点)
        self.local_l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hierarchy[0]),
            nn.Sigmoid()
        )

        # Level 2 Local: 预测作用模式 (2个节点: Agonist, Antagonist)
        self.local_l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hierarchy[1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, input_dim]

        # 1. 计算 Global States
        g1 = self.global_l1(x)
        # 拼接 g1 和 原始输入 x
        g2_input = torch.cat([g1, x], dim=1)
        g2 = self.global_l2(g2_input)

        # 2. 计算 Global Prediction (P_glob)
        # 这里的 g2 是最后一层的全局状态
        p_glob = self.global_consensus(g2) # [Batch, 3]

        # 3. 计算 Local Predictions (P_loc)
        p_loc_1 = self.local_l1(g1) # [Batch, 1]
        p_loc_2 = self.local_l2(g2) # [Batch, 2]

        p_loc = torch.cat([p_loc_1, p_loc_2], dim=1) # [Batch, 3]

        # 4. 加权融合
        output = (1 - self.beta) * p_glob + self.beta * p_loc

        return output

# ------------------------------------------------------------------------------
# 主模型: WideDeep + HMCN-F
# ------------------------------------------------------------------------------
class WideDeepHMCNFModel(nn.Module):
    def __init__(self, bermol_dim=768, esm_dim=640, morgan_dim=1024,
                 esm2_seq_dim=2560, # 默认 ESM2-3B 维度
                 hidden_dim=256, interaction_dim=128, interaction_layers=1, num_heads=4):
        super(WideDeepHMCNFModel, self).__init__()
        if num_heads <= 0 or interaction_dim <= 0 or interaction_dim % num_heads:
            raise ValueError('interaction_dim must be positive and divisible by num_heads.')
        if interaction_layers < 0:
            raise ValueError('interaction_layers cannot be negative.')

        # --- Wide Part ---
        self.drug_fusion_layer = DrugFeatureFusion(bermol_dim=bermol_dim, morgan_dim=morgan_dim, out_dim=bermol_dim)
        wide_input_dim = bermol_dim + esm_dim
        self.wide_net = nn.Sequential(
            nn.Linear(wide_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, hidden_dim)
        )

        # --- Deep Part ---
        self.drug_proj = nn.Linear(bermol_dim, interaction_dim)
        self.prot_proj = nn.Sequential(
            nn.Linear(esm2_seq_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.Dropout(0.1)
        )
        self.bi_intention = BiIntention(embed_dim=interaction_dim, layer=interaction_layers, num_head=num_heads)

        # --- HMCN-F Head ---
        total_dim = hidden_dim + (interaction_dim * 2)

        # hierarchy=[1, 2] -> Level 1 (Active/Inactive), Level 2 (Agonist, Antagonist)
        self.hmcnf = HMCNFHead(
            input_dim=total_dim,
            hierarchy=[1, 2],
            beta=0.5,
            hidden_dim=384
        )

    def forward(self, b_cls, e_cls, morgan_fp, b_seq, b_mask, p_seq_emb, p_mask):
        # Wide
        fused_drug_feat = self.drug_fusion_layer(b_cls, morgan_fp)
        wide_feat = torch.cat([fused_drug_feat, e_cls], dim=1)
        wide_out = self.wide_net(wide_feat)

        # Deep
        d_feat = self.drug_proj(b_seq) * b_mask.unsqueeze(-1)
        p_feat = self.prot_proj(p_seq_emb) * p_mask.unsqueeze(-1)

        # Interaction
        f = self.bi_intention(d_feat, p_feat, b_mask, p_mask)

        # Combine
        combined = torch.cat([wide_out, f], dim=1)

        # HMCN-F Output: [Batch, 3] (Probabilities)
        return self.hmcnf(combined)
