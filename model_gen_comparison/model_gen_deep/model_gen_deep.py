import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.pe.size(0): return x
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class TransformerDecoderWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(d_model=args.decoder_embed_dim, nhead=args.decoder_attention_heads, dim_feedforward=args.decoder_ffn_embed_dim, dropout=args.dropout, activation='relu', batch_first=False)
        self.layer_norm = nn.LayerNorm(args.decoder_embed_dim)
    def forward(self, x, mem, x_mask=None, x_padding_mask=None, mem_padding_mask=None):
        return self.layer(x, memory=mem, tgt_mask=x_mask, tgt_key_padding_mask=x_padding_mask, memory_key_padding_mask=mem_padding_mask)

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=args.encoder_embed_dim, nhead=args.encoder_attention_heads, dim_feedforward=args.encoder_ffn_embed_dim, dropout=args.dropout, activation='relu', batch_first=False)
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)
    def forward(self, x, encoder_padding_mask=None):
        return self.layer(x, src_key_padding_mask=encoder_padding_mask)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerEncoderWrapper(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'encoder_ffn_embed_dim': ff_dim,
                'dropout': 0.1,
                'encoder_normalize_before': True
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask=None):
        for layer in self.layer:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)
        x = self.layer_norm(x)
        return x


class Intention(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False):
        super(Intention, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.alpha = nn.Parameter(torch.rand(1))
        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)
        self.softmax = nn.Softmax(dim=-2)
        self.out = nn.Linear(dim, dim)
    def forward(self, x, query=None):
        if query is None: query = x
        q_out, k_out, v_out = self.wq(query), self.wk(x), self.wv(x)
        b, n, c = x.shape
        b_q, n_q, c_q = query.shape
        key = k_out.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        key_t = key.permute(0, 1, 3, 2)
        value = v_out.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        query_reshaped = q_out.reshape(b_q, n_q, self.head, self.head_dim).permute(0, 2, 1, 3)
        kk = key_t @ key
        kk = self.alpha * torch.eye(kk.shape[-1], device=x.device) + kk
        kk_inv = torch.inverse(kk)
        attn_map = self.softmax((kk_inv @ key_t) @ value)
        out = (query_reshaped @ attn_map).permute(0, 2, 1, 3).reshape(b_q, n_q, c_q)
        return self.out(out)

class BiIntention(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8):
        super(BiIntention, self).__init__()
        self.layer = layer
        self.drug_intention = nn.ModuleList([Intention(dim=embed_dim, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([Intention(dim=embed_dim, num_heads=num_head) for _ in range(layer)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, drug, protein):
        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein)
            v_d = self.protein_intention[i](protein, drug)
            drug = drug + v_d
            protein = protein + v_p
            drug = self.norm(drug)
            protein = self.norm(protein)
        return drug, protein


class DeepModel_Core(nn.Module):
    def __init__(self, bermol_dim=768, esm2_seq_dim=1280, interaction_dim=128, interaction_layers=1, num_heads=4):
        super(DeepModel_Core, self).__init__()


        self.drug_proj = nn.Linear(bermol_dim, interaction_dim)
        self.prot_proj = nn.Sequential(nn.Linear(esm2_seq_dim, interaction_dim), nn.LayerNorm(interaction_dim), nn.Dropout(0.1))
        self.bi_intention = BiIntention(embed_dim=interaction_dim, layer=interaction_layers, num_head=num_heads)


        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim * 2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, b_seq, b_mask, p_seq_emb, p_mask):


        d_feat, p_feat = self.drug_proj(b_seq), self.prot_proj(p_seq_emb)
        d_enhanced, p_enhanced = self.bi_intention(d_feat, p_feat)


        deep_out = torch.cat((d_enhanced.max(dim=1)[0], p_enhanced.max(dim=1)[0]), dim=1)


        logits = self.classifier(deep_out).squeeze(-1)


        return logits, deep_out, d_enhanced


class FusionDeepDTAGen(nn.Module):
    def __init__(self, tokenizer, bermol_dim=768, esm_seq_dim=1280):
        super(FusionDeepDTAGen, self).__init__()

        self.interaction_dim = 128


        self.dti_model = DeepModel_Core(
            bermol_dim=bermol_dim,
            esm2_seq_dim=esm_seq_dim,
            interaction_dim=self.interaction_dim,
            interaction_layers=1,
            num_heads=4
        )

        self.gen_hidden_dim = 384
        self.ff_dim = 1024
        self.heads = 4
        self.layers = 4
        self.max_len = 512


        input_dim_for_vae = self.interaction_dim * 2

        self.z_mean = nn.Linear(input_dim_for_vae, self.gen_hidden_dim)
        self.z_logvar = nn.Linear(input_dim_for_vae, self.gen_hidden_dim)


        self.seq_bridge = nn.Linear(self.interaction_dim, self.gen_hidden_dim)


        self.affinity_proj = nn.Linear(1, self.gen_hidden_dim)


        self.decoder = nn.ModuleList([TransformerDecoderWrapper(Namespace({'decoder_embed_dim': self.gen_hidden_dim, 'decoder_attention_heads': self.heads, 'decoder_ffn_embed_dim': self.ff_dim, 'dropout': 0.2, 'decoder_normalize_before': True})) for _ in range(self.layers)])
        self.decoder_norm = nn.LayerNorm(self.gen_hidden_dim)


        self.fusion_encoder = TransformerEncoder(dim=self.gen_hidden_dim, ff_dim=self.ff_dim, num_head=self.heads, num_layer=2)

        self.pos_encoding = PositionalEncoding(self.gen_hidden_dim, max_len=5000)
        self.zz_seg_encoding = nn.Parameter(torch.randn(self.gen_hidden_dim))


        self.vocab_size = len(tokenizer)
        self.word_pred = nn.Sequential(
            nn.Linear(self.gen_hidden_dim, self.gen_hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(self.gen_hidden_dim),
            nn.Linear(self.gen_hidden_dim, self.vocab_size)
        )
        self.word_embed = nn.Embedding(self.vocab_size, self.gen_hidden_dim)

        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.unk_value = tokenizer.s2i.get('<unk>', self.pad_value)

    def reparameterize(self, mu, logvar):
        logvar = 6.0 * torch.tanh(logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def expand_then_fusing(self, z, pp_mask, vvs):

        zzs = z.unsqueeze(0) + self.zz_seg_encoding

        device = z.device
        batch_size = z.size(0)
        z_mask = torch.zeros((batch_size, 1), device=device, dtype=pp_mask.dtype)
        full_mask = torch.cat((pp_mask, z_mask), dim=1)

        zzz = torch.cat((vvs, zzs), dim=0)
        zzz = self.fusion_encoder(zzz, encoder_padding_mask=full_mask)

        return zzz, full_mask

    def forward(self, data, word_dropout=0.0):

        b_seq, b_mask, p_seq, p_mask = data.b_seq, data.b_mask, data.e_seq, data.e_mask
        targets = data.target_seq
        affinity_val = getattr(data, 'affinity', None)


        dti_logits, dti_global, dti_seq = self.dti_model(b_seq, b_mask, p_seq, p_mask)
        dti_loss = F.binary_cross_entropy_with_logits(dti_logits, data.y.float())


        mu = self.z_mean(dti_global)
        logvar = self.z_logvar(dti_global)
        z_random = self.reparameterize(mu, logvar)


        if affinity_val is not None:
            val_emb = self.affinity_proj(affinity_val.unsqueeze(1).float())
            z_final = z_random + val_emb
        else:
            z_final = z_random


        d_enh_mapped = self.seq_bridge(dti_seq).permute(1, 0, 2)


        encoder_padding_mask = (b_mask == 0)
        zzz, encoder_mask = self.expand_then_fusing(z_final, encoder_padding_mask, d_enh_mapped)


        decoder_inputs = targets
        if self.training and word_dropout > 0:
            mask = (torch.rand(targets.shape, device=targets.device) < word_dropout) & (targets != self.pad_value)
            noise_tokens = torch.full(targets.shape, self.unk_value, dtype=torch.long, device=targets.device)
            decoder_inputs = torch.where(mask, noise_tokens, targets)

        target_embed = self.word_embed(decoder_inputs)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2))

        tgt_len = targets.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(targets.device)

        x = target_embed
        for layer in self.decoder:
            x = layer(x, zzz, x_mask=tgt_mask, mem_padding_mask=encoder_mask)

        output = self.decoder_norm(x).permute(1, 0, 2)
        prediction_scores = self.word_pred(output)

        shifted_preds = prediction_scores[:, :-1, :].reshape(-1, self.vocab_size)
        shifted_targets = targets[:, 1:].reshape(-1)

        lm_loss = F.cross_entropy(shifted_preds, shifted_targets, ignore_index=self.pad_value)

        kl_loss = torch.mean(torch.sum(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()), dim=-1))

        return dti_logits, prediction_scores, dti_loss, lm_loss, kl_loss

    def generate(self, data, random_sample=False, affinity_override=None, temperature=1.0):
        val_scalar = float(affinity_override) if affinity_override else 9.0
        val = torch.full((data.b_seq.size(0),), val_scalar, device=data.b_seq.device)

        with torch.no_grad():

            _, dti_global, dti_seq = self.dti_model(
                data.b_seq, data.b_mask,
                data.e_seq, data.e_mask
            )

        mu, logvar = self.z_mean(dti_global), self.z_logvar(dti_global)
        val_emb = self.affinity_proj(val.unsqueeze(1).float())
        z_final = self.reparameterize(mu, logvar) + val_emb

        d_enh_mapped = self.seq_bridge(dti_seq).permute(1, 0, 2)

        encoder_padding_mask = (data.b_mask == 0)
        zzz, encoder_mask = self.expand_then_fusing(z_final, encoder_padding_mask, d_enh_mapped)

        return self._generate(zzz, encoder_mask, random_sample, temperature)

    def _generate(self, zzz, encoder_mask, random_sample=False, temperature=1.0):
        batch_size, device = zzz.shape[1], zzz.device
        token = torch.full((batch_size, self.max_len), self.pad_value, dtype=torch.long, device=device)
        token[:, 0] = self.sos_value
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, self.max_len):
            curr = token[:, :t]
            x = self.word_embed(curr).permute(1, 0, 2)
            x = self.pos_encoding(x)

            mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(device)

            for layer in self.decoder:
                x = layer(x, zzz, x_mask=mask, mem_padding_mask=encoder_mask)

            last_token_feat = self.decoder_norm(x)[-1, :, :]
            logits = self.word_pred(last_token_feat)

            if random_sample:
                if temperature != 1.0: logits = logits / temperature
                k = torch.multinomial(torch.softmax(logits, 1), 1).squeeze(1)
            else:
                k = torch.argmax(logits, -1)

            token[:, t] = k
            finished |= (k == self.eos_value)
            if finished.all(): break

        return token[:, 1:]
