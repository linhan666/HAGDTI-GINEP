# GIN-EP 完整架构与接口

## 配体图

每个原子输入 34 维化学特征。RW-PE 使用不含自环的双向键图计算随机游走返回概率：`PE[v,k] = (P^k)[v,v]`，`k=1..20`。PE 经 BatchNorm 和 `Linear(20,8)`，与原子特征拼接为 42 维，再投影至 32 维。

键特征 12 维由键类型 4 维、共轭 1 维、环内 1 维、立体化学 6 维组成，经 `Linear(12,32)` 后供三个 GINE 层共享。每层计算：

`h_v' = MLP((1 + eps) h_v + sum_u ReLU(h_u + e_uv))`

每层之后使用 `ReLU → BatchNorm(32)`；GINE 采用 PyG 默认 `eps=0`、`train_eps=False`。特征图显式含自环，自环的原始键特征为零，但边线性层有 bias，因此投影后自环边嵌入未必为零。这些细节保留原始实现。

最后节点投影 `32→256`，按 `batch_atom` 还原 `[B,Amax,256]` 和 `[B,Amax]` 原子 mask。配体 wide 分支独立编码 ChemBERTa CLS 与 16 维药效团，得到 `[B,256]`；两条配体通路均保留。

## 蛋白表示

完整序列、正构口袋序列和七个 TM 片段共用一个 ESMC。TM 的 CLS 经线性投影、可学习 TM 位置偏置、LayerNorm 和 Dropout；七个片段 CLS 取平均。

`seq_repr = LayerNorm(full_proj(full_CLS) + 1.5 * pocket_proj(pocket_CLS) + mean(TM_CLS))`

12 维 BW motif 特征经过 `12→32→256` 的 MLP；门控融合为：

`protein = Dropout(seq_repr + sigmoid(Linear(concat(seq_repr,motif))) * motif)`

每个 TM 的逐 token ESMC 输出另行投影为 256 维，形成交叉注意力的 K/V。缓存接口保留 `full_cls`、`pocket_cls` 和 `tm_hidden` 三类冻结 ESMC 输出。

## 交叉注意力和输出

Q 为原子 token，K/V 为展平的七个 TM token。四个注意力头，每头 64 维，按蛋白 mask 屏蔽 padding。上下文经过输出线性层、Dropout，对真实原子做 max/mean pooling，拼接 `512→256` 并归一化。

`concat(protein, ligand_wide, cross_context)` 为 768 维，融合 MLP 为 `768→512→256`，每层后为 GELU 和 Dropout。默认 MLP 三分类头为 `256→128→3`，隐藏层使用 GELU 和 Dropout(0.1)。另支持 HMCN-F、HiAGM-TP、C-HMCNN 和 conditional-softmax，参数及概率口径见 README。独立辅助头为 `Linear(256,2)`，仅 MLP 且 `use_binary_aux=True` 时创建。二分类辅助损失权重原配置为 0.3，但本仓库不包含训练调度器。

## batch 输入

所有 token ID 和边索引为 `torch.long`，特征为 float32。mask 的 1 表示有效位置，0 表示 padding。

| 字段 | 形状 | 用途 |
|---|---|---|
| `full_ids`, `full_mask` | `[B,Lfull]` | GPCR 全局序列 |
| `pocket_ids`, `pocket_mask` | `[B,Lpocket]` | 口袋序列 |
| `tm_ids`, `tm_masks` | `[B,7,Ltm]` | 七个 TM 片段 |
| `motif_feat` | `[B,12]` | 六个 BW 位点的存在标志与归一化序列位置 |
| `lig_ids`, `lig_mask` | `[B,Lsmiles]` | ChemBERTa SMILES token |
| `pharm_feat` | `[B,16]` | 药效团描述符 |
| `atom_features` | `[Ntotal,34]` | 批量拼接原子特征 |
| `atom_edge_index` | `[2,Etotal]` | 加节点偏移的批量图边索引 |
| `atom_edge_attr` | `[Etotal,12]` | 与边一一对应的键特征 |
| `atom_rw_pe` | `[Ntotal,20]` | RW-PE |
| `batch_atom` | `[Ntotal]` | 原子的样本编号，按样本排序 |
| `prot_idx` | `[B]`，可选 | 使用 ESMC 缓存时的索引 |

批量化时，每个样本的 `atom_edge_index` 应增加此前样本的累计原子数；其余原子/边特征沿第一维拼接。GPCR 片段、BW 位置与序列必须来自同一靶点和相同序列版本。

输出字典包含实际选择的 `head_type`、`logits [B,3]`、`repr [B,256]`、`protein_attn`/`tm_attn [B,4,Amax,7*Ltm]`，以及 MLP 启用辅助头时的 `binary_logits [B,2]`。conditional-softmax 另含 `active_logit [B]` 和 `cond_logits [B,2]`；层级头的 `logits` 是节点 logits，应通过 `extract_3class_probs` 转为互斥三类概率。注意力用于接口解释，不能直接等同于已验证的关键残基作用。
