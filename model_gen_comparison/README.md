# model_gen 与 model_gen_deep 对比及原始代码快照

提取日期：2026-10-14。来源：linhan666/HAGDTIGen，提交 `3a9298603119ff245ab5be95727e3a0896fea1d4`。

这两个目录保存原始实现，模型、训练脚本和 utils.py 均与来源逐字节一致，SHA-256 见 PROVENANCE.json。它们是用于分析和追溯的代码快照；已知缺陷未在快照中改写。不能据此宣称可直接完成训练、生成分布合格或具备 GPCR 生物学效力。

## 文件对应关系

| 目录 | 模型 | 对应训练脚本 | 本地依赖 |
| --- | --- | --- | --- |
| model_gen | model_gen.py | train_gen.py | utils.py |
| model_gen_deep | model_gen_deep.py | train_gen_deep_nowide.py（适配入口）、train_gen_deep.py（旧入口） | utils.py |

`train_gen_deep_nowide.py` 是已适配 DeepOnly 的现有训练变体，本次一并原样提取。它去掉 CLS/Morgan 特征依赖，并仅向构造器传 bermol_dim/esm_seq_dim。外部 BerMol 包及权重、ESM 权重、真实数据、缓存和训练输出不在快照中。

## 架构对比

| 项目 | model_gen | model_gen_deep |
| --- | --- | --- |
| DTI 主干 | WideDeepModel_Core，Wide + Deep | DeepModel_Core，仅 Deep |
| Wide 分支 | BerMol CLS 与 Morgan 门控融合，再与 ESM CLS 拼接，经 MLP 得到 256 维 | 删除 |
| Deep 分支 | BerMol token 与 ESM residue 投影到 128 维，经一层双向 BiIntention，分别 max pooling | 相同基本结构 |
| 分类器输入 | Wide 256 + 双序列池化 256 = 512 维 | 双序列池化 256 维 |
| 潜变量来源 | 原始 BerMol 序列 → 384 维投影 → 两层 Transformer → 每个 token 的 mu/logvar | 交互后药物和蛋白的全局池化向量 256 → mu/logvar 384 |
| 潜变量形状 | L 个 384 维潜变量 | 一个 384 维全局潜变量 |
| 条件注入 | affinity 投影 + ESM CLS 投影，广播到全部潜变量 | affinity 投影加到全局潜变量，蛋白信息来自交互全局特征 |
| Fusion memory | 增强药物序列 L + 潜变量序列 L = 2L token | 增强药物序列 L + 全局潜变量 1 = L+1 token |
| Fusion encoder | 两层，最终 LayerNorm | 两层，最终 LayerNorm |
| Decoder | 4 层，384 维，4 heads，FFN 1024；最大生成长度 512 | 相同 |
| 生成交叉熵 | 同位置预测同位置 target，未移位 | 使用位置 t 预测 target t+1 |
| 采样方差 | log variance = -abs(raw)，方差不超过 1 | 采样使用 6*tanh(raw)，方差范围约 exp(-6) 到 exp(6) |
| KL | 同样使用 -abs(raw)，所有 token/batch/dim 求和后除以固定 64 | 对 raw logvar 算 KL，按 batch 平均，与采样变换不一致 |
| forward 第二返回值 | None | 完整 prediction_scores |
| 参数量* | 17,868,454 | 11,575,718 |

*参数量由当前 CPU PyTorch 实例化验证：默认特征维度，使用 CCO 构建词表（34 token）。词表变化会改变参数量。Deep 少 6,292,736 个参数，约 35.2%。这不等同于实际训练速度或质量测量。

Deep 并非单纯删除 Wide 的严格消融：它同时改变潜变量来源、形状、条件注入、memory 长度、LM 目标和 KL 定义。因此历史训练结果不能直接归因于 Wide 分支，也不能直接互换 checkpoint。

## 训练脚本对比

两个同名训练脚本 train_gen.py / train_gen_deep.py 仅有两处文本差异：模型导入名和默认 output_dir。其余数据清洗、特征提取、数据集、验证和优化流程相同。

- 数据：SMILES、Protein、Y；可选 Affinity。支持列名 Smiles/Sequence/Activity_Label/affinity 映射。Y 是二分类标签。
- 特征：两份脚本均提取 BerMol CLS/序列、ESM CLS/序列、1024 位 Morgan。Deep 模型不使用 CLS/Morgan，但脚本仍计算并要求它们存在。
- 数据处理：去盐取最大片段并规范化 SMILES，按 SMILES/Protein 保留第一条重复记录；特征缺失样本会返回 None 并被 collator 丢弃。
- 划分：90/10 随机配对划分，无固定 seed、scaffold 或 protein-disjoint 约束；词表在划分前构建。
- 默认参数：batch_size=128，epochs=50，lr=1e-4，num_workers=16，word_dropout=0.2，patience=6。
- 损失：DTI BCE + LM CE + KL；KL 权重前 20% 计划步数线性增加至 0.001。
- 优化：AdamW、CUDA AMP、梯度裁剪 1.0；按验证 DTI AUC 保存 best_model.pth 并早停，不按生成指标选 checkpoint。
- 缺少 affinity 时填入正样本 9.0、负样本 4.0；这是标签构造条件，不能视为测得亲和力。

### Deep 的适配训练脚本

`train_gen_deep_nowide.py` 导入同一个 model_gen_deep，但移除了 ESM CLS/Morgan 特征函数和数据集输入，只需要 BerMol 序列与 ESM 序列；构造器参数与模型签名一致。默认 num_workers=12、output_dir=./output_fusion_final_deep_nowide，其余主要优化参数相同。它仍保留 BerMol CLS 的提取缓存，不过模型不消费该输出。此脚本修复的是接口适配，并没有修复下面列出的 KL、padding、随机划分等科学与运行问题。

## 已知运行与科学解释问题

1. **旧 Deep 训练入口不能直接运行**：构造器仅接受 tokenizer、bermol_dim、esm_seq_dim，而 train_gen_deep.py 仍传 esm_cls_dim 和 morgan_dim。已在 CPU 上复现 TypeError。
2. **model_gen 的 LM 未移位**：decoder 可看到当前 token，再预测当前 token，存在目标直接可见的问题。重建 CE 不代表下一 token 生成能力。
3. **Deep 的 KL 与采样分布不同**：采样变换 logvar，KL 却使用原始 logvar。
4. **两者的交互与 max pooling 未使用 b_mask/p_mask**：虽然后续 Transformer 使用药物 padding mask，DTI 交互和全局池化仍可能受到 padding 与 batch 长度影响。
5. **Intention 数值风险**：直接求逆，alpha 无正值约束；CUDA AMP 下精度及可逆性需另行验证。
6. **训练数据与缓存风险**：缓存无输入哈希绑定；缺失特征静默跳过；重复标签冲突未审计；Affinity 的 NaN 不会被 float 转换异常捕获。
7. **训练边界**：num_workers=0 与 persistent_workers/prefetch 配置不兼容；Wide BatchNorm 遇到 batch_size=1 的尾批可能报错；验证损失按 loader 总批数计，跳过批次会偏置结果。
8. **生成边界**：affinity_override=0 会被当作未提供而变成 9；未校验 temperature；已 EOS 样本会继续写 token，直至全 batch 完成；潜变量在 eval 下仍随机采样。

独立审查结果见 REVIEW.md。快照保存历史缺陷以支持对比，不将“提取完整”表述为“代码审查通过”。若用于新实验，应先建立单独修复版本，保留此快照作为基线。

## 依赖及历史入口

Python 包见 requirements.txt（依据 import 提取，未锁定完整训练环境）。需要另外提供可信来源的 BerMol Python 包及预训练文件、两个 ESM 本地模型目录和输入表。torch.load/pickle 只能用于可信权重和缓存。旧 BerMol 序列化文件可能需要匹配旧 PyTorch 行为，当前 torch.load 的默认设置不能保证兼容。

以下为原始脚本调用方式，用于定位接口匹配的入口，未完成真实训练验证。旧 train_gen_deep.py 存在上述构造器错误，保留用于追溯。

```bash
cd model_gen
python train_gen.py --data_path /path/data.csv --output_dir /path/output_wide --bermol_path /path/bermol --esm_cls_path /path/esm_cls --esm_seq_path /path/esm_seq

cd ../model_gen_deep
python train_gen_deep_nowide.py --data_path /path/data.csv --output_dir /path/output_deep --bermol_path /path/bermol --esm_seq_path /path/esm_seq
```

## 本次验证范围

- 七份 Python 文件与来源逐字节一致，已记录 SHA-256；AST 语法解析通过。
- 默认模型可在 CPU 实例化并计数参数。
- 小维度合成 batch（2 个样本）完成两者 forward 和限制为 5 token 的短生成；Deep 单独使用正确构造器完成该检查。
- 复现 Deep 原始训练构造器 TypeError。
- 未运行真实特征提取、训练、GPU AMP、分布质量评价或生物学验证。
