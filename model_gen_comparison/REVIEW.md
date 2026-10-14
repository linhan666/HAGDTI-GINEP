# 原样提取独立审查

审查日期：2026-10-14。审查者：独立 Codex Agent，未参与归档复制；没有冒称 Sonnet。范围：model_gen_comparison 下七个 Python 文件。原始路径：C:/Users/Admin/.codex/worktrees/bea2/HAGDTIGen/gen。

结论：七个归档文件 SHA256 与源文件逐一一致，原样提取完整性通过；代码存在原始接口与训练风险，不能称为代码审查通过。本次没有修改归档代码，没有启动训练或下载权重。

## 完整性证据

|归档路径|SHA256|源副本相同|
|---|---|---|
|model_gen/model_gen.py|52411E5277DAE49F668C9BC2AC9C40E713F4752F9EA0710A0FEE1977F77A7A65|是|
|model_gen/train_gen.py|906CF350EBF9FE76B4A050952029FAFB3154E126CF9FBE320A55E3BBAFA09E2F|是|
|model_gen/utils.py|5ACF28AE00D0205F143EBA63D690836D7108CFDBED73602FA68BADF10AF6A2FE|是|
|model_gen_deep/model_gen_deep.py|AEBE70E950ACF90E135FD772A0C194F61DF4C69E33AE15E7C87FAA6F8D897EA6|是|
|model_gen_deep/train_gen_deep.py|C386BF81526DC6619B415055BE3810A9A4D620BFE03D81E1625EA79039424703|是|
|model_gen_deep/train_gen_deep_nowide.py|FC34B86CCA5CAB621B0131CFABF5BC97F183E250E0C3C18C6C4DE31CC41A74A7|是|
|model_gen_deep/utils.py|5ACF28AE00D0205F143EBA63D690836D7108CFDBED73602FA68BADF10AF6A2FE|是|

七文件 AST 语法解析通过；这只能证明语法有效，不证明运行、训练正确或科研有效。两个 utils.py 字节相同。将两个目录分开保留，可直接从相应目录调用各自脚本；需要相应 Python 依赖与外部模型资源。归档不含数据、BerMol/ESM 权重、训练结果或运行环境。

## 原始代码问题

以下问题均存在于完全相同的源代码，不是本次提取新增的问题；保留原样以支持历史对比。

0. **高：Wide 版 LM 训练目标直接可见。** `model_gen/model_gen.py:226` 将完整 targets 用作 decoder_inputs，`:235` 对同位置 targets 计算 CE；causal mask 允许当前位置，因而当前目标 token 可被模型读取。训练目标是同位置重构，生成却按下一个 token 解码，训练与生成不匹配。Deep 版 `model_gen_deep.py:291-295` 使用预测与目标移位，不存在这一相同缺陷。
1. **高：旧 Deep 训练入口与模型构造器不匹配。** `model_gen_deep/train_gen_deep.py:361-366` 传入 `esm_cls_dim`、`morgan_dim`，而 `model_gen_deep.py:163` 仅接受 tokenizer、bermol_dim、esm_seq_dim。静态 AST 对照确认两个不受支持的关键字，会触发 TypeError。该旧入口还强制提取 DeepOnly 不使用的 ESM CLS/Morgan 特征。`train_gen_deep_nowide.py:310` 的构造调用及仅序列 Dataset 与当前 Deep 模型相匹配，因此它是结构上匹配的 Deep 入口；结构匹配不等于训练风险已消除。
2. **高：padding 未进入交互与池化。** 两个模型的 Intention/BiIntention 不接收 mask，Core 的 forward 接收 b_mask/p_mask 却不使用。线性偏置及交互可把 padding 变成非零特征；随后未掩码 max pooling 可改变结果，预测可能随同批其他样本长度改变。需要在开发分支验证并实现 masked interaction/pooling。
3. **中：Deep VAE 采样与 KL 参数不一致。** `model_gen_deep.py:224` 对采样的 logvar 使用 `6*tanh(logvar)`，但 `:297` 的 KL 使用未变换的 logvar，正则项与实际采样分布不一致，且原始 exp(logvar) 可溢出。
4. **中：单进程加载参数不可用。** 三个训练脚本固定 `persistent_workers=True, prefetch_factor=2`；`--num_workers 0` 会因 DataLoader 参数约束失败。应按 workers 数量分支配置。
5. **中：Wide 模型有效 batch=1 风险。** Wide 支路使用 BatchNorm1d；DataLoader 未 drop_last，collate 又过滤缺少特征的记录，因此训练时最终有效 batch 可能只有一个样本并触发 BatchNorm 错误。
6. **中：缺失特征与空集合边界不完整。** Dataset 返回 None、collate 过滤，且验证按总 loader 长度平均损失、异常时把 AUC/ACC 设为 0；缺失特征可能静默降低覆盖，空集合可产生除零或 next(iter(...)) 失败。没有逐行跳过清单。
7. **中：Affinity 的 NaN 不被拦截。** `float(row['Affinity'])` 对 NaN 不抛异常；NaN 可进入 affinity projection，再造成 loss 为 NaN 被跳过。生成函数 `if affinity_override` 又把显式 0 覆盖为默认 9.0。
8. **中：缓存没有数据及模型身份绑定。** 存在即 pickle.load，没有输入哈希、模型版本、特征维度、tokenizer 身份校验；改变数据或权重但复用 output_dir 时可能读到陈旧缓存。pickle/旧对象 checkpoint 只应读取可信本地来源。
9. **科研边界：随机行切分和冲突消重。** 未固定 seed 的 random_split 不能复现，不能证明未见配体、scaffold 或未见靶点泛化；drop_duplicates(SMILES,Protein) 没有说明冲突标签处理。Wide KL 以固定 64 归一化，也会随有效 batch 与序列长度变化。上述结构差异不能直接解释为 fair ablation 或性能优劣。

## 外部审查状态

使用导出根目录本地 glm_review.py；凭据及脚本不属于可发布审查材料。

- 第一轮：七文件新增之前的六文件合并 stdin，脚本实际退出码 **0**；配置模型为 glm-5.3，但输出标签显示 GLM 5.1。报告总体评价为空、问题数为 0，不能采信为有效通过。
- 脚本 `_parse_json_from_text` 对无 JSON 的空字符串默认返回 pass=true；所以空结果形成假阳性。
- 第二轮：仅在独立审查进程内增加严格响应校验，再运行相同脚本主入口，对六文件合并审查。空 content、缺 summary 或无效结构判为 error，以触发脚本原有 GLM → DeepSeek 顺序降级。没有修改脚本文件。此校验仍允许原 parser 的非 JSON 文字 fallback，因此下述结果继续按无效处理。**实际退出码 0，但结果无效。** GLM 空响应被校验拒绝后，DeepSeek 返回了长段推理文字而非要求的 JSON；其中实际指出同位置 CE、旧 Deep 构造参数、padding 与加载器问题。脚本仍将非 JSON 文字转换成 pass=true/0 issues，本轮内存校验未阻止这一非空 fallback。不得将该名义退出码称为代码通过。
- 第三轮：同样严格校验，对后加入的 train_gen_deep_nowide.py 单独审查。**实际退出码 0，但结果无效。** GLM 空响应被校验拒绝后，DeepSeek 同样返回非 JSON 的长段推理文字而非最终结构化报告；脚本 fallback 再次形成 pass=true/0 issues。推理文字提到缓存身份、空特征集合、缺失样本与加载器边界等风险，但不具备有效通过结论。

有效代码审查结论：独立静态审查发现高/中问题，未修复，状态为未通过。外部脚本名义退出码不得覆盖可复核的接口错误。原样提取任务保留历史缺陷，不应在此归档中修复；若需要可训练实现，应另立修改分支、保留备份、逐项修复并重新独立审查。


最终状态：七个代码文件原样归档可发布；本 REVIEW.md 可作为审查证据发布。glm_review.py 和文档备份均不在发布清单中。三轮脚本退出码均为 0，但均没有取得有效结构化通过结论；独立代码审查确认高/中问题未修复。没有双引擎网络不可用的退出码 2，因此没有宣称触发 Sonnet 兜底。

