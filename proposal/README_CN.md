# SetFit + BANKING77 proposal

## 文件

- `SetFit_BANKING77_Proposal.tex`：按所给模板撰写的英文正文；无 instructions 环境或模板说明。
- `references.bib`：三篇主要文献，以及数据、模型和官方代码来源。
- `neurips_2019.sty`：NeurIPS 2019 官方样式；模板原有的会议声明已在正文中替换为课程声明。
- `SetFit_BANKING77_Proposal.pdf`：编译后的预览。
- `data_audit.json`：本次实际下载并核查的原作者 CSV 的来源、SHA-256 与数据数量；不是模型训练结果。
- `SetFit_BANKING77_Chinese_Guide.md`：逐节中文解释、术语与公式、Colab 起步示例及后续实现顺序。

本版保留模板章节、字体和正文样式，去掉标题上下装饰线；标题明确为银行客户诉求分类。PDF 共四页：第一页为关键信息与文献，第二至三页为项目描述，第四页为参考文献。源码包不是已经完成的模型训练项目。

## 编译

将整个文件夹上传到 Overleaf，主文件选择 `SetFit_BANKING77_Proposal.tex`，编译器使用 pdfLaTeX。
本地具备正常 TeX Live 环境时，依次运行：

```bash
pdflatex SetFit_BANKING77_Proposal.tex
bibtex SetFit_BANKING77_Proposal
pdflatex SetFit_BANKING77_Proposal.tex
pdflatex SetFit_BANKING77_Proposal.tex
```

## 提交前必须填写的个人信息

1. 三位组员的姓名与 UNI；如果只有两人，删除第三人并重新分配任务。
2. 实际可用 GCP credit 余额。正文中的 USD 25 是计划给本项目的预算，不是对课程实际额度的声明。
3. 确认 Mentor 栏目前的 “We have no particular mentor” 是否符合小组情况。
4. 计划日期沿用此前 proposal 的 Nov 23 / Dec 5 / Dec 12 / Dec 17；这些是写入本稿的工作目标，不是本次独立核实的课程官方 deadline。
5. 若有外部合作者或与其他课程共享项目，在 Key Information 补充；没有则无需保留可选字段。

## 已明确的实验设置

- 原始 10,003 / 3,080 train/test 行；按 `.strip()` 后的精确文本去掉 4 条训练重复行，再去掉与 test 相同的 6 条训练文本，训练侧剩 9,993 行。
- 测试集保留全部 3,080 行（其中 1 条重复文本），不为提高分数删测试样本。
- 每类取 floor(0.1 × 清理后类别数) 作固定验证集，共 965 行；训练候选池 9,028 行，最小类别仍有 32 行。
- 8-shot / 32-shot 指训练标注数量；额外使用的 965 个验证标签已公开说明，不能称为总标注预算只有每类 8 / 32 条。
- 调参 seed 0；最终 seed 11 / 22 / 33；同 seed、同 k 下各方法训练数据相同。
- 同 seed 每类先打乱一次，前 8 行包含在前 32 行中。
- 三种主方法：Frozen+LR、CE tuning、SetFit。主结果共 3 × 2 × 3 = 18 组，其中 12 组需要微调 encoder；加 8 个学习率选择训练，总计 20 次 encoder 训练。
- 初始模型相同；不加载已用 BANKING77 标签微调过的 checkpoint。
- CE head 在训练 seed 下随机初始化；使用同一 pooled、归一化句子表示。SetFit 与 Frozen 使用相同类型与参数的 logistic regression，因此两者是主要消融比较。CE vs SetFit 是整个训练流程的比较。
- 句子对训练的 cosine MSE 是原文思路的明确实现；固定 4 正 + 4 负 / anchor 的构造与步数上限是本项目的受控设置，不声称逐项复现原论文默认超参数。
- 正样本在同类中均匀无放回抽 4 条，排除自己；负样本先均匀抽其他类别，再均匀抽句子，直到获得 4 个不同伙伴。A→B 与 B→A 可以同时出现。按固定训练 seed 预生成后，每轮打乱，迭代到更新数上限。
- Encoder 优化 AdamW；CE 的线性 head 与 encoder 使用同一 LR；LR 指 sklearn logistic regression，不是 learning rate。
- 最大 128 tokens、动态 padding、FP16、batch=32；CE 为 32 条句子，SetFit 为 32 对即 64 条句子。若 pilot 显存不足，用 micro-batch=16、梯度累积=2 保持有效 batch 不变。
- 固定最大 300 / 1,000 encoder updates（8 / 32-shot）；验证间隔 50 / 100。以验证 macro-F1 选 checkpoint，同分选更早；冻结最终超参数后运行最终三 seeds。
- 同步记录完整流程训练时间（含训练中验证和头拟合）及独立测试推理时间、峰值 allocated GPU memory；在计时边界同步 CUDA。相同步数不等于相同 FLOPs，不能据此直接宣称 SetFit 更省计算。
- 不比较生成回复、不做未知类别检测，不把公开英文数据结果推广成银行生产效果。

## 资源估算的含义

核心 10 GPU-hours = 学习率选择 2h + 最终训练和 checkpoint 评价 4h + 其余 embedding/test 评价 1h + setup/debug 3h。可选负样本实验另留 2h，上限 12h。
其中学习率选择相当于约 15 分钟/次；最终 encoder 微调及训练内验证约 20 分钟/次。这是按小型 encoder、短输入和固定步数编制的预算，不是实测速度。
Pilot 必须覆盖一次 CE 和一次 SetFit，复用 8 个调参训练中的设置，不另启动超出表格的隐藏 sweep；测 step 时间、验证耗时和显存后，按该预算决定保留两个数据规模或仅完成 8-shot 主实验。
USD 2/小时是对完整云实例的预算假设，不是 GCP 报价；12h + USD 1 storage = USD 25。

## 文献与报告措辞

三篇主要文献分别支持 SetFit 方法、BANKING77 与强冻结 baseline、细粒度混淆分析。第三篇发表于 COLING 2025，满足近两年的时间范围。
“Papers you read” 是模板要求的标题；组员须实际阅读并核对这些总结，再以小组名义提交。该稿没有编造已跑出的模型结果。
困难负样本只是可选扩展；未复现 Clear Up Confusion 的 LLM 数据生成流程，也不把 SetFit 应用于 banking 声称为首次创新。
