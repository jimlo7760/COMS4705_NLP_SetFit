# SetFit + BANKING77：proposal 逐节解释与从零开始的操作指南

这是一份配合英文 proposal 阅读的中文学习指南。英文 proposal 是拟开展的实验计划；这里的数据清洗数量来自此前对官方 CSV 的审计，不是模型实验结果。训练分数、运行时间和 GPU 内存尚未实测。代码示例用于第一天的数据准备和冻结模型基线，不是已经完成的三方法训练项目。

## 先把整个项目串成一句话

我们想知道：只有少量带答案的银行客户提问时，是否值得再训练一个已经预训练好的句子编码器？我们使用同一个基础模型、完全相同的训练句子，比较“不更新编码器”“直接按分类答案微调”“先按同类/异类句子对微调，再训练分类器”三种做法；再看哪些客户诉求容易混淆、哪些错误得到改善，以及额外训练花多少资源。

你们不用发明新模型，也不用从零教模型英语。现成模型负责提供起点；你们负责把数据、训练目标、比较条件、评价和分析做完整。英文标题改为 **Few-Shot Banking Customer Intent Classification: A Controlled Study of SetFit**，中文是“少样本银行客户诉求分类：SetFit 的受控比较研究”。

这里 intent 是“客户想办理或询问什么”，不是“银行有什么意图”。例如“卡一直没送到”“卡支付被拒绝”属于不同诉求类别。Few-shot 表示每个类别只有少量用于训练的示例；controlled study 表示比较时尽量固定其他条件。

## 一、每个 section 在整体逻辑中做什么

| 模板部分 | 需要回答的问题 | 本项目的具体答案 |
|---|---|---|
| Key Information | 谁做、由谁指导？ | 三名组员真实姓名及 UNI；没有指定 mentor 就如实填写。 |
| Papers you read | 别人已经做过什么？ | SetFit 给训练方法；BANKING77 论文给任务、数据及强冻结基线；2025 年文献提示分析细粒度类别混淆。 |
| Questions | 读完文献后，究竟要问什么？ | 同样的模型和训练句子，SetFit 相对冻结表示有多大增益？8 条/类与 32 条/类是否不同？ |
| Positioning and motivating your idea | 为什么做，和已有工作是什么关系？ | 复现两阶段思想，补充匹配比较和错误分析，帮助判断额外训练是否值得。 |
| Goal | 最后要交付什么证据？ | 三方法的分数、变化幅度、成本，以及混淆和具体错误分析。 |
| Task | 一个输入对应怎样的输出？ | 一句英语客户请求 → 77 个已知类别中的一个。 |
| Data | 用哪些数据，怎么分？ | 官方 BANKING77，训练侧去重及去交集；固定验证集；少样本训练子集；官方测试集。 |
| Methods | 如何得到预测结果？ | 同一 MiniLM 起点，Frozen+LR、CE tuning、SetFit 三条流程。 |
| Evaluation | 怎样判断好坏？ | macro-F1、accuracy、类别表现、三次重复、耗时、内存和错误分析。 |
| Baselines | 用什么参照才能解释结果？ | 自己运行冻结模型与普通监督微调，不能只抄论文分数。 |
| Justification | 为什么符合课程项目？ | 使用课内的表示、优化、Transformer、微调和分类；有实际训练及编程工作。 |
| Team, compute, timeline | 谁做、要花多少、何时完成？ | 明确分工、20 次核心编码器训练的预算、先计时再决定是否缩小范围。 |

这条逻辑是：**实际问题 → 文献提供方法与疑问 → 明确研究问题 → 设计对照 → 运行训练 → 用指标和错误解释结果 → 回答额外训练是否值得。**

## 二、Literature review：三篇论文分别帮我们做什么

### 1. SetFit 原论文：给我们主要训练思路

[Efficient Few-Shot Learning Without Prompts，2022](https://arxiv.org/abs/2209.11055) 提出 SetFit，名称展开是 Sentence Transformer Fine-tuning。主要思想是：少量分类样本也能构成很多同类/异类句子对，先用这些关系调整句子表示，再训练分类头。不需要写分类 prompt，也不需要调用生成式大模型。

我们复现的是这个训练原则：**句子对训练编码器 → 冻结编码器 → 训练分类器**。我们不会宣称完全复现原论文全部数据集、全部骨干模型或每一个表格的数值。我们的固定四正四负采样、训练上限、学习率搜索等，是本项目为了控制范围明确选定的实验协议，不应写成原论文唯一的标准配置。

官方资料：[SetFit GitHub](https://github.com/huggingface/setfit)、[官方 Quickstart](https://huggingface.co/docs/setfit/quickstart)、[分类 notebook](https://github.com/huggingface/setfit/blob/main/notebooks/text-classification.ipynb)。可以先跑通官方小例子了解接口，再实现 proposal 的指定流程；直接沿用默认 Trainer，不一定得到我们写定的配对数量和训练步数。

### 2. BANKING77 论文：告诉我们为什么必须认真做冻结基线

[Efficient Intent Detection with Dual Sentence Encoders，2020](https://aclanthology.org/2020.nlp4convai-1.5/) 引入 BANKING77，并研究利用预训练句子编码器做高效客户诉求识别。论文中的 USE/ConveRT 等模型，与我们选的 MiniLM 不是同一个模型。它给我们的启发是：已经学到良好句子表示的编码器，搭配简单分类器也可能很强。

因此，不能只展示“SetFit 分数不错”，还要问“同一个 MiniLM 不微调，只训练分类器，能做到什么程度？”只有比较后才能判断增加的表示训练是否带来收益。

### 3. 2025 年混淆研究：给我们分析方向，不是要求复现第二套大项目

[Clear Up Confusion: Iterative Differential Generation for Fine-grained Intent Detection with Contrastive Feedback，2025](https://aclanthology.org/2025.coling-main.151/) 研究细粒度意图之间的混淆，涉及生成训练数据和反馈机制。我们借鉴“相近类别需要单独检查”的问题意识，不做它的整套生成框架，不需要付费 LLM API。

它也满足模板中需要近期论文的要求。文献不是只放在 References 里：它应解释为什么选择冻结基线、为什么检查相近类别、为什么考虑困难负样本。

## 三、Questions 和 Goal：真正研究什么，不研究什么

**主问题：**在每类只有 8 条训练句子时，SetFit 相对 Frozen+LR 的 macro-F1 有多少变化？这种变化在不同随机抽样下是否一致？

**第二个问题：**每类训练数据从 8 条增加到 32 条后，这个差值变大还是变小？这是同一个方法在不同数据量下的表现，不是说训练时只有 8 个类别。

**分析问题：**如果整体有提升，是否也减少了原本容易混淆的类别之间的错误？例如模型是否不再频繁把“卡未送到”判断成“查询卡的配送状态”？这里的例子是解释性的，实际分析的五对类别必须由验证集预测按规则选出。

“a strong frozen representation leaves little room for adaptation”意思是：基础模型可能已经把不同诉求区分得不错，再训练并不一定明显改善。因此结果可以是改善、持平或变差。若 SetFit 变差，应先检查训练代码、损失、标签和过拟合；确认实现合理后，再讨论在本协议下额外微调没有稳定优势。不能由三次小实验推断“SetFit 普遍无效”。

这个项目也不能证明模型是否真正“理解语言”。它测的是指定数据分布上的分类质量、错误结构和成本。它与前面讨论的 SNLI shortcut 项目是不同研究问题。

## 四、BANKING77 是什么，以及每一段 Data 的意思

### 1. 77 指 77 个客户诉求类别

一行数据通常包含 text 和 label：text 是客户的英语句子，label 是人工给定的正确类别。模型训练时看到一部分“句子 + 正确答案”，之后对没有见过的句子预测类别。

例如，句子 “My card still has not arrived.” 可以对应 card_arrival。类别在代码里可能用数字表示，但数字只是编号，例如 11 并不表示风险等级或金额。需要保存编号与类别名称的映射。

原始数据有 10,003 条训练行和 3,080 条测试行，共 77 类。测试集每类 40 行。它不是完整银行业务知识库，也不包含让模型生成客服回复所需的答案。

[Hugging Face 数据页面](https://huggingface.co/datasets/PolyAI/banking77) 可以看类别和例句；[作者数据仓库](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) 提供 CSV。

### 2. CC BY 4.0 是使用许可，不是模型或数据处理方法

它是 Creative Commons Attribution 4.0 International：允许分享和改编，但应署名、给出许可链接，并说明做过修改。项目中应引用原论文和数据来源，README 中写明使用 BANKING77、许可及训练侧清洗操作。[许可说明](https://creativecommons.org/licenses/by/4.0/)

### 3. exact text after outer-whitespace stripping 是怎样比较句子

只删除开头和结尾的空白字符，之后比较整条文本是否完全相同。例如 `"  Where is my card? \n"` 变成 `"Where is my card?"`。不删除句子内部空格，不统一大小写，不去掉问号，也不判断两个不同写法是否语义相同。

因此 `"Where is my card?"` 和 `"where is my card?"` 仍然不同。这是保守的完全重复检查，不是语义去重。hash 是文本的指纹，方便做交集检查；它不理解意思，也不是标签。

### 4. 四条训练重复、六条跨集合相同文本意味着什么

此前对官方 CSV 的审计发现：训练集有四条可以删除的冗余行；相同文本没有出现冲突标签。删除后再移除与测试文本重合的六条训练文本，得到 `10,003 - 4 - 6 = 9,993` 行。

训练集内重复可能使某些句子被额外强调；训练和测试句子相同，会让“对未见句子泛化”的测量受到影响。我们在训练侧处理，保留完整官方测试集。仅用测试文本检查完全相同的字符串，不使用测试标签决定训练、不根据测试成绩选择方法。

这种使用测试文本的去交集规则应公开记录；它也不保证消除了所有潜在泄漏，例如相似改写或模型预训练中见过的内容。测试集本身还有一条重复文本记录，我们保留并说明，避免擅自改变官方测试集。

“Published scores provide background”指：别人论文的结果能说明任务的大致背景，但我们的起点模型、清洗、验证集和抽样规则不同，不能拿自己的数字与论文数字直接比较后宣称胜过原论文。主要比较必须来自我们自己在相同条件下运行的三个方法。

### 5. 训练、验证、测试三个集合的职责

| 集合 | 本项目数量 | 用途 | 是否直接用于优化参数 |
|---|---:|---|---|
| 清洗后的训练侧数据 | 9,993 | 划分验证集及候选训练池 | 不整批拿去训练 |
| 固定验证集 | 965 | 选学习率、选 checkpoint、预选混淆类别对 | 不进入训练损失，但答案参与模型选择 |
| 候选训练池 | 9,028 | 从每类抽 8 或 32 条 | 仅抽中的部分用于训练 |
| 8-shot 训练集 | 616 = 77 × 8 | 训练三种方法 | 是 |
| 32-shot 训练集 | 2,464 = 77 × 32 | 训练三种方法 | 是 |
| 官方测试集 | 3,080 | 方法和配置确定后进行最终评价 | 否 |

验证标签为什么要单独说明？假设每类训练只用 8 条，但选模型还看了 965 条带答案的验证数据，就不能宣传为“只标注了 616 条数据就完成整个项目”。诚实描述是“训练受限为每类 8 条，另外有共同验证集”。这个设计可行，但不属于严格的总标注预算比较。并且从完整带标签数据按类别抽样，本身是少样本场景的离线模拟。

### 6. 公式、seed 和 nested 分别是什么意思

清洗后第 c 类有 n_c 条。`floor(0.1*n_c)` 表示取其 10% 并向下取整。例如有 113 条，就取 11 条验证。逐类这么做，合计得到 965 条。候选池中最少的类别仍有 32 条，因而能够支持每类 32 条的实验。

随机抽样需要一个随机数序列。seed 是生成这个序列的起点；相同数据顺序、算法和 seed 可以复现同一抽样。2026、0、11、22、33 没有特殊的统计意义，是在看最终结果之前预先规定的不同整数。不能先试很多 seed，再挑表现好的三个。

- 2026：固定验证划分；另在人工错误抽样中作为固定种子使用。
- 0：调参阶段的训练抽样与随机过程，不计入最后三次结果均值。
- 11、22、33：三次最终重复，同时控制抽样、训练打乱及模型随机过程。

每次重复，对每一类的候选句子打乱一次。取前 8 条是 8-shot；取前 32 条是 32-shot。因此前 8 条一定在那 32 条中，这叫 nested，能让数据量比较更容易解释。三个方法必须读取完全相同的名单，而不是各自随便抽一套。

“unused pool supplies no text, pairs, or labels to training”指剩余候选句子既不参与分类损失，也不参与句子配对或无标签训练。不能说只用 616 条训练，却偷偷把其他句子拿来做训练配对。

### 7. tokenizer、padding、128 tokens 和 truncation

tokenizer 是配套的文本转换工具，把句子分为 token 并变成整数编号；token 不一定等于一个英文单词。我们使用 MiniLM 自己的 tokenizer，不重新发明分词器。

同一 batch 中句子长短不同，通常要用 padding 补到同样长度；dynamic padding 是每批补到本批最长句子的长度，避免所有短句都补到 128。attention mask 标明哪些是真实 token、哪些是补位。

128 是本项目选定的最大 token 长度，包含 tokenizer 产生的特殊 token；长句超出后截断。它不是 128 个单词，也不是该模型唯一允许的长度。记录截断频率，是为了知道多少样本可能丢失句尾信息。标点可能帮助理解，因此不做无必要的删标点处理。

## 五、MiniLM、Sentence Transformer、SetFit 的关系

你可以先分清三个名词的职责：

| 名称 | 在项目里是什么 | 做什么 |
|---|---|---|
| all-MiniLM-L6-v2 | 已训练好的具体 checkpoint | 提供初始神经网络权重，把句子变成向量 |
| Sentence Transformer | 句子编码模型的组织方式及相关库 | 把 token 表示汇聚为整句表示，提供加载、编码等功能 |
| SetFit | 训练方法及其开源实现 | 用句子对调整编码器，再训练分类头 |

`sentence-transformers/all-MiniLM-L6-v2` 是 Hugging Face 上的仓库标识，前半段是组织名，后半段是模型名。它有六层 Transformer、约 2,270 万参数，输出 384 维句子向量。向量是 384 个数字组成的一行，并不是 384 个类别。

模型先产生各 token 的表示；attention-masked mean pooling 对有效 token 的表示取平均，排除 padding。unit normalization 再把向量长度缩放为 1，让余弦相似度计算一致。

“从 SetFit 开始”可以理解成“采用 SetFit 方法”，但仍要选择一个已有编码器作为起点。我们不是先找一个已经在 BANKING77 上微调好的分类器，否则可能不知道它见过哪些数据，影响比较。

## 六、三种方法到底怎样训练和预测

所有方法都从同一份初始 MiniLM 权重重新开始。不要先训练 CE，再拿已经训练过的 CE 权重去做 SetFit。

### A. Frozen+LR：句子编码器不动，只学分类规则

1. 将 616 条训练句子输入 MiniLM，得到 616 × 384 的向量矩阵。
2. 保持 MiniLM 参数不变。
3. 用这些向量和 616 个类别标签训练多分类 logistic regression。
4. 新句子经过同一编码器，分类器给出 77 类的分数/概率，选最高的一类。

Frozen 指编码器被冻结，不是整个系统完全不训练。这里的 logistic regression 虽叫“回归”，实际用于分类，不是在预测收入或股票价格。

可把分类头理解为 `s = W h + b`：h 是 384 维句子向量；W 大致是 77 × 384 的权重；b 是 77 个偏置。经 softmax 得到 77 类概率。训练就是调整这套分类权重，让正确类别概率更高。

### B. CE tuning：按分类答案同时调整编码器和分类头

输入仍是原始句子，但在编码器后接一个 77 类线性分类头，编码器和分类头一起训练。正确类别是 y 时，交叉熵损失为 `-log(p_y)`。

例如正确类别概率从 0.1 变成 0.8，损失从约 2.303 降为约 0.223。梯度把“哪里该调整”的信号传给分类头，也传进 MiniLM。这里 CE 是 cross-entropy，不是一种单独的预训练模型。

它代表常规监督微调：每条句子直接提供自己的正确类别。不先生成同类/异类训练对，也不在最后另训 sklearn 分类头。

### C. SetFit：先调整句子表示，再训练与 A 相同的分类器

第一阶段：从同一少量训练句子构造配对。相同类别标为 1，不同类别标为 0。把配对中的两句都送入同一个共享权重的编码器，按相似度损失调整 MiniLM。

第二阶段：冻结已经调整过的 MiniLM，重新为原始的 616 条句子生成向量，用这些向量和原始 77 类标签训练与 A 同设置的 logistic regression。

预测时只输入一句新请求：编码器 → 向量 → 分类器 → 类别。无需为每个用户请求寻找一对句子，也无需使用训练对中的 0/1 作为最终类别。**配对的 0/1 是第一阶段的关系标签，最终输出仍是 77 类中的一类。**

为什么 A 是关键对照？A 和 C 起点相同、第二阶段分类器相同；差别是中间有没有调整编码器。为什么还需要 B？因为要知道，使用 SetFit 整套流程，与直接做普通微调相比怎样。但 B 的分类头训练方式也变了，所以 B 与 C 的差异不能全部归因于“配对损失更好”。

## 七、句子配对和公式逐项拆解

anchor 是当前拿来找搭档的训练句子。假设它属于 card_arrival：同类搭档可以是另一条表达卡未送到的句子；异类搭档可以来自支付失败等类别。“正”“负”描述类别是否相同，不是句子情绪正面/负面。

每个 anchor 选四个不同的同类搭档，不能选自己；再选四个不同的异类搭档。8-shot 每类有八条，除自己还剩七条，所以能够选四个同类搭档。

异类抽样先从另外 76 个类别均匀选一个，再在那个类别内选一句。为了四个负搭档互不相同，抽到已选搭档时重抽。整张配对表打乱；A→B 与 B→A 都可能出现，它们是相关训练样本，不是独立的新标注信息。

训练句子数是 77k，每句产生八对：

| 设置 | 训练句子 | 正对 | 负对 | 总对数 |
|---|---:|---:|---:|---:|
| k=8 | 616 | 2,464 | 2,464 | 4,928 |
| k=32 | 2,464 | 9,856 | 9,856 | 19,712 |

配对数量多，不代表人工新标注了这么多句子；只是把已有类别标签转换为句间关系。我们选“四正四负”是一个固定且容易复现的项目配置，不是理论上必须如此。

公式是：

`L_pair = (1/B) * Σ_i [cos(f_theta(x_i), f_theta(x'_i)) - z_i]^2`

- x_i 和 x'_i：第 i 对中的两句文本。
- f_theta：参数为 theta 的句子编码器。
- cos：两个输出向量的余弦相似度，范围为 -1 到 1。
- z_i：同类为 1，异类为 0。
- B：当前 batch 中的句子对数量，正常为 32，最后一批可能不足。
- 平方、相加、除以 B：平均每一对的相似度预测误差。

同类相似度为 0.4，则损失是 `(0.4-1)^2=0.36`；异类相似度为 0.8，则是 `(0.8-0)^2=0.64`。调整网络，希望前者趋近 1、后者趋近 0。这里异类目标是 0，不能说它要求所有异类都变成 -1；也不保证所有同类都会达到相似度 1。

## 八、那一长串训练设置分别在干什么

它们是为了让“训练怎么运行”可以被复现，不是要在 proposal 阶段证明每一个值最优。

| 设置 | 初学者解释 | 本项目为什么这样规定 |
|---|---|---|
| AdamW | 根据梯度更新神经网络参数的优化器 | 两种微调方法使用相同基础优化工具 |
| learning rate | 每次参数更新的步子大小 | 候选 0.00001 与 0.00003；太大可能破坏表示，太小可能学得慢 |
| weight decay=0.01 | 优化时对部分权重加入收缩作用 | 常用的正则化起点；不等于把模型强行拉回最初 checkpoint |
| gradient clipping=1.0 | 梯度整体范数过大时缩放 | 减少偶发过大更新造成的不稳定 |
| 10% warmup | 前 10% 的更新逐步增大学习率 | 避免训练一开始就用较大步长 |
| linear decay | warmup 后学习率逐步下降 | 后期更新更小 |
| FP16 | 支持的 GPU 上使用混合精度 | 节省显存、可能加速；不是把所有计算和参数永远都设为半精度 |
| batch size=32 | 每次取一小批样本计算梯度 | CE 是 32 句；SetFit 是 32 对，即最多处理 64 句 |
| update | 一次参数更新 | 不等于训练整份数据一遍 |
| checkpoint | 某一训练时刻保存的模型状态 | 让我们能保留验证集表现最好的版本 |

8-shot 最多 300 次参数更新，32-shot 最多 1,000 次。前者 warmup 30 次，后者 100 次。跑到配对表或句子表末尾但还没达到上限时，重新打乱、继续遍历。训练循环应按“更新数”结束，不能只跑一遍就停止。

CE 每次 32 句，SetFit 每次 32 对，两者的实际计算不相等。即使上限相同，也不能宣称“完全同计算量公平比较”。所以同时报告真实训练时间和显存。作为近似理解，8-shot CE 的 300 次更新最多接触约 9,600 个句子位置，SetFit 约 9,600 个配对位置；反复看到同一句话是正常的。

8-shot 每 50 次更新检查验证集，32-shot 每 100 次检查。每次 SetFit 检查需要先用当前编码器提取训练向量、只在训练标签上拟合分类器，再预测验证集。验证标签只用来计算分数，不进入分类器拟合。选择验证 macro-F1 最高的 checkpoint；相同分数保留更早版本。

Frozen+LR 不需要这个神经网络微调循环，但也要拟合分类器。logistic regression 使用 lbfgs 求解器、C=1、有 intercept、最多 2,000 次求解器迭代。C 是正则化强度的倒数：通常越小，约束越强。这里 LR 指 logistic regression；learning rate 是另一个概念，不能混淆。2,000 次是分类器求解器上限，不是 MiniLM 更新 2,000 步。

调参只做一个小网格：两种编码器微调方法 × 两个训练规模 × 两个学习率 = 八次训练，固定 seed 0。每种“方法+数据量”用验证集选出一个学习率，然后固定，进行 11/22/33 三个最终重复。不能每看到测试结果不好就换学习率。

| 阶段 | 数量 | 是否训练编码器 |
|---|---:|---|
| seed 0 调参 | 2方法 × 2规模 × 2学习率 = 8 | 是 |
| 最终 CE 与 SetFit | 2方法 × 2规模 × 3seeds = 12 | 是 |
| 最终 Frozen+LR | 1方法 × 2规模 × 3seeds = 6 | 否，只提取向量并训练分类器 |
| 最终结果表里的设置 | 3方法 × 2规模 × 3seeds = 18 | 其中12个需编码器训练 |

因此“18 个最终设置”和“20 次核心编码器训练”并不冲突：20 = 8 次调参 + 12 次最终训练。两次计时 pilot 可直接作为八次调参中的两次，不额外重复算。

## 九、Evaluation：不是只比较分数，怎样真正分析

### 1. 自动指标

Accuracy 是预测正确的条数除以总条数。假设 100 条对了 80 条，就是 80%。

对一个类别，precision 是“预测成此类的样本中有多少真是此类”；recall 是“真实属于此类的样本中找对了多少”；F1 是二者的调和平均 `2PR/(P+R)`。macro-F1 先分别算 77 类的 F1，再简单平均，每类占相同权重。它帮助观察类别之间的表现，而不是只看整体对了多少。某类没有可定义的 F1 时按预先约定记 0。

测试集各类等量，accuracy 本身也很有解释性。两个指标一起报告，是为了避免只用一个总分掩盖某些类别的精确率/召回率变化，不必为了显得复杂而堆所有可用指标。

### 2. 三个 seed：报告变化，而不是挑一次好结果

每个方法得到三组最终分数，报告均值和样本标准差，也列出每个 seed 的 SetFit 减 Frozen 差值。例如 11 号 seed 的两个方法使用同一批训练句子，这是一组配对比较。

这些重复同时改变训练抽样和训练随机性，所以波动包含“拿到哪些训练句子”的影响，不只是神经网络初始化。三次重复不足以支撑很强的统计显著性结论；应展示数值、方向和范围，避免把很小差异写成确定突破。

### 3. 混淆矩阵和五对易混类别

混淆矩阵的行是真实类别，列是预测类别。C_ij 表示真实为 i、却被预测成 j 的条数。对一对类别，用 C_ij+C_ji 找它们互相误认的数量。

先用 seed 0、8-shot 的 Frozen 基线在验证集上的预测选出五对最容易互相误认的类别，数量相同按类别名称排序打破平局。这个步骤必须在查看最终测试对比前确定，不能事后只挑 SetFit 改善最大的五对。

最终测试时看这五对在不同方法下是否改善。对类别 i、j，双向混淆率是“i误成j的条数 + j误成i的条数”除以“真实属于i或j的测试样本条数”。它不是这两类所有错误的比率，例如 i 错成第三类不在分子中。每对也应显示分子、分母，避免比例缺乏上下文。

如果总分提高，但最难区分的类别没改善，结论应说明提升可能来自其他类别。即使五对改善，也要同时呈现整体结果，不能只展示有利局部。

### 4. 具体句子错误分析

固定 seed 11，对每个测试样本比较 Frozen 与 SetFit 的正确性，可分四组：两者都对；Frozen 对而 SetFit 错；Frozen 错而 SetFit 对；两者都错。各组固定抽样，最多十条；某组不足十条就全取。

由组员阅读文本、正确类别和两个预测，标记可能原因：类别含义接近、表达改写/措辞、句子缺信息、其他。再挑有代表性的例子写入报告。不要只选 SetFit 改对的例子，也要检查它新引入的错误。

文字应基于实际证据，例如“在被修正的样本中，若干句子使用了训练例句未直接出现的措辞”；不要凭分数就写“模型真正理解了金融语义”。这些原因是人工解释，不是内部机制的证明。

### 5. 训练成本

分别计时编码器训练、验证/分类器拟合、向量提取和最终预测，注明是否包含下载与首次初始化。记录 peak GPU memory。Frozen 方法的编码也消耗计算，不能当作绝对零成本；它只是没有编码器反向传播训练。

## 十、可选的 hard-negative 实验究竟是什么

同类对是 positive；不同类对是 negative。random negative 从其他类别随机抽句子，可能非常容易区分；hard negative 是虽然类别不同、当前表示却很相似的句子。

可选实验用最初冻结的编码器，在已经抽中的训练句子中找余弦相似度高的异类邻居。将每个 anchor 四个随机负搭档中的两个替换为这样的困难负搭档，仍保留四个正对、四个负对及相同更新上限。不要到未使用候选池或验证/测试集中找搭档；也要去掉重复搭档。

动机是训练模型更关注相近客户诉求的边界。但这也可能放大噪声或本来有歧义的标签，不能保证提升。使用原始冻结模型选邻居，是为了规则固定且便于实现，而不是每一步重新挖掘困难样本。

step-matched 只表示优化器更新次数相同。pair-matched 表示总配对数也相同；并不自动保证所有运行耗时完全相同。核心项目可以没有这个扩展，先把三方法比较和分析做完。若增加它，两个规模 × 三个 seed = 六次额外训练，复用已选的 SetFit 学习率，不再另开大范围调参。

我们主要新增的是受控比较、数据量比较、混淆与错误分析；困难负样本是有时间才做的额外方法变体。因此既不是“发明全新架构”，也不是原样运行一个现成 notebook 就结束。

## 十一、Justification：为什么符合课程要求

英文已按模板完整名称改写为五项：

1. **Relevance to course material**：The project builds on 句子表示与优化、分类与评价、Transformer 编码器及预训练模型微调。我们不会说项目实现了没有用到的 LoRA、prefix tuning 或 KL penalty。
2. **Technical component**：自己写训练循环、配对采样和损失、普通监督基线，以及划分和评价代码。下载预训练权重、复用 logistic regression 和指标库是合理起点，但不能把主要实验工作写成自己发明的组件。
3. **Language/code/math data**：本项目实际使用的是英语客户文本，输出诉求类别，不需要同时具备代码或数学数据。
4. **Feasibility**：小编码器、有限数据、两种训练规模和明确更新上限，将规模限制在单 GPU 的计划内，三人各自有代码和写作任务。
5. **Novelty is not required**：复现 SetFit 原理，并补充匹配的冻结对照、嵌套训练数据量比较、混淆及错误分析；不要求新架构或 SOTA。扩展贡献应据实表达，不能声称首次在 BANKING77 上使用 SetFit。

模板的章节与关键词、三篇文献、近期文献、输入输出例子、数据规模、实现/下载边界、数值评价、基线、五项理由、分工、计算预算和时间表均已覆盖。文献部分一页以内，项目描述两页以内；References 单独排。姓名、UNI 和真实剩余 GCP credits 必须由组员补入，不能用虚构数字替代。

## 十二、GPU-hours 是什么，10 小时是怎样得出的

GPU 是适合神经网络并行计算的硬件，显存是它能同时容纳数据和模型的工作空间。GPU-hours 是资源使用量：一张 GPU 用两小时，就是 2 GPU-hours；两张各用两小时，就是 4 GPU-hours。它不是 CPU 运行时间，也不代表报告写作时间。

我们只用一张 GPU 时，10 GPU-hours 大致对应累计十小时的 GPU 占用。实例开着但没有训练也可能收费；账单还可能包含 CPU、内存、磁盘、地区和机器类型，因此费用不能简单只看“模型真正计算了几分钟”。

proposal 使用的是初始预算：调参 2 小时，最终训练及期间验证 4 小时，剩余向量提取及测试 1 小时，环境及调试余量 3 小时，共 10 GPU-hours。可选六次扩展再留 2 小时，最多 12。这个划分是计划额度，不是已经证明的运行时间；T4 和 L4 的速度也不能视为相同。

怎样把估算变可靠：

1. 在实际将使用的 GPU 上完成一次 CE 和一次 SetFit 计时，可计入八次调参。
2. 分开测下载、训练、验证拟合分类器的时间，记录实际 batch 和序列长度。
3. 按每个规模的更新数、每种方法的剩余运行次数进行外推，加上初始化和验证开销。
4. 扩展前再检查实际剩余预算。超过 10 小时的核心计划，就先做三个方法 × 8-shot × 三个 seed；32-shot 作为优先补充，并在最终报告中据实更新范围。

USD 25 是支出上限的假设：最多 12 小时 × 假定整机不高于 2 美元/小时，再加约 1 美元存储。它不是云服务当前报价。真正开机前看团队账户里的地区、GPU 配额、机器报价和余额；如果整机价格超过假设，就重新算，不能照抄 25 美元。免费 Colab 是否能一直分到 GPU 也不能保证。

## 十三、具体从哪里开始：第一天的可执行操作

### 第一步：先看三个网页，分清“数据、模型、方法”

打开 BANKING77 的 Hugging Face 页面，查看 Dataset card 和数据预览。选 train，读十几条文本和标签；再查看 test 的数量。这一步的目标是能自己解释“输入一句话，输出一个类别”，不是点击某个在线聊天按钮。

打开 all-MiniLM-L6-v2 页面，查看 Model card、使用示例和 Files 列表。文件夹里是模型配置、tokenizer 配置与权重，不需要逐个手动下载。公开模型在 Colab 中由加载函数自动取得；这里不需要付费推理 API，也不需要在网页点 Deploy。

打开 SetFit Quickstart，看两阶段训练示例，并保留 GitHub 链接。它是参考实现，不是第三份需要下载的训练数据。先理解结构，暂时不要试图同时读完整个仓库。

### 第二步：建立 Colab notebook

打开 [Google Colab](https://colab.research.google.com/)，新建 notebook，命名为 `01_data_and_frozen_baseline.ipynb`，保存到自己的 Google Drive。通常在“运行时/Runtime → 更改运行时类型/Change runtime type”选择 GPU；界面名称可能因语言和版本稍有不同。

第一个代码单元检查硬件：

```python
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

输出 False 时，CPU 仍能检查数据和跑小规模编码，但不要用 CPU 的速度估计 GPU-hours。并不是必须在第一天就开 GCP 虚拟机。

### 第三步：安装所需 Python 库

单独一个单元：

```python
%pip -q install sentence-transformers datasets scikit-learn pandas matplotlib
```

先使用 Colab 已有的 torch，不要为了这个项目无差别升级所有底层包。若安装明确提示重启 session，先重启，再从 import 开始运行。跑通后记录实际版本；本文没有预先验证一组版本组合，不能把它当作已经测试过的环境锁定文件。

只有要运行官方 SetFit 快速示例时，再在独立试验 notebook 中安装 `setfit`。核心自写训练循环并不一定依赖 SetFit Trainer；它依赖 PyTorch、编码器、tokenizer 和我们自己实现的损失。独立 notebook 可以减少修改官方演示时影响正式实验环境。

### 第四步：下载数据并对齐清洗来源

用 Hugging Face 的标准接口查看数据：

```python
from datasets import load_dataset
bank = load_dataset('PolyAI/banking77')
print(bank)
print(bank['train'][0])
print(bank['train'].features)
```

它会自动下载，不需要在网页手工复制一万句。若 label 是 ClassLabel，可以用 `bank['train'].features['label'].names` 查看编号与类别映射。

为严格对齐 proposal 已审计的数据源，下面正式准备使用作者的原始 CSV，而不是混用不同版本或不同列名的数据：

```python
import pandas as pd
import numpy as np
from pathlib import Path

root = Path('/content/banking77_project')
(root / 'data').mkdir(parents=True, exist_ok=True)
(root / 'results').mkdir(exist_ok=True)
base = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/'
train_raw = pd.read_csv(base + 'train.csv')
test_raw = pd.read_csv(base + 'test.csv')
print(train_raw.columns, train_raw.shape, test_raw.shape)
train_raw.to_csv(root / 'data/train_original.csv', index=False)
test_raw.to_csv(root / 'data/test_original.csv', index=False)

train = train_raw.rename(columns={'category': 'label'}).copy()
test = test_raw.rename(columns={'category': 'label'}).copy()
train['row_id'] = np.arange(len(train))
test['row_id'] = np.arange(len(test))
train['text'] = train['text'].str.strip()
test['text'] = test['text'].str.strip()

conflicts = train.groupby('text')['label'].nunique().gt(1).sum()
redundant = train.duplicated('text').sum()
overlap = set(train['text']) & set(test['text'])
print('label conflicts:', conflicts)
print('redundant train rows:', redundant)
print('distinct train/test overlaps:', len(overlap))
assert conflicts == 0
clean = train.drop_duplicates('text', keep='first')
clean = clean.loc[~clean['text'].isin(overlap)].copy()
print('clean:', len(clean), 'classes:', clean['label'].nunique())
assert len(clean) == 9993 and clean['label'].nunique() == 77
```

这段示例用字符串集合做完全相同文本的交集检查，与 hash 检查的目的相同；正式数据审计文件还应保存原始下载文件的 SHA256，以便来源更新时发现差异。不要把 `assert` 改成其他数字来掩盖来源变化；如果不一致，先检查下载版本、列名及清洗顺序。测试标签没有参与上述过滤。

### 第五步：固定验证集和嵌套训练名单

```python
rng_dev = np.random.default_rng(2026)
dev_ids = []
for label, part in clean.groupby('label', sort=True):
    ids = rng_dev.permutation(part.index.to_numpy())
    dev_ids.extend(ids[:len(part) // 10])
dev = clean.loc[dev_ids].copy()
pool = clean.drop(index=dev_ids).copy()
assert len(dev) == 965 and len(pool) == 9028
assert pool.groupby('label').size().min() >= 32
dev.to_csv(root / 'data/validation.csv', index=False)

def make_nested_samples(seed):
    rng = np.random.default_rng(seed)
    chosen8, chosen32 = [], []
    for label, part in pool.groupby('label', sort=True):
        ids = rng.permutation(part.index.to_numpy())
        chosen8.extend(ids[:8])
        chosen32.extend(ids[:32])
    return pool.loc[chosen8].copy(), pool.loc[chosen32].copy()

for seed in [0, 11, 22, 33]:
    few8, few32 = make_nested_samples(seed)
    assert set(few8.row_id).issubset(set(few32.row_id))
    few8.to_csv(root / f'data/train_k8_seed{seed}.csv', index=False)
    few32.to_csv(root / f'data/train_k32_seed{seed}.csv', index=False)
```

数据行顺序、按类别排序方式、随机数生成器都是协议的一部分；只写 seed 而不共享实际名单，并不足以保证不同人的代码抽出同一批数据。三位组员应直接共享这些 CSV 和行号文件。

### 第六步：下载模型，先确认“句子能变成向量”

```python
from sentence_transformers import SentenceTransformer

model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = SentenceTransformer(model_id, device=device)
encoder.max_seq_length = 128
examples = ['My card has not arrived.', 'My card payment was declined.']
emb = encoder.encode(examples, normalize_embeddings=True)
print(emb.shape)  # 应为 (2, 384)
```

第一次运行自动下载权重；之后通常使用缓存。`(2,384)` 说明两句各产生一个 384 维向量，不说明它已经会分 BANKING77 的 77 类。不要把模型卡上的通用示例分数当成项目结果。

### 第七步：先完成 Frozen 基线的验证预测

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

few = pd.read_csv(root / 'data/train_k8_seed0.csv')
x_train = encoder.encode(few['text'].tolist(), batch_size=32,
                         normalize_embeddings=True, show_progress_bar=True)
x_dev = encoder.encode(dev['text'].tolist(), batch_size=32,
                       normalize_embeddings=True, show_progress_bar=True)
classifier = LogisticRegression(solver='lbfgs', C=1.0,
                                fit_intercept=True, max_iter=2000)
classifier.fit(x_train, few['label'])
pred_dev = classifier.predict(x_dev)
print('validation accuracy:', accuracy_score(dev['label'], pred_dev))
print('validation macro-F1:', f1_score(dev['label'], pred_dev,
                                      average='macro', zero_division=0))
```

这一步就形成了完整的“文本 → 表示 → 分类 → 验证”流程。没有预设应该得到多少分；如果报错或表现异常，检查标签对应、抽样、向量维度、NaN、分类器是否收敛。现在先不计算官方测试集成绩，因为后面还会做训练与选择。

`encoder.encode()` 适合提取固定向量，一般不是带梯度的训练接口。后面自写 CE/SetFit 微调时，需要通过模型的 forward 得到可求梯度的句子表示，然后 `loss.backward()` 和优化器更新；不能先用 encode 把所有句子固定成 numpy 向量，再声称训练到了编码器。

## 十四、第二天以后怎样搭建正式代码

以下是建议建立的模块，并不是压缩包里已经完成的训练脚本。压缩包提供 proposal 源文件和指南；正式模型训练代码需要依照这些步骤实现。

| 建议文件 | 输入 | 核心工作 | 输出/检查 |
|---|---|---|---|
| data.py | 官方 CSV、seed | 清洗、划分、抽样、tokenize | 保存名单；检查没有重叠、每类数量正确 |
| models.py | 初始 checkpoint | 编码器、均值池化、归一化、77类头 | 两句输出2×384；分类输出2×77 |
| pairs.py | 已选训练子集 | 四正四负采样；可选困难负样本 | 配对表及0/1；检查无自配对、数量及类别关系 |
| train_ce.py | 初始模型、训练/验证、配置 | 交叉熵训练及验证选模型 | checkpoint、日志、最佳更新步数 |
| train_setfit.py | 初始模型、配对表、验证 | 配对MSE训练、验证时拟合LR | checkpoint、训练记录及第二阶段分类器 |
| evaluate.py | 锁定模型、官方测试 | 指标、计时、保存每条预测 | CSV/JSON、混淆矩阵和结果表 |
| analyze.py | 各方法预测、预定类别对 | 混淆比较和人工抽样 | 图表及待人工解释的例句 |
| config.json | 所有固定设置 | 保存模型ID、版本、seed、超参数 | 别人能按配置复现 |

### 1. 先小批检查训练，再开完整运行

取少量训练句子，运行几个更新，确认输出形状、损失有限、梯度存在、编码器参数实际变化。CE 检查正确标签进入交叉熵；SetFit 检查同类关系标 1、异类标 0，两个分支共享同一个编码器。

验证时切换 `eval()`，关闭梯度；继续训练时切回 `train()`。两条分支“共享参数”不等于需要两份独立模型，各自有不同参数反而违背方案。

正式运行的配置中保存初始模型的 Hub revision/commit、库版本、随机种子及数据名单。尽可能固定随机性，但不同硬件/库上的 GPU 计算不保证逐比特完全一样。

### 2. 再跑两次 pilot，核算成本

同一张 GPU 上做一次 8-shot CE 与一次 8-shot SetFit 完整计时，计入 seed 0 的调参。记下每次验证的额外时间。不能把 Frozen 的编码时间乘一下就当作 SetFit 训练时间；后者有反向传播和重复验证拟合。

用剩余更新数和运行次数外推，判断 32-shot 是否保留为核心。初步估算若明显超预算，先缩到 8-shot，而不是临时减少某个方法的训练量后仍声称使用相同设置。

### 3. 完成调参，之后锁住规则

跑八次小网格，记录全部验证分数，不只保存最优那次。按预定规则选择学习率。用 frozen seed 0 的验证预测预选五对类别，保存为 JSON。之后才进入最终三 seed，避免根据最终测试结果反复更改假设。

### 4. 运行三方法 × 两数据量 × 三seed

每次从初始 checkpoint 单独加载。为每次运行保存：方法、k、seed、学习率、最佳更新步数、数据名单、耗时、显存、验证指标、模型文件。Frozen 分支保存分类器和初始编码器标识；SetFit 保存微调编码器和训练后的分类器；CE 保存编码器及线性头。

最终只对锁定模型做一次官方测试预测并保存。之后的全局指标、类别混淆和错误抽样可以从同一份预测计算，不需要为每张图重新训练或调整模型。

### 5. 保存成果，避免 Colab 断开后丢失

Colab 的 `/content` 是临时空间。把代码 notebook、CSV 名单、日志、配置、选中的 checkpoint 和预测文件保存到团队共享位置，例如自己的 Google Drive，再由组员共享。可以在 Colab 文件栏下载，或按界面授权挂载 Drive 后复制项目目录。API 密钥和云账户凭证不要写进共享 notebook。

第一次完整跑通后保存依赖版本，例如 `pip freeze` 的结果；正式结果使用同一环境。模型权重较大，可以按最终选定版本保存，过程中的临时检查点在确认不需要后清理，控制磁盘成本。

### 6. 根据实际结果写报告

先做主结果表，再做两种数据量的差值图、五对类别混淆表和代表性例句表。结论回答“在什么训练规模、哪些类别、付出多少成本后，哪种方法表现怎样”。若没提升，报告观察到的结果和限制，不伪造动机或事后改成另一个成功故事。

## 十五、你们具体复现、补充和下载哪些东西

| 性质 | 本项目内容 |
|---|---|
| 下载复用 | MiniLM 权重和 tokenizer；BANKING77；PyTorch、Sentence Transformers、sklearn等基础库；官方 SetFit 资料作为参照 |
| 复现核心 | SetFit 的句子对表示训练 + 冻结表示后的分类器两阶段思想 |
| 自己实现 | 数据协议、采样、句子对生成、损失及神经网络训练循环、验证选择、指标和结果保存 |
| 必做补充 | 同编码器/同样本冻结对照；普通微调比较；8/32嵌套抽样；三次重复；混淆及错误分析；实测成本 |
| 可选补充 | 四负搭档中两个替换为固定初始表示下的困难异类邻居，保持配对数和更新上限 |
| 不在范围 | 从零预训练、开发完整银行客服、在线部署、复现2025论文的LLM生成框架、声称新的SOTA |

对你目前的基础，主要难点是理解数据与梯度如何流动，以及把比较条件保持一致，不是显卡能否容纳这个小模型。最有效的起步是先完成本指南第一天的七步，亲眼看到一个句子经过编码器变成向量，再由分类器变成类别；之后再加入两种微调方法。

## 提交前需要填入的真实信息

英文稿的三位姓名/UNI 仍为占位，GCP 剩余余额也需要按真实账户填写。时间表里的 Nov. 23、Dec. 5、Dec. 12、Dec. 17 是团队工作目标，不是对课程官方截止日的断言；应与课程发布的实际 milestones 对齐。预算是计划值，完成 pilot 后应更新。其余实验设置已经具体写定，不需要为了增加篇幅补充没有使用的技术名词。
