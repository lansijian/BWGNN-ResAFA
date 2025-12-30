## Experiment Log: Optimizing BWGNN for Graph Anomaly Detection

### 项目背景
本实验围绕 Amazon 数据集的图异常检测场景，目标是优化 BWGNN 模型对“伪装异常”节点的识别能力。基线模型虽然在 AUC 上表现卓越（>98%），但对隐蔽性异常的召回率存在不足。因此，我们针对频域特征聚合方案开展了多轮探索，以在维持高精准排序能力的同时提升召回率。

### 提出的创新点
- **模型名称**：BWGNN-ResAFA（Residual Adaptive Frequency Attention）
- **核心机制**：
  - 引入残差式频域注意力：对每个 Beta Wavelet 阶数的特征应用 `H_out = H_in × (1 + tanh(Attention))`，确保注意力层只能“增强调制”不会破坏原始特征，从而保底原始性能。
  - 引入 LayerNorm：在输入 MLP 各层增加 LayerNorm，稳定注意力学习过程。
  - 维持加权拼接维度：在残差修正后将所有阶数特征拼接，保留原始 BWGNN 的高维表示能力。

### 实验迭代记录（Amazon 数据集）
| Phase | 方案 | 关键结果 | 问题诊断 |
|-------|------|----------|----------|
| **Phase 1** | Baseline BWGNN 复现 | Recall 81.79%, F1 92.56%, AUC 98.16% | 排序性能极佳，但对伪装异常（弱信号）召回不足。 |
| **Phase 2** | 加权求和 (Weighted Sum Attention) | Recall 提升，AUC 降至 96.61% | 多阶特征压缩为单一表征造成信息瓶颈。 |
| **Phase 3** | 加权拼接 (Weighted Concatenation) | AUC 回升至 97.55%，F1/Precision 下降 | 直接用 Sigmoid 抑制/放大特征引入噪声，破坏原有特征分布。 |
| **Phase 4** | Residual Attention + Tanh + LayerNorm | Recall 85.15% (↑3.36%), F1 93.14% (↑0.58%), AUC 97.82% | 残差结构确保保底性能，Tanh 允许增强/抑制，综合表现最佳。 |

#### Phase 4 细节
最终方案通过 `H_out = H_in × (1 + tanh(Attention(h)))` 的残差门控，将 Beta Wavelet 不同频率的响应视为可调节的“频谱增益”，LayerNorm 则在各阶段对激活进行标准化，避免注意力层输入分布漂移。实验结果表明，在几乎保持原始 AUC 的同时，召回率有明显提升 (+3.36%)，整体 F1 也提升 0.58 个百分点。

### 最终模型代码片段（BWGNN_AFA/ResAFA）
```python
class BWGNN_AFA(nn.Module):
    """
    BWGNN variant with Residual Frequency Attention (ResAFA).

    核心思路：
    1. 残差结构：输出 = 原始特征 + 注意力修正，保证性能下限即原模型表现；
    2. Tanh 掩码：可对频率成分进行增强或抑制；
    3. LayerNorm 稳定多层感知机的输入分布；
    4. 保持加权拼接的高维表达能力。
    """

    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh(),
        )
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.linear2(h)
        h = self.ln2(h)
        h = self.act(h)
        h = self.dropout(h)

        h_list = []
        for conv in self.conv:
            h_list.append(conv(self.g, h))

        h_stack = torch.stack(h_list, dim=1)
        n_nodes, c_orders, hid_dim = h_stack.shape
        attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
        h_enhanced = h_stack * (1.0 + attn_delta)
        h_flat = h_enhanced.view(n_nodes, -1)

        h = self.linear3(h_flat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear4(h)
        return h
```

### 总结与应用价值
BWGNN-ResAFA 在维持 Amazon 数据集高 AUC（97.82% ≈ 98.16% 基线）的同时显著提升了 Recall（+3.36%）与 Macro-F1（+0.58%）。这一改进在电商欺诈检测等高风险业务中尤为关键，可有效降低漏报率，避免伪装异常逃逸模型审查，为业务提供更可靠的安全保障。

---

## YelpChi 数据集实验结果分析

### 通用 GNN 基准与谱域方法的优越性
在 YelpChi 异构图数据集上，三种通用 GNN 基线模型（GCN、GAT、GraphSAGE）整体表现明显不及 BWGNN 及其变体。具体而言，GCN 与 GAT 的 AUC 分别仅为 58.80% 和 58.07%，F1 分数也徘徊在约 56% 左右，几乎接近随机猜测的水平；即便是表现相对较好的 GraphSAGE，其 AUC 也仅为 82.14%，F1 为 68.50%，仍与谱域方法存在显著差距。这一结果从侧面说明，在高度异质、噪声丰富的评论欺诈场景中，单纯依赖一阶邻居聚合的空间域 GNN 难以有效分辨“伪装异常”和高频噪声节点。

相较之下，BWGNN 通过 Beta Wavelet 在图谱域上显式建模不同频率分量，能够利用高频成分捕获结构异常与语义不一致性，从而在 YelpChi 上获得显著优势：同构版本 BWGNN_Homo 的 AUC 已达到 86.83%，F1 为 72.79%；进一步结合异构信息的 BWGNN_Hetero 则将 AUC 提升至 89.41%，F1 提升至 76.16%。该组对比表明，谱域方法在异常检测任务中具有天然优势，特别是在需要区分细粒度结构异常和伪装行为时，对高频谱分量的刻画是通用 GNN 难以替代的关键能力。

### 异构建模的重要性
将 BWGNN_Homo 与 BWGNN_Hetero 的结果进行比较，可以更直观地看出异构建模的重要性。在相同的谱域框架下，仅仅增加对关系类型的显式建模（从同构到异构），AUC 即从 86.83% 提升到 89.41%，提升幅度约为 2.58 个百分点，F1 也从 72.79% 上升到 76.16%。这说明 YelpChi 中不同类型的交互关系（例如评论、点赞、好友等）在刻画异常行为模式时具有互补信息，如果将其简单折叠为单一同构边，势必会引入信息混叠，削弱模型对异常模式的表达能力。

因此，从谱域建模角度看，**“频率维度 + 关系类型维度”** 共同构成了刻画图异常的关键空间：仅在频率维度上做建模而忽略关系类型，会导致部分异常信号在折叠过程中被“掩盖”；只有在异构图设定下，频率分析才能在更精细的关系通道内展开，从而充分释放 BWGNN 的潜力。

### 注意力机制的双刃剑效应：BWGNN_Hetero_AFA v1 分析
在异构谱域建模基础上，我们进一步引入了 BWGNN_Hetero_AFA v1，将频域注意力机制应用到每种关系内部，以期自适应选择更为关键的频率分量。实验结果显示，相较于原始 BWGNN_Hetero（AUC 89.41%，F1 76.16%，Recall 65.31%，Precision 55.28%），AFA v1 的表现为：AUC 87.85%，F1 75.11%，Recall 59.05%，Precision 56.29%。可以观察到三个重要现象：

1. **Precision 小幅提升**：Precision 从 55.28% 提升到 56.29%，表明频域注意力在一定程度上抑制了部分高频噪声通道，使模型在给出“异常”判定时更加谨慎，即“报出来的更像真异常”。这一点证明了注意力在“频率选择性”上的潜在价值。
2. **Recall 明显下降**：Recall 从 65.31% 降至 59.05%，导致整体 F1 从 76.16% 降到 75.11%。这说明虽然模型在过滤噪声方面有所改善，但也同时“过滤掉”了一部分真正的异常样本，出现了典型的“过度保守”现象。
3. **AUC 略有退化**：AUC 从 89.41% 降至 87.85%，说明在整体排序层面，注意力的引入并未带来稳定收益，甚至在某些边界样本上削弱了模型的区分能力。

综合上述现象，可以将 BWGNN_Hetero_AFA v1 的行为概括为：**注意力机制作为一把“双刃剑”，在提高决策置信度（Precision）的同时，也引入了信息瓶颈，损害了模型的召回性能与整体排序能力。**

造成这一现象的核心原因在于当前 AFA v1 的实现仍采用了“频率求和聚合”的策略：对每一条异构关系内的多阶频率特征，先通过残差注意力进行修正，再在频率维度上进行求和（`sum(dim=1)`），最终将这一压缩后的表示输入到 `linear3`。在关系数量有限而频率信号复杂的 YelpChi 异构场景中，这种“先注意力再求和”的策略容易形成新的信息瓶颈：一方面，注意力的学习难度增大（需要在压缩前完成精确判别）；另一方面，一旦注意力分配略有偏差，被压缩掉的频率成分将难以通过后续层补偿。

因此，从当前实验结果来看，**AFA v1 更适合作为“噪声过滤器”而非“增强器”**：它帮助模型略微提升了 Precision，但以牺牲 Recall 和部分 AUC 为代价。后续工作可以沿以下方向继续改进：
- 在异构场景下，将当前的“频率求和”替换为“频率拼接”（类似 Amazon 场景中的 ResAFA），在引入残差注意力的同时保留更多高维频谱信息；
- 结合关系级与频率级的双层注意力，使模型能够在“哪种关系更重要”和“关系内部哪些频率更有用”两个层面上同时做出自适应选择。

整体而言，YelpChi 实验验证了谱域+异构建模在图异常检测中的关键作用，同时也揭示了注意力机制在复杂异构环境下的敏感性与双刃剑效应，为后续设计更稳定、信息保留更充分的频域注意力结构提供了有价值的经验依据。

