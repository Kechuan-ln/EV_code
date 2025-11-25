# 问题定义：分布约束下的语义锚点空间嵌入

## 一、记号与数据抽象

### 1.1 空间与时间离散化

| 符号 | 定义 | 数据来源 |
|:----|:----|:--------|
| $G$ | 城市栅格集合，$g = (c^{\text{lon}}, c^{\text{lat}}) \in \mathbb{Z}^2$ | 所有CSV的 `loncol`, `latcol` |
| $\mathcal{H}$ | 小时集合 $\{0, 1, \ldots, 23\}$ | `shour`, `ehour` 字段 |

**栅格参数**（基于数据统计）：
- 经度索引范围：$c^{\text{lon}} \in [-97, 87]$，共185列
- 纬度索引范围：$c^{\text{lat}} \in [-53, 167]$，共221行
- 栅格分辨率：约 $1.25 \text{ km} \times 0.83 \text{ km}$

### 1.2 用户与语义地点

| 符号 | 定义 | 数据来源 |
|:----|:----|:--------|
| $\mathcal{U}$ | 用户集合，$\|\mathcal{U}\| = 93,362$ | `lifepattern_*.csv` 的 `reindex` |
| $\mathcal{C}$ | 语义类型集合 $\{H, W, O\}$ | `type` 字段的类别前缀 |
| $\mathcal{L}_i$ | 用户 $i$ 的语义地点集合 | 从 `type` 字段解析 |

**语义地点编码**：每个语义地点 $\ell \in \mathcal{L}_i$ 表示为 $\ell = (c, k)$，其中：
- $c \in \mathcal{C}$ 为类型（Home/Work/Other）
- $k \in \mathbb{N}$ 为用户内编号

**用户语义地点数量上界**（从数据统计）：
$$|\mathcal{L}_i^H| \leq 3, \quad |\mathcal{L}_i^W| \leq 5, \quad |\mathcal{L}_i^O| \leq 10$$
$$\Rightarrow |\mathcal{L}_i| \leq 18$$

### 1.3 真实宏观统计（观测数据）

#### 一阶空间分布
$$\mu_c^{\text{real}}: G \to \mathbb{R}_{\geq 0}, \quad c \in \mathcal{C}$$
- **数据来源**：`HWO_distribute.csv`
- **含义**：类型 $c$ 的地点在栅格 $g$ 上的聚合计数
- **记录数**：$H$: 11,501 | $W$: 5,447 | $O$: 13,474

#### 二阶空间交互分布
$$\pi_{c_1 c_2}^{\text{real}}: G \times G \to \mathbb{R}_{\geq 0}, \quad (c_1, c_2) \in \{HW, HO, WO\}$$
- **数据来源**：`HW_interact.csv`, `HO_interact.csv`, `WO_interact.csv`
- **含义**：同一用户的 $c_1$ 类地点在 $g_1$、$c_2$ 类地点在 $g_2$ 的共现计数
- **记录数**：$HW$: 1,034,099 | $HO$: 3,988,341 | $WO$: 3,975,927

#### 动态OD流量
$$F^{(h)}_{\text{real}}: G \times G \to \mathbb{R}_{\geq 0}, \quad h \in \mathcal{H}$$
- **数据来源**：`OD_Hour_{0-23}.csv`
- **含义**：小时 $h$ 内从栅格 $g_o$ 到 $g_d$ 的流量
- **总记录数**：1,242,860

---

## 二、个体生活模式（已知先验）

### 2.1 活动驻留分布

对于用户 $i \in \mathcal{U}$，其在小时 $h$ 处于语义地点 $\ell$ 的概率由 `lifepattern_activity.csv` 给出：

$$P_i^{\text{act}}(\ell, h) = \frac{n_i(\ell, h)}{\sum_{\ell' \in \mathcal{L}_i} n_i(\ell', h)}$$

其中 $n_i(\ell, h)$ 为用户 $i$ 在小时 $h$ 驻留于地点 $\ell$ 的观测频次。

**跨日处理**：当 `shour` > `ehour` 时（如 22:00 至次日 6:00），需将时段拆分为 $[h_s, 23] \cup [0, h_e]$。

### 2.2 状态转移分布

用户 $i$ 在小时 $h$ 从地点 $\ell_s$ 转移到 $\ell_e$ 的条件概率由 `lifepattern_move.csv` 给出：

$$P_i^{\text{trans}}(\ell_e \mid \ell_s, h) = \frac{n_i(\ell_s \to \ell_e, h)}{\sum_{\ell'} n_i(\ell_s \to \ell', h)}$$

### 2.3 日程序列的生成模型

基于上述分布，用户 $i$ 的 24 小时语义位置序列可建模为：

$$\mathbf{s}_i = (s_i^{(0)}, s_i^{(1)}, \ldots, s_i^{(23)}), \quad s_i^{(h)} \in \mathcal{L}_i$$

其联合分布为：
$$P_i(\mathbf{s}_i) = P_i^{\text{act}}(s_i^{(0)}, 0) \cdot \prod_{h=0}^{22} P_i^{\text{trans}}(s_i^{(h+1)} \mid s_i^{(h)}, h)$$

**关键观察**：生活模式数据已提供了 $P_i(\mathbf{s}_i)$ 的完整参数化，因此 $\mathbf{s}_i$ 可视为**可采样的已知分布**，而非待学习对象。

---

## 三、待求解对象

### 3.1 语义-空间映射函数

对于每个用户 $i \in \mathcal{U}$，需要确定其语义地点到城市栅格的映射：

$$\phi_i: \mathcal{L}_i \to G$$

即为用户 $i$ 的每个语义地点 $\ell \in \mathcal{L}_i$ 分配一个具体的空间坐标 $\phi_i(\ell) = g \in G$。

### 3.2 诱导的空间位置序列

给定日程 $\mathbf{s}_i$ 和映射 $\phi_i$，用户 $i$ 在小时 $h$ 的空间位置为：

$$x_i^{(h)} = \phi_i(s_i^{(h)}) \in G$$

### 3.3 诱导的聚合统计

由全体用户的 $\{\phi_i\}$ 和 $\{\mathbf{s}_i\}$ 可诱导出以下生成统计：

**一阶空间分布**：
$$\mu_{c,\text{gen}}(g) = \sum_{i \in \mathcal{U}} \sum_{\ell \in \mathcal{L}_i^c} \mathbb{I}[\phi_i(\ell) = g]$$

**二阶空间交互**：
$$\pi_{c_1 c_2, \text{gen}}(g_1, g_2) = \sum_{i \in \mathcal{U}} \sum_{\substack{\ell_1 \in \mathcal{L}_i^{c_1} \\ \ell_2 \in \mathcal{L}_i^{c_2}}} \mathbb{I}[\phi_i(\ell_1) = g_1, \phi_i(\ell_2) = g_2]$$

**动态OD流量**：
$$F_{\text{gen}}^{(h)}(g_o, g_d) = \sum_{i \in \mathcal{U}} \mathbb{I}[x_i^{(h)} = g_o, x_i^{(h+1)} = g_d]$$

其中 $h+1$ 按模 24 计算。

---

## 四、优化问题形式化

### 4.1 核心问题陈述

> **问题（分布约束下的语义锚点空间嵌入）**
>
> 给定：
> 1. 城市宏观统计约束 $\{\mu_c^{\text{real}}\}_{c \in \mathcal{C}}$，$\{\pi_{c_1c_2}^{\text{real}}\}_{(c_1,c_2)}$，$\{F_{\text{real}}^{(h)}\}_{h \in \mathcal{H}}$
> 2. 个体生活模式分布 $\{P_i(\mathbf{s}_i)\}_{i \in \mathcal{U}}$
>
> 求解映射族 $\{\phi_i: \mathcal{L}_i \to G\}_{i \in \mathcal{U}}$，使得当每个用户按其生活模式移动时，诱导的聚合统计与真实观测尽可能一致。

### 4.2 目标函数

$$\min_{\{\phi_i\}_{i \in \mathcal{U}}} \mathcal{L}(\{\phi_i\})$$

其中总损失分解为：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{spatial}} + \lambda_2 \mathcal{L}_{\text{interact}} + \lambda_3 \mathcal{L}_{\text{OD}}$$

#### 一阶空间分布损失
$$\mathcal{L}_{\text{spatial}} = \sum_{c \in \mathcal{C}} D\left(\bar{\mu}_{c,\text{gen}}, \bar{\mu}_c^{\text{real}}\right)$$

其中 $\bar{\mu}$ 表示归一化后的概率分布，$D(\cdot, \cdot)$ 为分布距离度量。

#### 二阶空间交互损失
$$\mathcal{L}_{\text{interact}} = \sum_{(c_1, c_2) \in \{HW, HO, WO\}} D\left(\bar{\pi}_{c_1c_2,\text{gen}}, \bar{\pi}_{c_1c_2}^{\text{real}}\right)$$

#### 动态OD流量损失
$$\mathcal{L}_{\text{OD}} = \sum_{h \in \mathcal{H}} D\left(\bar{F}_{\text{gen}}^{(h)}, \bar{F}_{\text{real}}^{(h)}\right)$$

### 4.3 距离度量选择

| 度量 | 定义 | 适用场景 |
|:----|:----|:--------|
| KL散度 | $D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$ | 概率分布匹配 |
| JSD散度 | $D_{JS} = \frac{1}{2}D_{KL}(p\|m) + \frac{1}{2}D_{KL}(q\|m)$, $m=\frac{p+q}{2}$ | 对称分布比较 |
| Wasserstein | $W_1(p,q) = \inf_{\gamma} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$ | 空间结构保持 |
| 负PCC | $-\text{corr}(\log(p+\epsilon), \log(q+\epsilon))$ | 流量强度相关性 |

---

## 五、问题的数学难点

### 5.1 高维离散组合优化

**决策变量规模**：
$$\text{dim}(\{\phi_i\}) = \sum_{i \in \mathcal{U}} |\mathcal{L}_i| \approx 93,362 \times 18 \approx 1.68 \times 10^6$$

每个变量取值于离散栅格空间 $|G| \approx 185 \times 221 \approx 4 \times 10^4$。

**解空间大小**：
$$(|G|)^{\sum_i |\mathcal{L}_i|} \approx (4 \times 10^4)^{1.68 \times 10^6}$$

这是一个天文数字级的组合优化问题。

### 5.2 多边缘约束的耦合性

三类约束并非独立：
- $\mathcal{L}_{\text{spatial}}$ 约束每类地点的**边缘分布**
- $\mathcal{L}_{\text{interact}}$ 约束跨类型地点的**联合分布**
- $\mathcal{L}_{\text{OD}}$ 约束**时变条件分布**（依赖于 $\mathbf{s}_i$ 的采样）

满足单一约束容易，同时满足三类约束则形成**多边缘最优传输**类问题，计算复杂度极高。

### 5.3 随机性传播

目标函数依赖于 $\mathbf{s}_i \sim P_i(\mathbf{s}_i)$ 的采样：

$$\mathcal{L}_{\text{OD}} = \mathbb{E}_{\{\mathbf{s}_i\} \sim \prod_i P_i} \left[ \sum_h D(F_{\text{gen}}^{(h)}, F_{\text{real}}^{(h)}) \right]$$

这导致：
1. 目标函数本身是随机的，需要蒙特卡洛估计
2. 对 $\phi_i$ 的梯度需要通过离散采样传播，存在高方差问题

### 5.4 非唯一性与可识别性

**定理（解的非唯一性）**：设 $\{\phi_i^*\}$ 是一个最优解，则对于任意保持聚合统计不变的变换 $T$，$\{T \circ \phi_i^*\}$ 也是最优解。

这意味着：
- 同一栅格内的用户地点可以任意交换
- 不同用户间只要保持总体分布不变，个体分配可以重排

因此，问题本质上是求解一个**等价类**而非唯一解。

---

## 六、为何通用生成模型不足

### 6.1 标准VAE/CVAE的局限

**问题**：VAE通过最大化ELBO优化：
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

这是**样本级重构损失**，而非**聚合统计约束**。

**具体困难**：
- 无法直接表达 $\mu_{\text{gen}} \approx \mu^{\text{real}}$ 这类全局约束
- 条件VAE的条件 $c$ 通常是低维标签，而非高维统计张量

### 6.2 标准Diffusion的局限

**问题**：Diffusion模型通过去噪得分匹配：
$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

这优化的是**逐样本的噪声预测**，生成样本"看起来像"训练样本。

**具体困难**：
- 生成样本的聚合统计不受直接控制
- 对离散输出（栅格坐标）需要额外的量化/离散化处理
- 难以注入OD矩阵这类二元关系约束

### 6.3 标准GAN的局限

**问题**：GAN通过对抗训练匹配分布：
$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**具体困难**：
- 判别器作用于单样本，无法直接判断聚合统计是否匹配
- 需要设计特殊的"统计判别器"来比较批次级分布
- 训练不稳定性在高维离散输出上更加严重

### 6.4 核心差异总结

| 方面 | 通用生成模型 | 本问题需求 |
|:----|:-----------|:----------|
| 优化目标 | 样本似然 $p(x)$ | 聚合统计 $\mu, \pi, F$ |
| 输出空间 | 连续向量 | 离散栅格组合 |
| 约束类型 | 隐式（通过数据） | 显式多边缘约束 |
| 条件信息 | 低维标签/特征 | 高维统计张量 |

---

## 七、问题的等价表述

### 7.1 作为约束优化问题

$$\begin{aligned}
\min_{\{\phi_i\}} \quad & 0 \\
\text{s.t.} \quad & \mu_{c,\text{gen}} = \mu_c^{\text{real}}, \quad \forall c \in \mathcal{C} \\
& \pi_{c_1c_2,\text{gen}} = \pi_{c_1c_2}^{\text{real}}, \quad \forall (c_1,c_2) \\
& F_{\text{gen}}^{(h)} = F_{\text{real}}^{(h)}, \quad \forall h \in \mathcal{H}
\end{aligned}$$

由于精确满足约束通常不可行，松弛为软约束优化（即第四节的形式）。

### 7.2 作为逆问题

给定聚合统计 $(\mu^{\text{real}}, \pi^{\text{real}}, F^{\text{real}})$，反演微观配置 $\{\phi_i\}$：

$$\text{Aggregation}(\{\phi_i\}, \{P_i\}) = (\mu, \pi, F)$$

这是一个**病态逆问题**（ill-posed inverse problem），因为：
1. 解不唯一（见5.4节）
2. 对观测噪声敏感
3. 需要正则化或先验约束

### 7.3 作为多边缘最优传输

将问题视为寻找一个联合分布，其多个边缘投影分别匹配 $\mu_H, \mu_W, \mu_O, \pi_{HW}, \ldots$：

$$\min_{\gamma \in \Pi(\mu_H, \mu_W, \mu_O, \ldots)} \int c(g_H, g_W, g_O, \ldots) \, d\gamma$$

这与**多边缘最优传输**（Multi-marginal Optimal Transport）的数学结构一致。

---

## 八、输入-输出规范

### 8.1 训练/优化阶段输入

| 输入 | 形状 | 来源 |
|:----|:----|:----|
| $\mu_c^{\text{real}}$ | $(3, H, W)$ 张量 | `HWO_distribute.csv` |
| $\pi_{c_1c_2}^{\text{real}}$ | $(3, H \times W, H \times W)$ 稀疏矩阵 | `*_interact.csv` |
| $F_{\text{real}}^{(h)}$ | $(24, H \times W, H \times W)$ 稀疏矩阵 | `OD_Hour_*.csv` |
| $\{P_i^{\text{act}}, P_i^{\text{trans}}\}$ | 每用户的概率表 | `lifepattern_*.csv` |

其中 $H = 221$, $W = 185$ 为栅格尺寸。

### 8.2 输出

| 输出 | 形状 | 含义 |
|:----|:----|:----|
| $\phi_i(\ell)$ | $(N, L_{\max}, 2)$ 张量 | 每用户每语义地点的栅格坐标 |

其中 $N = 93,362$ 为用户数，$L_{\max} = 18$ 为语义地点数上界。

### 8.3 验证指标

| 指标 | 公式 | 目标 |
|:----|:----|:----|
| 空间分布JSD | $\text{JSD}(\mu_{\text{gen}} \| \mu^{\text{real}})$ | $\to 0$ |
| 交互分布JSD | $\text{JSD}(\pi_{\text{gen}} \| \pi^{\text{real}})$ | $\to 0$ |
| OD流量PCC | $\text{corr}(\log F_{\text{gen}}, \log F^{\text{real}})$ | $\to 1$ |

**PCC评价标准**：
- $> 0.8$：优秀
- $[0.6, 0.8]$：良好
- $< 0.5$：需改进

---

## 九、与EV数据的关系

### 9.1 数据异源性

| 数据源 | 用户规模 | 时间范围 | 用途 |
|:------|-------:|:--------|:----|
| 手机信令 (lifepattern) | 93,362 | 2023年11月 | 生活模式先验 |
| EV GPS (move/stay) | 1,994 | 2020年Q4 | 子群统计校准 |

两套数据的用户ID体系不对应，只能在**统计层面**对齐。

### 9.2 EV子群约束（可选）

若需额外约束EV子群的出行特征，可增加损失项：

$$\mathcal{L}_{\text{EV}} = D\left(P_{\text{gen}}^{\text{EV}}(\delta, \Delta t), P_{\text{real}}^{\text{EV}}(\delta, \Delta t)\right)$$

其中 $P^{\text{EV}}(\delta, \Delta t)$ 为出行距离-时长的联合分布。

**经验统计**（来自 `move.csv`）：
- 平均出行距离：$\bar{\delta} = 9.65$ km
- 平均出行时长：$\bar{\Delta t} = 50.6$ min
- 平均驻留时长：$9.71$ h

---

## 十、一句话总结

> **本问题的本质**：在给定城市宏观流动统计和个体生活模式先验的条件下，为近10万用户的语义地点（H/W/O）求解空间坐标分配，使得微观分配的聚合效果在一阶分布、二阶交互、动态OD三个维度上与真实观测一致——这是一个高维离散组合优化与多边缘分布匹配的耦合问题。
