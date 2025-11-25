# 数据规范与学术问题定义

## 一、数据集概述

本数据集源自上海市电动汽车 (EV) GPS 轨迹数据（2020年10月-12月），经过预处理后形成多层次的城市出行数据体系，支持**出行链生成**与**轨迹合成**研究。

### 1.1 数据规模统计

| 数据层级 | 数据文件 | 记录数量 | 个体/用户数 |
|:--------|:--------|--------:|----------:|
| 原始轨迹层 | `move.csv` | 369,512 条出行记录 | 1,994 辆车 |
| 原始轨迹层 | `stay.csv` | 367,518 条驻留记录 | 1,994 辆车 |
| 生活模式层 | `lifepattern_activity.csv` | 4,865,456 条活动记录 | 93,362 用户 |
| 生活模式层 | `lifepattern_move.csv` | — | 93,362 用户 |
| 宏观分布层 | `HWO_distribute.csv` | 30,422 条栅格记录 | — |
| 宏观交互层 | `HW_interact.csv` | 1,034,099 条OD对 | — |
| 宏观交互层 | `HO_interact.csv` | 3,988,341 条OD对 | — |
| 宏观交互层 | `WO_interact.csv` | 3,975,927 条OD对 | — |
| 时序OD层 | `OD_Hour_{0-23}.csv` | 1,242,860 条（合计） | — |

### 1.2 时空覆盖范围

| 维度 | 范围 |
|:----|:----|
| **时间跨度** | 2020-10-01 至 2020-12-31（92天） |
| **空间范围** | 经度 119.67°E ~ 121.97°E，纬度 30.27°N ~ 32.10°N |
| **栅格索引** | `loncol` ∈ [-97, 87]，`latcol` ∈ [-53, 167] |
| **栅格分辨率** | 约 1.25 km × 0.83 km（经纬度各约 0.0125°）|

---

## 二、数据结构详细规范

### 2.1 原始轨迹数据（Raw Trajectory Data）

#### 2.1.1 出行记录表 `move.csv`

记录车辆每次出行的起终点信息，形式化定义为出行事件 $m = (v, o, d, t_s, t_e, \Delta t, \delta)$。

| 字段名 | 数据类型 | 含义 | 形式化符号 |
|:------|:--------|:----|:---------|
| `vin` | `string` | 车辆唯一标识符（SHA256哈希） | $v \in \mathcal{V}$ |
| `slon`, `slat` | `float64` | 起点经纬度坐标 | $o = (lon_s, lat_s) \in \mathbb{R}^2$ |
| `elon`, `elat` | `float64` | 终点经纬度坐标 | $d = (lon_e, lat_e) \in \mathbb{R}^2$ |
| `SLONCOL`, `SLATCOL` | `int64` | 起点栅格索引 | $g_o = (c_s^{lon}, c_s^{lat}) \in \mathbb{Z}^2$ |
| `ELONCOL`, `ELATCOL` | `int64` | 终点栅格索引 | $g_d = (c_e^{lon}, c_e^{lat}) \in \mathbb{Z}^2$ |
| `stime` | `datetime` | 出行开始时间 | $t_s$ |
| `etime` | `datetime` | 出行结束时间 | $t_e$ |
| `duration` | `float64` | 出行时长（秒） | $\Delta t = t_e - t_s$ |
| `distance` | `float64` | 出行距离（千米） | $\delta$ |
| `moveid` | `int64` | 出行序号（车辆内） | — |
| `date` | `date` | 出行日期 | — |

**统计特征**：
- 平均出行距离：$\bar{\delta} \approx 9.65$ km
- 平均出行时长：$\bar{\Delta t} \approx 50.6$ min

#### 2.1.2 驻留记录表 `stay.csv`

记录车辆每次停留的时空信息，形式化定义为驻留事件 $s = (v, z, t_s, t_e, \Delta t)$。

| 字段名 | 数据类型 | 含义 | 形式化符号 |
|:------|:--------|:----|:---------|
| `vin` | `string` | 车辆唯一标识符 | $v \in \mathcal{V}$ |
| `lon`, `lat` | `float64` | 驻留点经纬度 | $z = (lon, lat)$ |
| `LONCOL`, `LATCOL` | `int64` | 驻留点栅格索引 | $g_z = (c^{lon}, c^{lat})$ |
| `stime` | `datetime` | 驻留开始时间 | $t_s$ |
| `etime` | `datetime` | 驻留结束时间 | $t_e$ |
| `duration` | `float64` | 驻留时长（秒） | $\Delta t$ |
| `stayid` | `int64` | 驻留序号（车辆内） | — |

**统计特征**：
- 平均驻留时长：$\bar{\Delta t} \approx 9.71$ 小时

---

### 2.2 生活模式数据（Life Pattern Data）

生活模式数据从原始轨迹中提取，描述用户的时空行为规律。数据源标识 `sh_2311` 表示上海市2023年11月数据。

#### 2.2.1 活动驻留表 `lifepattern_activity.csv`

描述用户"在何时段、驻留于何类地点"的统计规律，用于构建**一阶时空分布**。

| 字段名 | 数据类型 | 含义 | 形式化定义 |
|:------|:--------|:----|:---------|
| `reindex` | `int64` | 用户唯一标识 | $i \in \mathcal{U}$，$|\mathcal{U}| = 93,362$ |
| `shour` | `int64` | 驻留开始小时 (0-23) | $h_s \in \{0, 1, ..., 23\}$ |
| `ehour` | `int64` | 驻留结束小时 (0-23) | $h_e \in \{0, 1, ..., 23\}$ |
| `type` | `string` | 语义地点标识 | $\ell = \text{Type}\_\text{Index}$ |
| `count` | `int64` | 观测频次 | $n$ |
| `rank` | `int64` | 用户内排序 | — |

**语义地点类型 `type` 编码规则**：
- 格式：`{Category}_{Index}`，如 `H_0`, `W_1`, `O_5`
- 类别集合：$\mathcal{L} = \{H, W, O\}$
  - $H$ (Home)：居住地
  - $W$ (Work)：工作地
  - $O$ (Other)：其他常去地点
- 索引范围：$H \in \{0, 1, 2\}$，$W \in \{0, 1, 2, 3, 4\}$，$O \in \{0, 1, ..., 9\}$

**类别分布统计**：
| 类别 | 记录数 | 占比 |
|:----|------:|----:|
| Home (H) | 1,521,455 | 31.3% |
| Work (W) | 875,907 | 18.0% |
| Other (O) | 2,468,094 | 50.7% |

**跨日处理逻辑**：当 $h_s > h_e$ 时（如 22:00 至次日 6:00），需拆分为 $[h_s, 23]$ 和 $[0, h_e - 1]$ 两段。

#### 2.2.2 状态转移表 `lifepattern_move.csv`

描述用户"在何时、从何处、去往何处"的转移规律，用于构建**二阶转移概率**。

| 字段名 | 数据类型 | 含义 | 形式化定义 |
|:------|:--------|:----|:---------|
| `reindex` | `int64` | 用户唯一标识 | $i \in \mathcal{U}$ |
| `stype` | `string` | 出发地点类型 | $\ell_s \in \mathcal{L} \times \mathbb{N}$ |
| `etype` | `string` | 到达地点类型 | $\ell_e \in \mathcal{L} \times \mathbb{N}$ |
| `shour` | `int64` | 出发小时 | $h \in \{0, ..., 23\}$ |
| `count` | `int64` | 转移频次 | $n$ |
| `rank` | `int64` | 用户内排序 | — |

**概率解释**：对于用户 $i$ 在时刻 $h$，从地点 $\ell_s$ 出发，可计算条件转移概率：

$$P(\ell_e \mid \ell_s, h; i) = \frac{n_{i}(\ell_s \to \ell_e, h)}{\sum_{\ell'} n_{i}(\ell_s \to \ell', h)}$$

---

### 2.3 宏观空间分布数据（Anchor Points Data）

#### 2.3.1 HWO 空间分布 `HWO_distribute.csv`

城市级 H/W/O 地点的空间密度分布，形式化为概率测度 $\mu_c(g)$，$c \in \{H, W, O\}$。

| 字段名 | 数据类型 | 含义 |
|:------|:--------|:----|
| `loncol`, `latcol` | `int64` | 栅格索引 $g = (c^{lon}, c^{lat})$ |
| `lon`, `lat` | `float64` | 栅格中心经纬度 |
| `type` | `string` | 地点类型 $c \in \{H, W, O\}$ |
| `count` | `int64` | 该栅格该类型的聚合计数 |

**各类型栅格数量**：
| 类型 | 栅格数 |
|:----|------:|
| H | 11,501 |
| W | 5,447 |
| O | 13,474 |

#### 2.3.2 空间交互矩阵 `{HW,HO,WO}_interact.csv`

描述不同类型地点之间的空间共现关系，形式化为联合分布 $P(g_1, g_2)$，其中 $g_1$ 和 $g_2$ 分别属于不同类型。

| 字段名 | 数据类型 | 含义 |
|:------|:--------|:----|
| `{h,w,o}loncol`, `{h,w,o}latcol` | `int64` | 第一类地点栅格索引 |
| `{w,o}loncol`, `{o}latcol` | `int64` | 第二类地点栅格索引 |
| `count` | `int64` | 该配对的观测人数 |

**数据规模**：
- HW 交互：1,034,099 对（约束 $H \leftrightarrow W$ 空间关系）
- HO 交互：3,988,341 对（约束 $H \leftrightarrow O$ 空间关系）
- WO 交互：3,975,927 对（约束 $W \leftrightarrow O$ 空间关系）

---

### 2.4 时序OD流量数据（OD Data）

#### `OD_Hour_{h}.csv`，$h \in \{0, 1, ..., 23\}$

描述城市每小时的栅格级 OD 流量，形式化为时变 OD 矩阵 $\mathbf{F}^{(h)} \in \mathbb{R}^{|G| \times |G|}$。

| 字段名 | 数据类型 | 含义 |
|:------|:--------|:----|
| `sloncol`, `slatcol` | `int64` | 起点 (Origin) 栅格索引 $g_o$ |
| `eloncol`, `elatcol` | `int64` | 终点 (Destination) 栅格索引 $g_d$ |
| `count` | `int64` | 流量 $f_{g_o \to g_d}^{(h)}$ |

**时段特征**（代表性时段记录数）：
| 时段 | 记录数 | 特征 |
|:----|------:|:----|
| Hour 7 | 85,432 | 早高峰前 |
| Hour 8 | 135,483 | 早高峰 |
| Hour 9 | 90,832 | 早高峰后 |
| Hour 18 | ~120,000 | 晚高峰 |

---

## 三、学术问题形式化定义

### 3.1 问题背景与动机

基于上述多层次数据体系，本研究可形式化为一个**条件约束下的分层轨迹生成问题**，其核心挑战在于：

1. **微观-宏观一致性**：个体生成的轨迹在聚合后需满足城市级的统计约束
2. **时空耦合性**：出行链的时间模式与空间分布存在复杂的交互依赖
3. **多尺度表示**：需要同时处理语义地点（H/W/O）和物理位置（栅格坐标）两种尺度

### 3.2 问题定义

#### 定义 1：用户出行链（Trip Chain）

对于用户 $i \in \mathcal{U}$，其日出行链定义为时序位置序列：

$$\mathcal{C}_i = \{(\ell_1, h_1), (\ell_2, h_2), \ldots, (\ell_K, h_K)\}$$

其中 $\ell_k \in \mathcal{L} \times \mathbb{N}$ 为语义地点标识（如 $H_0, W_1$），$h_k \in \{0, ..., 23\}$ 为到达/离开时刻。

#### 定义 2：语义地点坐标化（Semantic Location Embedding）

定义映射函数 $\phi_i: \mathcal{L} \times \mathbb{N} \to \mathbb{R}^2$，将用户 $i$ 的语义地点映射到城市栅格坐标：

$$\phi_i(\ell) = (x_\ell^{(i)}, y_\ell^{(i)}) \in [0, W) \times [0, H)$$

其中 $W, H$ 为栅格空间的宽度和高度。

#### 定义 3：小时级位置调度（Hourly Schedule）

定义用户 $i$ 的 24 小时位置调度向量：

$$\mathbf{s}_i = [s_i^{(0)}, s_i^{(1)}, \ldots, s_i^{(23)}] \in (\mathcal{L} \times \mathbb{N})^{24}$$

其中 $s_i^{(h)}$ 表示用户 $i$ 在第 $h$ 小时所处的语义地点。

### 3.3 优化目标

#### 目标 1：一阶空间分布一致性

对于每种地点类型 $c \in \{H, W, O\}$，生成的语义坐标在聚合后应匹配真实空间分布：

$$\min_{\{\phi_i\}} \mathcal{D}\left( \frac{1}{|\mathcal{U}|} \sum_{i} \sum_{\ell \in c} \delta(\phi_i(\ell)), \mu_c \right)$$

其中 $\delta(\cdot)$ 为 Dirac 测度，$\mathcal{D}(\cdot, \cdot)$ 为分布距离度量（如 KL 散度、Wasserstein 距离）。

#### 目标 2：二阶时空交互一致性

生成坐标需满足 H-W、H-O、W-O 之间的空间共现约束：

$$\min_{\{\phi_i\}} \sum_{(c_1, c_2) \in \{HW, HO, WO\}} \mathcal{D}\left( P_{\text{gen}}(\phi_i(\ell_{c_1}), \phi_i(\ell_{c_2})), P_{\text{real}}(g_{c_1}, g_{c_2}) \right)$$

#### 目标 3：动态 OD 流量一致性

用户按照生活模式调度移动时，生成的小时级 OD 流量应匹配真值：

$$\min_{\{\phi_i, \mathbf{s}_i\}} \sum_{h=0}^{23} \mathcal{D}\left( \mathbf{F}_{\text{gen}}^{(h)}, \mathbf{F}_{\text{real}}^{(h)} \right)$$

其中生成流量矩阵定义为：

$$F_{\text{gen}, g_o \to g_d}^{(h)} = \sum_{i: \phi_i(s_i^{(h)}) = g_o, \phi_i(s_i^{(h+1)}) = g_d} 1$$

### 3.4 总体优化框架

综合上述目标，本问题可表述为多目标约束优化：

$$\min_{\Theta} \mathcal{L} = \lambda_1 \mathcal{L}_{\text{spatial}} + \lambda_2 \mathcal{L}_{\text{interact}} + \lambda_3 \mathcal{L}_{\text{OD}}$$

其中 $\Theta = \{\phi_i, \mathbf{s}_i\}_{i \in \mathcal{U}}$ 为待优化参数，$\lambda_1, \lambda_2, \lambda_3$ 为平衡系数。

---

## 四、与两级生成框架的对应关系

### 4.1 当前数据所支持的研究层级

| 层级 | 您的定义 | 本数据集对应 | 支持程度 |
|:----|:--------|:-----------|:--------|
| **问题 1** | 出行链生成 | Life Pattern 数据 → 语义地点序列 | ✅ 完全支持 |
| **问题 2** | 路网轨迹生成 | 原始轨迹数据 → 栅格级 OD | ⚠️ 部分支持（缺乏路网拓扑） |

### 4.2 数据与模型输入输出的对应

**问题 1 的数据映射**：

| 您的定义 | 本数据集对应 |
|:--------|:-----------|
| 城市条件 $X_{\text{city}}$ | `HWO_distribute.csv` + `*_interact.csv` + `OD_Hour_*.csv` |
| 个体属性 $a_i$ | `reindex` + Life Pattern 统计特征 |
| 生成目标 $\mathcal{C}_i$ | 24 小时语义地点序列 + 栅格坐标 |

**问题 2 的数据限制**：

当前数据集提供**栅格级 OD**（`move.csv` 的起终点），但缺乏：
- 路网拓扑结构 $G = (V, E)$
- 逐点轨迹序列（仅有起终点，无中间路径）
- 实时交通状态

### 4.3 学术定位建议

基于现有数据，建议将研究定位为：

> **基于深度生成模型的城市级出行链合成框架**：
> 在微观生活模式与宏观流动约束下，为合成个体生成高保真的语义出行链，使其聚合统计特性与真实城市观测一致。

具体贡献点可包括：
1. **分层生成架构**：先生成语义调度序列 $\mathbf{s}_i$，再求解最优坐标映射 $\phi_i$
2. **多尺度一致性约束**：同时优化一阶分布、二阶交互和动态 OD
3. **可微分验证框架**：将验证指标（PCC、分布距离）嵌入训练目标

---

## 五、验证指标定义

### 5.1 空间分布相似度

$$\text{SpatialSim}(c) = 1 - \text{JSD}(\hat{\mu}_c \| \mu_c)$$

其中 JSD 为 Jensen-Shannon 散度。

### 5.2 OD 流量相关性

$$\text{FlowCorr}(h) = \text{PCC}\left(\log(\mathbf{F}_{\text{gen}}^{(h)} + \epsilon), \log(\mathbf{F}_{\text{real}}^{(h)} + \epsilon)\right)$$

评价标准：
- PCC > 0.8：优秀
- PCC ∈ [0.6, 0.8]：良好
- PCC < 0.5：需改进

### 5.3 潮汐效应捕捉

定性评估早/晚高峰（8:00, 18:00）的 O/D 热力图是否呈现正确的职住潮汐模式。

---

## 六、附录：栅格坐标转换

### 经纬度 ↔ 栅格索引转换公式

设栅格分辨率为 $\Delta_{lon} = 0.0125°$，$\Delta_{lat} \approx 0.00833°$，参考原点为 $(lon_0, lat_0)$：

$$\text{loncol} = \left\lfloor \frac{lon - lon_0}{\Delta_{lon}} \right\rfloor$$

$$\text{latcol} = \left\lfloor \frac{lat - lat_0}{\Delta_{lat}} \right\rfloor$$

### 数据来源说明

- 原始轨迹数据：上海市 2020 年 Q4 EV GPS 数据
- 生活模式数据：上海市 2023 年 11 月手机信令派生数据（`sh_2311` 前缀）
