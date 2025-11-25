# SS-DMFO 2.0 系统架构设计

## 一、系统概览

### 1.1 核心架构原则

- **模块化设计**：核心算法、数据处理、评估系统相互独立
- **渐进式实现**：从一阶约束开始，逐步增加复杂度
- **基准对比优先**：每个组件都要与基线方法对比
- **性能可观测**：内置profiling和监控

### 1.2 技术栈

```yaml
core:
  language: Python 3.10+
  framework: PyTorch 2.0+
  compute: CUDA 11.8+

dependencies:
  numerical: numpy, scipy
  sparse: torch-sparse, scipy.sparse
  data: pandas, h5py
  visualization: matplotlib, plotly
  profiling: torch.profiler, line_profiler
  experiment: wandb, hydra
```

### 1.3 系统架构图

```
┌─────────────────────────────────────────────────────┐
│                    Main Pipeline                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │   Data   │──▶│  Model   │──▶│ Evaluate │       │
│  │  Loader  │   │  Engine  │   │  System  │       │
│  └──────────┘   └──────────┘   └──────────┘       │
│       │              │              │               │
│       ▼              ▼              ▼               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │  Cache   │   │Optimizer │   │ Metrics  │       │
│  │  System  │   │  Module  │   │  Store   │       │
│  └──────────┘   └──────────┘   └──────────┘       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 二、模块设计

### 2.1 数据层 (ssdmfo/data/)

#### 2.1.1 数据加载器

```python
# ssdmfo/data/loader.py
class ConstraintDataLoader:
    """加载和管理城市约束数据"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def load_spatial_constraints(self) -> SpatialConstraints:
        """加载一阶空间分布约束 μ_c^real"""
        # 从 HWO_distribute.csv 加载
        return SpatialConstraints(
            H=self._load_distribution('H'),
            W=self._load_distribution('W'),
            O=self._load_distribution('O')
        )

    def load_interaction_constraints(self) -> InteractionConstraints:
        """加载二阶交互约束 π_cc'^real"""
        # 从 HW/HO/WO_interact.csv 加载
        return InteractionConstraints(
            HW=self._load_sparse_matrix('HW'),
            HO=self._load_sparse_matrix('HO'),
            WO=self._load_sparse_matrix('WO')
        )

    def load_od_constraints(self) -> ODConstraints:
        """加载动态OD约束 F^(h)_real"""
        # 从 OD_Hour_*.csv 加载
        od_matrices = []
        for h in range(24):
            od_matrices.append(self._load_sparse_od(h))
        return ODConstraints(od_matrices)
```

#### 2.1.2 生活模式处理器

```python
# ssdmfo/data/life_pattern.py
class LifePatternProcessor:
    """处理用户生活模式数据"""

    def __init__(self, activity_file: str, move_file: str):
        self.activity_df = self._load_compressed(activity_file)
        self.move_df = self._load_compressed(move_file)
        self.users = None
        self.transition_probs = {}  # 缓存DP结果

    def build_user_patterns(self) -> Dict[int, UserPattern]:
        """构建每个用户的生活模式"""
        users = {}
        for user_id in self.get_unique_users():
            users[user_id] = UserPattern(
                locations=self._extract_locations(user_id),
                activity_probs=self._compute_activity_probs(user_id),
                transition_probs=self._compute_transition_probs(user_id)
            )
        return users

    def compute_dp_statistics(self, user_id: int) -> DPStatistics:
        """动态规划计算充分统计量 T_i(h,ℓ_s,ℓ_e)"""
        if user_id in self.transition_probs:
            return self.transition_probs[user_id]

        # 前向-后向算法计算边缘概率
        T = self._forward_backward(user_id)
        self.transition_probs[user_id] = T
        return T
```

#### 2.1.3 数据结构定义

```python
# ssdmfo/data/structures.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from scipy.sparse import csr_matrix

@dataclass
class SpatialConstraints:
    """一阶空间分布约束"""
    H: torch.Tensor  # shape: (H, W)
    W: torch.Tensor  # shape: (H, W)
    O: torch.Tensor  # shape: (H, W)

    def normalize(self):
        """归一化为概率分布"""
        self.H = self.H / self.H.sum()
        self.W = self.W / self.W.sum()
        self.O = self.O / self.O.sum()

@dataclass
class InteractionConstraints:
    """二阶交互约束（稀疏）"""
    HW: csr_matrix  # shape: (H*W, H*W)
    HO: csr_matrix  # shape: (H*W, H*W)
    WO: csr_matrix  # shape: (H*W, H*W)

    @property
    def total_nnz(self) -> int:
        """总非零元素数"""
        return self.HW.nnz + self.HO.nnz + self.WO.nnz

@dataclass
class UserPattern:
    """用户生活模式"""
    locations: List[SemanticLocation]  # 语义地点列表
    activity_probs: Dict[int, torch.Tensor]  # P(ℓ|h)
    transition_probs: torch.Tensor  # P(ℓ'|ℓ,h)

@dataclass
class SemanticLocation:
    """语义地点"""
    type: str  # 'H', 'W', 'O'
    index: int  # 0, 1, 2, ...

    def __str__(self):
        return f"{self.type}_{self.index}"
```

### 2.2 核心算法层 (ssdmfo/core/)

#### 2.2.1 对偶优化器

```python
# ssdmfo/core/dual_optimizer.py
class DualOptimizer:
    """SS-DMFO 2.0 对偶优化核心"""

    def __init__(self, config: DualConfig):
        self.config = config
        self.device = config.device

        # 对偶变量（势函数）
        self.potentials = DualPotentials(config)

        # 优化器
        self.optimizer = self._build_optimizer()

        # 监控器
        self.monitor = ConvergenceMonitor(config)

    def optimize(self,
                 constraints: Constraints,
                 user_patterns: Dict[int, UserPattern],
                 max_iter: int = 5000) -> OptimizationResult:
        """主优化循环"""

        for iteration in range(max_iter):
            # 1. 采样批次用户
            batch_users = self._sample_batch(user_patterns)

            # 2. 计算微观响应
            responses = self._compute_responses(batch_users)

            # 3. 聚合统计
            stats = self._aggregate_statistics(responses)

            # 4. 计算对偶梯度
            gradients = self._compute_gradients(stats, constraints)

            # 5. 更新对偶变量
            self.optimizer.step(gradients)

            # 6. 监控收敛
            metrics = self.monitor.update(stats, constraints)
            if self.monitor.has_converged():
                break

            # 7. 记录日志
            if iteration % config.log_freq == 0:
                self._log_progress(iteration, metrics)

        return self._finalize_results()
```

#### 2.2.2 平均场变分推断

```python
# ssdmfo/core/mean_field.py
class MeanFieldVI:
    """平均场变分推断求解器"""

    def __init__(self, config: MFVIConfig):
        self.config = config
        self.temperature = config.temperature
        self.damping = config.damping_factor

    def compute_response(self,
                        user: UserPattern,
                        potentials: DualPotentials,
                        dp_stats: DPStatistics) -> UserResponse:
        """计算单个用户的最优响应 Q_i"""

        n_locations = len(user.locations)
        grid_size = potentials.grid_h * potentials.grid_w

        # 初始化Q_i (log域)
        log_q = torch.zeros(n_locations, grid_size)

        # MFVI迭代
        for iter in range(self.config.max_iter):
            log_q_old = log_q.clone()

            # 逐个地点更新
            for ℓ_idx in range(n_locations):
                # 计算局部场
                field = self._compute_local_field(
                    ℓ_idx, user, potentials, log_q, dp_stats
                )

                # Boltzmann分布更新（log域）
                log_q[ℓ_idx] = -field / self.temperature
                log_q[ℓ_idx] = log_q[ℓ_idx] - torch.logsumexp(log_q[ℓ_idx], dim=0)

            # 阻尼更新
            log_q = self.damping * log_q + (1 - self.damping) * log_q_old

            # 检查收敛
            if self._check_convergence(log_q, log_q_old):
                break

        return UserResponse(
            user_id=user.id,
            log_probs=log_q,
            probs=torch.exp(log_q)
        )

    def _compute_local_field(self, ℓ_idx, user, potentials, log_q, dp_stats):
        """计算局部场（核心计算）"""

        ℓ = user.locations[ℓ_idx]

        # 一阶势能
        if ℓ.type == 'H':
            field = potentials.alpha_H.flatten()
        elif ℓ.type == 'W':
            field = potentials.alpha_W.flatten()
        else:
            field = potentials.alpha_O.flatten()

        # 二阶势能（稀疏矩阵乘法）
        for ℓ2_idx in range(len(user.locations)):
            if ℓ_idx == ℓ2_idx:
                continue

            interaction = self._get_sparse_interaction(
                ℓ, user.locations[ℓ2_idx], potentials
            )

            if interaction is not None:
                # 稀疏矩阵-向量乘法
                q_ℓ2 = torch.exp(log_q[ℓ2_idx])
                field += interaction @ q_ℓ2

        # 动态OD势能（利用DP结果）
        field += self._compute_od_field(ℓ_idx, user, potentials, log_q, dp_stats)

        return field
```

#### 2.2.3 稀疏张量加速

```python
# ssdmfo/core/sparse_ops.py
import torch.sparse as sparse

class SparseAccelerator:
    """GPU加速的稀疏运算"""

    def __init__(self, device='cuda'):
        self.device = device
        self._cache = {}

    def build_sparse_tensor(self, csr_matrix) -> torch.sparse.Tensor:
        """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
        coo = csr_matrix.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        shape = coo.shape

        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape,
            dtype=torch.float32,
            device=self.device
        )
        return sparse_tensor

    def sparse_dense_matmul(self,
                           sparse_mat: torch.sparse.Tensor,
                           dense_vec: torch.Tensor) -> torch.Tensor:
        """优化的稀疏-稠密矩阵乘法"""
        # 利用PyTorch的优化内核
        return torch.sparse.mm(sparse_mat, dense_vec.unsqueeze(1)).squeeze()

    def batch_sparse_lookup(self,
                           sparse_tensors: Dict[str, torch.sparse.Tensor],
                           indices: torch.Tensor) -> torch.Tensor:
        """批量稀疏查找"""
        # 实现高效的批量查找
        pass
```

### 2.3 基线方法层 (ssdmfo/baselines/)

#### 2.3.1 IPF基线

```python
# ssdmfo/baselines/ipf.py
class IterativeProportionalFitting:
    """迭代比例拟合基线方法"""

    def __init__(self, config: IPFConfig):
        self.config = config
        self.max_iter = config.max_iter
        self.tolerance = config.tolerance

    def fit(self, constraints: Constraints,
            user_patterns: Dict[int, UserPattern]) -> IPFResult:
        """
        IPF算法：交替调整矩阵以满足边缘约束
        这是处理多边缘约束的经典方法
        """

        # 初始化：随机分配或均匀分配
        allocations = self._initialize_allocations(user_patterns)

        for iteration in range(self.max_iter):
            # 1. 调整以满足空间分布约束
            allocations = self._adjust_spatial(allocations, constraints.spatial)

            # 2. 调整以满足交互约束（如果启用）
            if self.config.use_interactions:
                allocations = self._adjust_interactions(
                    allocations, constraints.interactions
                )

            # 3. 检查收敛
            if self._check_convergence(allocations, constraints):
                break

        return IPFResult(allocations, iteration)

    def _adjust_spatial(self, allocations, spatial_constraints):
        """调整分配以满足一阶空间约束"""
        # Sinkhorn-Knopp类型的行列归一化
        pass
```

#### 2.3.2 重力模型基线

```python
# ssdmfo/baselines/gravity.py
class GravityModel:
    """重力模型基线方法"""

    def __init__(self, config: GravityConfig):
        self.config = config
        self.beta = config.distance_decay  # 距离衰减参数

    def generate(self, constraints: Constraints,
                user_patterns: Dict[int, UserPattern]) -> GravityResult:
        """
        重力模型：P(i→j) ∝ M_i × M_j / d_ij^β
        其中M为质量（人口/活动强度），d为距离
        """

        # 1. 计算每个栅格的"质量"
        masses = self._compute_masses(constraints.spatial)

        # 2. 计算距离矩阵
        distances = self._compute_distances()

        # 3. 生成分配
        allocations = {}
        for user_id, pattern in user_patterns.items():
            user_alloc = self._allocate_user(pattern, masses, distances)
            allocations[user_id] = user_alloc

        return GravityResult(allocations)
```

### 2.4 评估系统 (ssdmfo/evaluation/)

#### 2.4.1 评估指标

```python
# ssdmfo/evaluation/metrics.py
class MetricsCalculator:
    """计算各种评估指标"""

    def __init__(self):
        self.epsilon = 1e-9  # 防止log(0)

    def jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """计算JS散度"""
        p = p.flatten()
        q = q.flatten()
        m = 0.5 * (p + q)
        jsd = 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)
        return jsd.item()

    def pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """计算Pearson相关系数（log域）"""
        x_log = torch.log(x + self.epsilon)
        y_log = torch.log(y + self.epsilon)

        # 只计算正值
        mask = (x > self.epsilon) | (y > self.epsilon)
        x_log = x_log[mask]
        y_log = y_log[mask]

        if len(x_log) < 2:
            return 0.0

        return torch.corrcoef(torch.stack([x_log, y_log]))[0, 1].item()

    def compute_all_metrics(self, generated: GeneratedStats,
                          real: Constraints) -> MetricsDict:
        """计算所有指标"""
        metrics = {}

        # 一阶指标
        metrics['jsd_H'] = self.jensen_shannon_divergence(
            generated.spatial_H, real.spatial.H
        )
        metrics['jsd_W'] = self.jensen_shannon_divergence(
            generated.spatial_W, real.spatial.W
        )
        metrics['jsd_O'] = self.jensen_shannon_divergence(
            generated.spatial_O, real.spatial.O
        )

        # OD指标
        for h in range(24):
            metrics[f'pcc_od_{h}'] = self.pearson_correlation(
                generated.od_flows[h], real.od_flows[h]
            )

        # 聚合指标
        metrics['mean_jsd'] = np.mean([metrics['jsd_H'],
                                      metrics['jsd_W'],
                                      metrics['jsd_O']])
        metrics['mean_pcc'] = np.mean([metrics[f'pcc_od_{h}']
                                      for h in range(24)])

        return metrics
```

#### 2.4.2 比较框架

```python
# ssdmfo/evaluation/comparison.py
class MethodComparison:
    """方法对比框架"""

    def __init__(self, constraints: Constraints, user_patterns: Dict):
        self.constraints = constraints
        self.user_patterns = user_patterns
        self.results = {}

    def add_method(self, name: str, method: BaseMethod):
        """添加待比较的方法"""
        self.methods[name] = method

    def run_comparison(self) -> ComparisonResults:
        """运行所有方法并比较"""

        for name, method in self.methods.items():
            print(f"Running {name}...")

            # 运行方法
            start_time = time.time()
            result = method.run(self.constraints, self.user_patterns)
            runtime = time.time() - start_time

            # 计算指标
            metrics = self.calculator.compute_all_metrics(
                result.generated_stats, self.constraints
            )

            # 存储结果
            self.results[name] = {
                'metrics': metrics,
                'runtime': runtime,
                'memory': self._measure_memory(),
                'result': result
            }

        return self._create_comparison_report()
```

### 2.5 实验管理 (ssdmfo/experiments/)

```python
# ssdmfo/experiments/runner.py
import hydra
from omegaconf import DictConfig
import wandb

class ExperimentRunner:
    """实验运行器"""

    @hydra.main(config_path="configs", config_name="default")
    def run(self, cfg: DictConfig):
        """主实验入口"""

        # 初始化wandb
        wandb.init(
            project="ssdmfo",
            config=cfg,
            name=f"{cfg.method}_{cfg.phase}"
        )

        # 加载数据
        data_loader = ConstraintDataLoader(cfg.data_dir)
        constraints = data_loader.load_all_constraints(cfg.phase)
        user_patterns = data_loader.load_user_patterns()

        # 选择方法
        if cfg.method == "ssdmfo":
            method = SSDMFO(cfg.ssdmfo)
        elif cfg.method == "ipf":
            method = IPF(cfg.ipf)
        elif cfg.method == "gravity":
            method = GravityModel(cfg.gravity)
        else:
            raise ValueError(f"Unknown method: {cfg.method}")

        # 运行
        result = method.run(constraints, user_patterns)

        # 评估
        metrics = evaluate(result, constraints)

        # 记录
        wandb.log(metrics)

        return result
```

---

## 三、第一步实施计划 (Phase 1)

### 3.1 目标与范围

**Phase 1 目标**：验证核心假设，建立基准性能

**约束范围**：仅一阶空间分布约束
- ✅ 空间分布 $\mu_c^{\text{real}}$
- ❌ 交互约束 $\pi_{cc'}^{\text{real}}$（Phase 2）
- ❌ OD约束 $F^{(h)}_{\text{real}}$（Phase 3）

### 3.2 实施步骤

```python
# phase1_implementation.py

def phase1_pipeline():
    """Phase 1 实施流程"""

    # Step 1: 数据准备（1天）
    prepare_phase1_data()

    # Step 2: 实现基线方法（2天）
    implement_baselines()

    # Step 3: 实现简化版SS-DMFO（3天）
    implement_ssdmfo_phase1()

    # Step 4: 对比实验（2天）
    run_comparison_experiments()

    # Step 5: 分析与报告（2天）
    analyze_results()

def prepare_phase1_data():
    """准备Phase 1数据"""
    # 1. 加载HWO_distribute.csv
    spatial_constraints = load_spatial_constraints()

    # 2. 采样10%用户作为开发集
    sampled_users = sample_users(ratio=0.1)  # ~9,336用户

    # 3. 预处理生活模式
    user_patterns = preprocess_life_patterns(sampled_users)

    # 4. 保存为HDF5格式
    save_phase1_data(spatial_constraints, user_patterns)

def implement_baselines():
    """实现基线方法"""
    methods = {
        "random": RandomAssignment(),      # 随机基线
        "uniform": UniformDistribution(),  # 均匀分布
        "gravity": SimpleGravityModel(),   # 重力模型
        "ipf": SpatialIPF()               # 仅空间约束的IPF
    }
    return methods

def implement_ssdmfo_phase1():
    """实现Phase 1版本的SS-DMFO"""
    class SSDMFO_Phase1:
        def __init__(self):
            # 仅初始化α（一阶势函数）
            self.alpha_H = torch.zeros(H, W)
            self.alpha_W = torch.zeros(H, W)
            self.alpha_O = torch.zeros(H, W)

        def optimize(self, constraints, users, config):
            # 简化版：无二阶项，无DP
            # 目标：仅满足空间分布约束
            pass
```

### 3.3 对比实验设计

```python
# experiments/phase1_comparison.py

class Phase1Comparison:
    """Phase 1 对比实验"""

    def __init__(self):
        self.methods = {
            "Random": RandomBaseline(),
            "Gravity": GravityModel(beta=1.5),
            "IPF": SpatialIPF(max_iter=100),
            "SSDMFO-P1": SSDMFO_Phase1(temperature=0.1)
        }

        self.metrics = [
            "jsd_H", "jsd_W", "jsd_O",  # 空间分布指标
            "runtime", "memory",          # 性能指标
            "convergence_iter"            # 收敛速度
        ]

    def run_experiment(self, n_runs=5):
        """运行对比实验"""
        results = defaultdict(list)

        for run in range(n_runs):
            for name, method in self.methods.items():
                # 运行方法
                result = self.run_single_method(method)

                # 记录结果
                results[name].append(result)

        # 统计分析
        summary = self.compute_statistics(results)

        # 生成报告
        self.generate_report(summary)

        return summary

    def expected_results(self):
        """预期结果（用于验证）"""
        return {
            "Random": {"jsd_mean": 0.5-0.8},     # 最差
            "Gravity": {"jsd_mean": 0.2-0.4},    # 中等
            "IPF": {"jsd_mean": 0.05-0.15},      # 较好
            "SSDMFO-P1": {"jsd_mean": 0.01-0.05} # 最好
        }
```

### 3.4 成功标准

| 指标 | 目标值 | 必须达到 |
|:----|:------|:--------|
| **JSD (H/W/O)** | < 0.1 | < 0.2 |
| **运行时间** | < 10分钟 | < 30分钟 |
| **内存占用** | < 4GB | < 8GB |
| **收敛轮数** | < 1000 | < 5000 |

### 3.5 风险与缓解

| 风险 | 概率 | 缓解措施 |
|:----|:----|:--------|
| MFVI不收敛 | 中 | 调整温度参数，增加阻尼 |
| 内存溢出 | 低 | 减少批次大小，使用gradient accumulation |
| 性能不如IPF | 低 | 分析原因，调整超参数 |

---

## 四、代码组织结构

```
ssdmfo/
├── __init__.py
├── setup.py
├── requirements.txt
├── README.md
├── ssdmfo/
│   ├── __init__.py
│   ├── core/               # 核心算法
│   │   ├── __init__.py
│   │   ├── dual_optimizer.py
│   │   ├── mean_field.py
│   │   ├── sparse_ops.py
│   │   └── potentials.py
│   ├── data/               # 数据处理
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── life_pattern.py
│   │   ├── structures.py
│   │   └── cache.py
│   ├── baselines/          # 基线方法
│   │   ├── __init__.py
│   │   ├── random.py
│   │   ├── gravity.py
│   │   ├── ipf.py
│   │   └── base.py
│   ├── evaluation/         # 评估系统
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── comparison.py
│   │   └── visualization.py
│   └── utils/              # 工具函数
│       ├── __init__.py
│       ├── profiler.py
│       ├── logger.py
│       └── config.py
├── experiments/            # 实验脚本
│   ├── configs/           # Hydra配置
│   │   ├── default.yaml
│   │   ├── method/
│   │   └── data/
│   ├── phase1_comparison.py
│   ├── ablation_study.py
│   └── scalability_test.py
├── tests/                  # 单元测试
│   ├── test_core/
│   ├── test_data/
│   └── test_baselines/
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_phase1_results.ipynb
│   └── 03_visualization.ipynb
└── scripts/                # 辅助脚本
    ├── download_data.sh
    ├── setup_env.sh
    └── run_phase1.sh
```

---

## 五、下一步行动

### 5.1 立即行动（本周）

```bash
# 1. 环境搭建
conda create -n ssdmfo python=3.10
conda activate ssdmfo
pip install -r requirements.txt

# 2. 数据预处理
python scripts/prepare_phase1_data.py \
    --data-dir ./data \
    --output-dir ./processed \
    --sample-ratio 0.1

# 3. 实现基线
python -m ssdmfo.baselines.gravity --test
python -m ssdmfo.baselines.ipf --test

# 4. 运行首个对比
python experiments/phase1_comparison.py \
    --methods random,gravity,ipf \
    --metrics jsd,runtime
```

### 5.2 Phase 1 时间线

| 任务 | 时间 | 交付物 |
|:----|:----|:------|
| 数据准备 | Day 1-2 | 预处理的HDF5数据集 |
| 基线实现 | Day 3-4 | 可运行的基线代码 |
| SS-DMFO Phase 1 | Day 5-7 | 简化版核心算法 |
| 对比实验 | Day 8-9 | 实验结果表格 |
| 分析报告 | Day 10 | Phase 1技术报告 |

### 5.3 成功后的Phase 2规划

如果Phase 1成功（JSD < 0.1），则进入Phase 2：
- 加入二阶交互约束
- 实现稀疏矩阵加速
- 扩展到全量用户
- 预期额外2-3周

---

## 六、附录：关键接口定义

```python
# 核心接口
class BaseMethod(ABC):
    """所有方法的基类"""

    @abstractmethod
    def run(self, constraints: Constraints,
           user_patterns: Dict[int, UserPattern]) -> Result:
        pass

class Result:
    """标准结果格式"""
    allocations: Dict[int, torch.Tensor]  # user_id -> (n_locs, grid_size)
    runtime: float
    iterations: int
    memory_peak: float

    def compute_statistics(self) -> GeneratedStats:
        """计算生成的统计量"""
        pass
```