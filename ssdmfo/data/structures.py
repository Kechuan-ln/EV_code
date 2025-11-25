"""数据结构定义"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse


@dataclass
class SemanticLocation:
    """语义地点"""
    type: str  # 'H', 'W', 'O'
    index: int  # 0, 1, 2, ...

    def __str__(self):
        return f"{self.type}_{self.index}"

    def __repr__(self):
        return self.__str__()


@dataclass
class UserPattern:
    """用户生活模式"""
    user_id: int
    locations: List[SemanticLocation]  # 语义地点列表
    activity_probs: Optional[Dict[int, np.ndarray]] = None  # P(ℓ|h)
    transition_probs: Optional[np.ndarray] = None  # P(ℓ'|ℓ,h)

    def __post_init__(self):
        if self.activity_probs is None:
            self.activity_probs = {}
        if self.transition_probs is None:
            # 默认均匀转移
            n_locs = len(self.locations)
            self.transition_probs = np.ones((24, n_locs, n_locs)) / n_locs


@dataclass
class SpatialConstraints:
    """一阶空间分布约束"""
    H: np.ndarray  # shape: (grid_h, grid_w)
    W: np.ndarray  # shape: (grid_h, grid_w)
    O: np.ndarray  # shape: (grid_h, grid_w)

    def normalize(self):
        """归一化为概率分布"""
        self.H = self.H / (self.H.sum() + 1e-10)
        self.W = self.W / (self.W.sum() + 1e-10)
        self.O = self.O / (self.O.sum() + 1e-10)
        return self

    @property
    def shape(self):
        return self.H.shape


@dataclass
class InteractionConstraints:
    """二阶交互约束（稀疏矩阵）

    每个矩阵 shape: (grid_size, grid_size)，表示两种地点类型的联合分布
    使用稀疏矩阵存储以节省内存
    """
    HW: sparse.csr_matrix  # P(H位置, W位置)
    HO: sparse.csr_matrix  # P(H位置, O位置)
    WO: sparse.csr_matrix  # P(W位置, O位置)

    def normalize(self):
        """归一化为概率分布"""
        self.HW = self.HW / (self.HW.sum() + 1e-10)
        self.HO = self.HO / (self.HO.sum() + 1e-10)
        self.WO = self.WO / (self.WO.sum() + 1e-10)
        return self

    @property
    def total_nnz(self) -> int:
        """总非零元素数"""
        return self.HW.nnz + self.HO.nnz + self.WO.nnz


@dataclass
class Constraints:
    """所有约束的容器"""
    spatial: SpatialConstraints
    interaction: Optional[InteractionConstraints] = None

    @property
    def grid_h(self):
        return self.spatial.H.shape[0]

    @property
    def grid_w(self):
        return self.spatial.H.shape[1]

    @property
    def grid_size(self):
        return self.grid_h * self.grid_w


@dataclass
class Result:
    """方法运行结果"""
    method_name: str
    allocations: Dict[int, np.ndarray]  # user_id -> (n_locs, grid_size)
    runtime: float
    iterations: int = 0
    memory_peak: float = 0.0

    def compute_spatial_stats(self, user_patterns: Dict[int, UserPattern],
                             grid_h: int, grid_w: int) -> SpatialConstraints:
        """从分配计算空间统计"""
        H_map = np.zeros((grid_h, grid_w))
        W_map = np.zeros((grid_h, grid_w))
        O_map = np.zeros((grid_h, grid_w))

        for user_id, alloc in self.allocations.items():
            pattern = user_patterns[user_id]
            # alloc: (n_locs, grid_size)
            for loc_idx, location in enumerate(pattern.locations):
                probs = alloc[loc_idx].reshape(grid_h, grid_w)

                if location.type == 'H':
                    H_map += probs
                elif location.type == 'W':
                    W_map += probs
                elif location.type == 'O':
                    O_map += probs

        return SpatialConstraints(H=H_map, W=W_map, O=O_map)

    def compute_interaction_stats(self, user_patterns: Dict[int, UserPattern],
                                  grid_h: int, grid_w: int) -> InteractionConstraints:
        """从分配计算交互统计（二阶联合分布） - Optimized
        
        使用矩阵乘法代替嵌套循环以加速计算。
        原理: sum(outer(v1, v2)) = V1.T @ V2
        """
        grid_size = grid_h * grid_w
        
        def compute_matrix(type1, type2, batch_size=1000):
            # 预分配累加矩阵 (使用float32节省内存)
            # 注意: 40000x40000 float32 约占 6.4GB 内存
            total_matrix = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            # 收集所有需要计算的对
            pairs = []
            for user_id, alloc in self.allocations.items():
                pattern = user_patterns[user_id]
                indices1 = [i for i, loc in enumerate(pattern.locations) if loc.type == type1]
                indices2 = [i for i, loc in enumerate(pattern.locations) if loc.type == type2]
                
                for i1 in indices1:
                    for i2 in indices2:
                        pairs.append((alloc[i1], alloc[i2]))
            
            # 分批处理以控制中间变量内存
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                if not batch:
                    continue
                    
                # 堆叠批次数据
                # vec1: (batch, grid), vec2: (batch, grid)
                vecs1 = np.stack([p[0] for p in batch]).astype(np.float32)
                vecs2 = np.stack([p[1] for p in batch]).astype(np.float32)
                
                # 矩阵乘法累加: (Grid, Batch) @ (Batch, Grid) -> (Grid, Grid)
                # 这等价于对batch中每一对向量做外积然后求和
                total_matrix += np.dot(vecs1.T, vecs2)
                
            # 阈值处理保持稀疏性 (匹配原逻辑)
            total_matrix[total_matrix < 1e-15] = 0
            
            return sparse.csr_matrix(total_matrix)

        print("  Computing HW interaction (Vectorized)...")
        HW = compute_matrix('H', 'W')
        print("  Computing HO interaction (Vectorized)...")
        HO = compute_matrix('H', 'O')
        print("  Computing WO interaction (Vectorized)...")
        WO = compute_matrix('W', 'O')

        return InteractionConstraints(HW=HW, HO=HO, WO=WO)
