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
                                  grid_h: int, grid_w: int,
                                  top_k: int = 50) -> InteractionConstraints:
        """从分配计算交互统计（二阶联合分布）

        使用top-k截断策略：只保留每个分配中概率最高的k个位置
        这样避免了O(grid_size^2)的内存和计算开销

        Args:
            top_k: 每个分配保留的位置数（默认50，内存需求约50×50×用户数×20对）
        """
        grid_size = grid_h * grid_w

        # 使用字典累积稀疏结果
        hw_dict = {}
        ho_dict = {}
        wo_dict = {}

        def get_top_k(probs, k):
            """获取概率最高的k个位置"""
            if k >= len(probs):
                indices = np.where(probs > 1e-10)[0]
                return indices, probs[indices]
            # argpartition比argsort快
            top_idx = np.argpartition(probs, -k)[-k:]
            top_probs = probs[top_idx]
            # 过滤极小值
            mask = top_probs > 1e-10
            return top_idx[mask], top_probs[mask]

        def accumulate_outer(d, indices1, probs1, indices2, probs2):
            """累积外积到字典"""
            for i, idx1 in enumerate(indices1):
                p1 = probs1[i]
                for j, idx2 in enumerate(indices2):
                    val = p1 * probs2[j]
                    if val > 1e-15:
                        key = (int(idx1), int(idx2))
                        d[key] = d.get(key, 0.0) + val

        n_users = len(self.allocations)
        for i, (user_id, alloc) in enumerate(self.allocations.items()):
            if (i + 1) % 100 == 0:
                print(f"    Processing user {i+1}/{n_users}...")

            pattern = user_patterns[user_id]

            # 按类型分组并预计算top-k
            h_data = [(idx, *get_top_k(alloc[idx], top_k))
                      for idx, loc in enumerate(pattern.locations) if loc.type == 'H']
            w_data = [(idx, *get_top_k(alloc[idx], top_k))
                      for idx, loc in enumerate(pattern.locations) if loc.type == 'W']
            o_data = [(idx, *get_top_k(alloc[idx], top_k))
                      for idx, loc in enumerate(pattern.locations) if loc.type == 'O']

            # HW交互
            for _, h_indices, h_probs in h_data:
                for _, w_indices, w_probs in w_data:
                    accumulate_outer(hw_dict, h_indices, h_probs, w_indices, w_probs)

            # HO交互
            for _, h_indices, h_probs in h_data:
                for _, o_indices, o_probs in o_data:
                    accumulate_outer(ho_dict, h_indices, h_probs, o_indices, o_probs)

            # WO交互
            for _, w_indices, w_probs in w_data:
                for _, o_indices, o_probs in o_data:
                    accumulate_outer(wo_dict, w_indices, w_probs, o_indices, o_probs)

        def dict_to_sparse(d):
            if not d:
                return sparse.csr_matrix((grid_size, grid_size))
            rows, cols, data = zip(*[(k[0], k[1], v) for k, v in d.items()])
            return sparse.csr_matrix((data, (rows, cols)), shape=(grid_size, grid_size))

        print(f"  Building sparse matrices (HW:{len(hw_dict)}, HO:{len(ho_dict)}, WO:{len(wo_dict)} entries)...")
        HW = dict_to_sparse(hw_dict)
        HO = dict_to_sparse(ho_dict)
        WO = dict_to_sparse(wo_dict)

        return InteractionConstraints(HW=HW, HO=HO, WO=WO)
