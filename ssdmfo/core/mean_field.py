"""平均场变分推断 (Mean Field Variational Inference)

给定势函数，计算每个用户的最优响应分布 Q_i。
核心思想：每个语义地点的分布独立更新，考虑势函数的影响。

Q_i(ℓ → g) ∝ exp(-α_c(g) - Σ_{ℓ'} β_{cc'}(g, g') * Q_i(ℓ' → g'))
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import sparse

from ..data.structures import UserPattern, Constraints
from .potentials import DualPotentials


class MeanFieldSolver:
    """平均场变分推断求解器"""

    def __init__(self,
                 temperature: float = 1.0,
                 max_iter: int = 10,
                 tolerance: float = 1e-4,
                 damping: float = 0.5):
        """
        Args:
            temperature: 温度参数，越小分布越尖锐
            max_iter: MFVI最大迭代次数
            tolerance: 收敛容差
            damping: 阻尼系数，防止震荡
        """
        self.temperature = temperature
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping = damping

    def compute_user_response(self,
                              user: UserPattern,
                              potentials: DualPotentials,
                              grid_h: int, grid_w: int,
                              phase: int = 1) -> np.ndarray:
        """计算单个用户的最优响应分布

        Args:
            user: 用户模式
            potentials: 当前势函数
            grid_h, grid_w: 栅格尺寸
            phase: 优化阶段

        Returns:
            Q: shape (n_locations, grid_size)，每个语义地点的空间分布
        """
        n_locs = len(user.locations)
        grid_size = grid_h * grid_w

        # 初始化：均匀分布
        Q = np.ones((n_locs, grid_size)) / grid_size

        # 预计算每个地点的类型和对应的一阶势
        loc_types = [loc.type for loc in user.locations]
        alpha_flat = {
            'H': potentials.alpha_H.flatten(),
            'W': potentials.alpha_W.flatten(),
            'O': potentials.alpha_O.flatten()
        }

        for iteration in range(self.max_iter):
            Q_old = Q.copy()

            # 逐个地点更新
            for loc_idx in range(n_locs):
                loc_type = loc_types[loc_idx]

                # 一阶场：来自α
                field = alpha_flat[loc_type].copy()

                # 二阶场：来自β和其他地点的Q（仅Phase 2）
                if phase >= 2:
                    field += self._compute_interaction_field(
                        loc_idx, loc_types, Q, potentials, grid_size
                    )

                # Boltzmann分布
                log_q = -field / self.temperature
                log_q -= log_q.max()  # 数值稳定性
                q_new = np.exp(log_q)
                q_new /= q_new.sum() + 1e-10

                # 阻尼更新
                Q[loc_idx] = self.damping * q_new + (1 - self.damping) * Q_old[loc_idx]

            # 检查收敛
            diff = np.abs(Q - Q_old).max()
            if diff < self.tolerance:
                break

        return Q

    def _compute_interaction_field(self,
                                   loc_idx: int,
                                   loc_types: List[str],
                                   Q: np.ndarray,
                                   potentials: DualPotentials,
                                   grid_size: int) -> np.ndarray:
        """计算二阶交互场

        field_ℓ(g) = Σ_{ℓ' ≠ ℓ} Σ_{g'} β_{cc'}(g,g') * Q(ℓ' → g')
        """
        field = np.zeros(grid_size)
        loc_type = loc_types[loc_idx]

        for other_idx, other_type in enumerate(loc_types):
            if other_idx == loc_idx:
                continue

            # 获取对应的β矩阵
            beta = potentials.get_beta(loc_type, other_type)
            if beta is None or beta.nnz == 0:
                continue

            # 稀疏矩阵-向量乘法: β @ Q[other]
            q_other = Q[other_idx]
            field += beta.dot(q_other)

        return field

    def compute_all_responses(self,
                              user_patterns: Dict[int, UserPattern],
                              potentials: DualPotentials,
                              constraints: Constraints,
                              phase: int = 1) -> Dict[int, np.ndarray]:
        """计算所有用户的响应分布

        Returns:
            responses: {user_id: Q} 字典
        """
        responses = {}
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w

        for user_id, user in user_patterns.items():
            Q = self.compute_user_response(
                user, potentials, grid_h, grid_w, phase
            )
            responses[user_id] = Q

        return responses


class FastMeanFieldSolver(MeanFieldSolver):
    """优化版平均场求解器（批处理 + 向量化）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_all_responses_fast(self,
                                   user_patterns: Dict[int, UserPattern],
                                   potentials: DualPotentials,
                                   constraints: Constraints,
                                   phase: int = 1) -> Dict[int, np.ndarray]:
        """快速计算所有用户响应（Phase 1优化版）

        对于Phase 1（无交互），所有同类型地点的分布相同，可以预计算。
        """
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        if phase == 1:
            # Phase 1: 所有同类型地点分布相同
            # Q_c(g) ∝ exp(-α_c(g) / T)
            Q_templates = {}
            for loc_type in ['H', 'W', 'O']:
                alpha = potentials.get_alpha(loc_type).flatten()
                log_q = -alpha / self.temperature
                log_q -= log_q.max()
                q = np.exp(log_q)
                q /= q.sum() + 1e-10
                Q_templates[loc_type] = q

            # 为每个用户分配模板
            responses = {}
            for user_id, user in user_patterns.items():
                n_locs = len(user.locations)
                Q = np.zeros((n_locs, grid_size))
                for loc_idx, loc in enumerate(user.locations):
                    Q[loc_idx] = Q_templates[loc.type]
                responses[user_id] = Q

            return responses

        else:
            # Phase 2+: 需要迭代求解
            return self.compute_all_responses(
                user_patterns, potentials, constraints, phase
            )
