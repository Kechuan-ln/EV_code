"""迭代比例拟合(IPF)基线"""
import numpy as np
from typing import Dict

from .base import BaseMethod
from ..data.structures import Constraints, UserPattern, SpatialConstraints


class IterativeProportionalFitting(BaseMethod):
    """迭代比例拟合基线方法

    核心思想：交替调整分配矩阵使其边缘分布匹配目标约束
    这是一个经典的Sinkhorn-Knopp类型算法
    """

    def __init__(self, max_iter: int = 100, tolerance: float = 1e-4):
        """
        Args:
            max_iter: 最大迭代次数
            tolerance: 收敛容差（JSD变化小于此值则停止）
        """
        super().__init__(f"IPF(iter={max_iter})")
        self.max_iter = max_iter
        self.tolerance = tolerance

    def _initialize_allocations(self,
                                constraints: Constraints,
                                user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """初始化分配（使用均匀分布）"""
        grid_size = constraints.grid_h * constraints.grid_w
        allocations = {}

        for user_id, pattern in user_patterns.items():
            n_locs = len(pattern.locations)
            # 均匀初始化
            alloc = np.ones((n_locs, grid_size)) / grid_size
            allocations[user_id] = alloc

        return allocations

    def _compute_current_distribution(self,
                                     allocations: Dict[int, np.ndarray],
                                     user_patterns: Dict[int, UserPattern],
                                     grid_h: int,
                                     grid_w: int) -> SpatialConstraints:
        """计算当前分配的聚合分布"""
        H_map = np.zeros((grid_h, grid_w))
        W_map = np.zeros((grid_h, grid_w))
        O_map = np.zeros((grid_h, grid_w))

        for user_id, alloc in allocations.items():
            pattern = user_patterns[user_id]

            for loc_idx, location in enumerate(pattern.locations):
                probs = alloc[loc_idx].reshape(grid_h, grid_w)

                if location.type == 'H':
                    H_map += probs
                elif location.type == 'W':
                    W_map += probs
                elif location.type == 'O':
                    O_map += probs

        return SpatialConstraints(H=H_map, W=W_map, O=O_map)

    def _adjust_allocations(self,
                           allocations: Dict[int, np.ndarray],
                           user_patterns: Dict[int, UserPattern],
                           target: SpatialConstraints,
                           current: SpatialConstraints) -> Dict[int, np.ndarray]:
        """调整分配以匹配目标分布"""
        grid_h, grid_w = target.shape

        # 计算缩放因子（避免除零）
        scale_factors = {
            'H': (target.H + 1e-10) / (current.H + 1e-10),
            'W': (target.W + 1e-10) / (current.W + 1e-10),
            'O': (target.O + 1e-10) / (current.O + 1e-10)
        }

        # 调整每个用户的分配
        new_allocations = {}

        for user_id, alloc in allocations.items():
            pattern = user_patterns[user_id]
            new_alloc = alloc.copy()

            for loc_idx, location in enumerate(pattern.locations):
                # 获取对应的缩放因子
                scale = scale_factors[location.type].flatten()

                # 应用缩放
                new_alloc[loc_idx] = alloc[loc_idx] * scale

                # 重新归一化（保证每个地点的概率和为1）
                new_alloc[loc_idx] = new_alloc[loc_idx] / (new_alloc[loc_idx].sum() + 1e-10)

            new_allocations[user_id] = new_alloc

        return new_allocations

    def _compute_jsd(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算JSD（用于监控收敛）"""
        p = p.flatten() + 1e-10
        q = q.flatten() + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        kl1 = np.sum(p * np.log(p / m))
        kl2 = np.sum(q * np.log(q / m))
        return 0.5 * (kl1 + kl2)

    def _generate_allocations(self,
                             constraints: Constraints,
                             user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """使用IPF算法生成分配"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w

        # 归一化目标分布
        target = constraints.spatial
        target.normalize()

        # 初始化
        print("Initializing allocations...")
        allocations = self._initialize_allocations(constraints, user_patterns)

        # IPF迭代
        print(f"Running IPF iterations (max_iter={self.max_iter})...")
        prev_jsd = float('inf')

        for iteration in range(self.max_iter):
            # 计算当前分布
            current = self._compute_current_distribution(
                allocations, user_patterns, grid_h, grid_w
            )
            current.normalize()

            # 计算JSD（监控收敛）
            jsd_H = self._compute_jsd(current.H, target.H)
            jsd_W = self._compute_jsd(current.W, target.W)
            jsd_O = self._compute_jsd(current.O, target.O)
            mean_jsd = (jsd_H + jsd_W + jsd_O) / 3

            # 打印进度
            if iteration % 10 == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: JSD = {mean_jsd:.6f}")

            # 检查收敛
            if abs(prev_jsd - mean_jsd) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break

            prev_jsd = mean_jsd

            # 调整分配
            allocations = self._adjust_allocations(
                allocations, user_patterns, target, current
            )

        return allocations


def test_ipf():
    """测试IPF方法"""
    from ..data.loader import ConstraintDataLoader
    from ..evaluation.metrics import MetricsCalculator

    # 加载数据
    import os
    loader = ConstraintDataLoader(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'EV_Splatting'))
    constraints = loader.load_all_constraints(phase=1)
    users = loader.load_user_patterns(n_users=100)

    # 运行IPF
    method = IterativeProportionalFitting(max_iter=100, tolerance=1e-5)
    result = method.run(constraints, users)

    # 评估
    calculator = MetricsCalculator()
    generated_stats = result.compute_spatial_stats(
        users, constraints.grid_h, constraints.grid_w
    )
    generated_stats.normalize()

    metrics = calculator.compute_spatial_metrics(generated_stats, constraints.spatial)
    calculator.print_metrics(metrics)


if __name__ == '__main__':
    test_ipf()
