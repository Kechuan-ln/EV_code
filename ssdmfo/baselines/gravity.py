"""重力模型基线"""
import numpy as np
from typing import Dict

from .base import BaseMethod
from ..data.structures import Constraints, UserPattern


class GravityModel(BaseMethod):
    """重力模型基线方法

    核心思想：P(location → grid) ∝ Mass(grid) / Distance^β
    其中 Mass 来自真实的空间分布
    """

    def __init__(self, beta: float = 1.5):
        """
        Args:
            beta: 距离衰减参数，越大表示距离影响越大
        """
        super().__init__(f"Gravity(β={beta})")
        self.beta = beta
        self._distance_cache = {}

    def _compute_distances(self, grid_h: int, grid_w: int) -> np.ndarray:
        """计算栅格中心点之间的距离矩阵

        Returns:
            distance_matrix: shape (grid_size, grid_size)
        """
        cache_key = (grid_h, grid_w)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # 创建栅格中心坐标
        y_coords = np.arange(grid_h)
        x_coords = np.arange(grid_w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        # 展平为 (grid_size, 2)
        coords = np.stack([yy.flatten(), xx.flatten()], axis=1)

        # 计算欧氏距离矩阵
        # 使用广播计算所有点对之间的距离
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (grid_size, grid_size, 2)
        distances = np.sqrt((diff ** 2).sum(axis=2))  # (grid_size, grid_size)

        # 避免除零
        distances = np.maximum(distances, 1e-3)

        self._distance_cache[cache_key] = distances
        return distances

    def _allocate_with_gravity(self,
                              mass: np.ndarray,
                              center_idx: int,
                              distances: np.ndarray) -> np.ndarray:
        """使用重力模型分配一个地点

        Args:
            mass: 栅格质量（吸引力） shape: (grid_size,)
            center_idx: 参考中心点索引
            distances: 距离矩阵 shape: (grid_size, grid_size)

        Returns:
            probs: 概率分布 shape: (grid_size,)
        """
        # 计算从中心点到所有栅格的距离
        dist_from_center = distances[center_idx]

        # 重力模型：P ∝ Mass / Distance^β
        gravity = mass / (dist_from_center ** self.beta)

        # 归一化
        probs = gravity / (gravity.sum() + 1e-10)

        return probs

    def _generate_allocations(self,
                             constraints: Constraints,
                             user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """使用重力模型生成分配"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        # 预计算距离矩阵
        print("Computing distance matrix...")
        distances = self._compute_distances(grid_h, grid_w)

        # 提取质量（使用真实的空间分布）
        masses = {
            'H': constraints.spatial.H.flatten(),
            'W': constraints.spatial.W.flatten(),
            'O': constraints.spatial.O.flatten()
        }

        # 为每个质量添加小的偏移，避免零质量
        for key in masses:
            masses[key] = masses[key] + 1e-6

        allocations = {}

        for user_id, pattern in user_patterns.items():
            n_locs = len(pattern.locations)
            alloc = np.zeros((n_locs, grid_size))

            # 为每个语义地点选择一个"中心"作为参考点
            # 策略：使用该类型的质量中心（质心）
            for loc_idx, location in enumerate(pattern.locations):
                mass = masses[location.type]

                # 计算质心作为中心点
                mass_2d = mass.reshape(grid_h, grid_w)
                y_center = (mass_2d.sum(axis=1) * np.arange(grid_h)).sum() / mass_2d.sum()
                x_center = (mass_2d.sum(axis=0) * np.arange(grid_w)).sum() / mass_2d.sum()

                # 转换为栅格索引
                center_y = int(np.clip(y_center, 0, grid_h - 1))
                center_x = int(np.clip(x_center, 0, grid_w - 1))
                center_idx = center_y * grid_w + center_x

                # 使用重力模型分配
                probs = self._allocate_with_gravity(mass, center_idx, distances)

                alloc[loc_idx] = probs

            allocations[user_id] = alloc

        return allocations


def test_gravity_model():
    """测试重力模型"""
    from ..data.loader import ConstraintDataLoader
    from ..evaluation.metrics import MetricsCalculator

    # 加载数据
    loader = ConstraintDataLoader('/Volumes/FastACIS/Project/EVproject/EV_Splatting')
    constraints = loader.load_all_constraints(phase=1)
    users = loader.load_user_patterns(n_users=100)

    # 测试不同的β值
    betas = [0.5, 1.0, 1.5, 2.0]
    calculator = MetricsCalculator()

    print("\n" + "=" * 60)
    print("Testing Gravity Model with different β values")
    print("=" * 60)

    for beta in betas:
        method = GravityModel(beta=beta)
        result = method.run(constraints, users)

        # 计算指标
        generated_stats = result.compute_spatial_stats(
            users, constraints.grid_h, constraints.grid_w
        )
        generated_stats.normalize()

        metrics = calculator.compute_spatial_metrics(generated_stats, constraints.spatial)

        print(f"\nβ = {beta}:")
        print(f"  JSD mean: {metrics['jsd_mean']:.6f}")
        print(f"  TVD mean: {metrics['tvd_mean']:.6f}")


if __name__ == '__main__':
    test_gravity_model()
