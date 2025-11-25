"""随机分配基线"""
import numpy as np
from typing import Dict

from .base import BaseMethod
from ..data.structures import Constraints, UserPattern


class RandomBaseline(BaseMethod):
    """随机分配基线方法"""

    def __init__(self):
        super().__init__("Random")

    def _generate_allocations(self,
                             constraints: Constraints,
                             user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """随机分配每个用户的语义地点到栅格"""
        grid_size = constraints.grid_h * constraints.grid_w
        allocations = {}

        for user_id, pattern in user_patterns.items():
            n_locs = len(pattern.locations)

            # 为每个地点随机生成概率分布
            alloc = np.random.rand(n_locs, grid_size)

            # 归一化（每个地点的概率和为1）
            alloc = alloc / alloc.sum(axis=1, keepdims=True)

            allocations[user_id] = alloc

        return allocations


def test_random_baseline():
    """测试随机基线"""
    from ..data.loader import ConstraintDataLoader

    # 加载数据
    loader = ConstraintDataLoader('/Volumes/FastACIS/Project/EVproject/EV_Splatting')
    constraints = loader.load_all_constraints(phase=1)
    users = loader.load_user_patterns(n_users=10)

    # 运行随机基线
    method = RandomBaseline()
    result = method.run(constraints, users)

    # 检查结果
    print(f"\nResult check:")
    print(f"  Allocations for {len(result.allocations)} users")

    # 检查一个用户
    user_id = list(result.allocations.keys())[0]
    alloc = result.allocations[user_id]
    print(f"  User {user_id} allocation shape: {alloc.shape}")
    print(f"  Row sums (should be ~1): {alloc.sum(axis=1)}")

    # 计算生成的空间统计
    generated_stats = result.compute_spatial_stats(
        users, constraints.grid_h, constraints.grid_w
    )
    print(f"\nGenerated spatial stats:")
    print(f"  H sum: {generated_stats.H.sum():.6f}")
    print(f"  W sum: {generated_stats.W.sum():.6f}")
    print(f"  O sum: {generated_stats.O.sum():.6f}")


if __name__ == '__main__':
    test_random_baseline()
