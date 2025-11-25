#!/usr/bin/env python3
"""Phase 1 测试脚本 - 完整的端到端测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.random import RandomBaseline
from ssdmfo.evaluation.metrics import MetricsCalculator


def main():
    """主测试函数"""
    print("=" * 60)
    print("Phase 1 End-to-End Test")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[Step 1] Loading data...")
    loader = ConstraintDataLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting'))

    # 加载约束
    constraints = loader.load_all_constraints(phase=1)
    print(f"✓ Loaded spatial constraints: {constraints.grid_h}x{constraints.grid_w}")

    # 加载用户（先测试100个用户）
    n_users = 1000
    user_patterns = loader.load_user_patterns(n_users=n_users)
    print(f"✓ Loaded {len(user_patterns)} user patterns")

    # 打印一些统计信息
    n_locs_list = [len(p.locations) for p in user_patterns.values()]
    print(f"  User location counts: min={min(n_locs_list)}, "
          f"max={max(n_locs_list)}, mean={sum(n_locs_list)/len(n_locs_list):.1f}")

    # Step 2: 运行随机基线
    print("\n[Step 2] Running Random Baseline...")
    random_method = RandomBaseline()
    result = random_method.run(constraints, user_patterns)

    # Step 3: 计算生成的统计
    print("\n[Step 3] Computing generated statistics...")
    generated_stats = result.compute_spatial_stats(
        user_patterns,
        constraints.grid_h,
        constraints.grid_w
    )
    generated_stats.normalize()

    print(f"✓ Generated spatial stats computed")
    print(f"  H sum: {generated_stats.H.sum():.6f}")
    print(f"  W sum: {generated_stats.W.sum():.6f}")
    print(f"  O sum: {generated_stats.O.sum():.6f}")

    # Step 4: 评估
    print("\n[Step 4] Evaluating...")
    calculator = MetricsCalculator()
    metrics = calculator.compute_spatial_metrics(generated_stats, constraints.spatial)
    calculator.print_metrics(metrics)

    # Step 5: 解读结果
    print("\n[Step 5] Result Interpretation:")
    print("-" * 60)
    if metrics['jsd_mean'] > 0.5:
        print("⚠ As expected, Random baseline performs poorly")
        print(f"  Mean JSD = {metrics['jsd_mean']:.4f} >> 0.1 (target)")
    else:
        print("⚠ Unexpected: Random performs better than expected!")

    print("\nNext steps:")
    print("  1. Implement Gravity model baseline")
    print("  2. Implement IPF baseline")
    print("  3. Implement SS-DMFO Phase 1")
    print("  4. Compare all methods")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
