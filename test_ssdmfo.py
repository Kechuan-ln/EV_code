#!/usr/bin/env python3
"""测试 SS-DMFO 优化器"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.random import RandomBaseline
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer import SSDMFOOptimizer, SSDMFOPhase2
from ssdmfo.evaluation.metrics import MetricsCalculator


def test_method(name, method, constraints, user_patterns, calculator, top_k=30):
    """测试单个方法"""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print('=' * 70)

    method_start = time.time()
    result = method.run(constraints, user_patterns)

    # 计算空间统计
    print("Computing spatial statistics...")
    generated_spatial = result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    generated_spatial.normalize()

    # 计算交互统计
    print(f"Computing interaction statistics (top_k={top_k})...")
    generated_interaction = result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w,
        top_k=top_k
    )
    generated_interaction.normalize()

    # 计算指标
    metrics = calculator.compute_all_metrics(
        generated_spatial, constraints.spatial,
        generated_interaction, constraints.interaction
    )

    calculator.print_metrics(metrics, phase=2)
    print(f"Method total time: {time.time() - method_start:.1f}s")

    return metrics


def main():
    """主测试函数"""
    print("=" * 70)
    print("SS-DMFO OPTIMIZATION TEST")
    print("=" * 70)

    total_start = time.time()

    # 加载数据
    print("\n[Step 1] Loading data...")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    print(f"\nGrid: {constraints.grid_h} x {constraints.grid_w}")
    print(f"Interaction constraints: HW={constraints.interaction.HW.nnz}, "
          f"HO={constraints.interaction.HO.nnz}, WO={constraints.interaction.WO.nnz}")

    # 加载用户
    print("\n[Step 2] Loading user patterns...")
    n_users = 100
    user_patterns = loader.load_user_patterns(n_users=n_users)
    print(f"Loaded {len(user_patterns)} users")

    # 测试方法
    calculator = MetricsCalculator()
    results = {}

    # 1. Random baseline
    results["Random"] = test_method(
        "Random", RandomBaseline(),
        constraints, user_patterns, calculator
    )

    # 2. IPF Phase 1 (空间约束)
    results["IPF-P1"] = test_method(
        "IPF Phase 1", IterativeProportionalFitting(max_iter=20),
        constraints, user_patterns, calculator
    )

    # 3. SS-DMFO Phase 1 (仅空间)
    results["SSDMFO-P1"] = test_method(
        "SS-DMFO Phase 1",
        SSDMFOOptimizer(phase=1, max_iter=50, lr=0.1, temperature=1.0),
        constraints, user_patterns, calculator
    )

    # 4. SS-DMFO Phase 2 (空间+交互)
    results["SSDMFO-P2"] = test_method(
        "SS-DMFO Phase 2",
        SSDMFOPhase2(max_iter=100, lr=0.05, temperature=0.5),
        constraints, user_patterns, calculator
    )

    # 总结
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<15} {'Spatial':<12} {'Interact':<12} {'Total':<12}")
    print("-" * 55)

    for name, metrics in results.items():
        spatial = metrics['jsd_mean']
        interact = metrics.get('jsd_interaction_mean', 0)
        total = metrics['jsd_total_mean']
        print(f"{name:<15} {spatial:<12.4f} {interact:<12.4f} {total:<12.4f}")

    # 分析
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    ipf_spatial = results["IPF-P1"]['jsd_mean']
    ipf_interact = results["IPF-P1"]['jsd_interaction_mean']
    ssdmfo_spatial = results["SSDMFO-P2"]['jsd_mean']
    ssdmfo_interact = results["SSDMFO-P2"]['jsd_interaction_mean']

    print(f"\nIPF Phase 1:")
    print(f"  Spatial:     {ipf_spatial:.4f}")
    print(f"  Interaction: {ipf_interact:.4f}")

    print(f"\nSS-DMFO Phase 2:")
    print(f"  Spatial:     {ssdmfo_spatial:.4f}")
    print(f"  Interaction: {ssdmfo_interact:.4f}")

    if ssdmfo_interact < ipf_interact:
        improvement = (ipf_interact - ssdmfo_interact) / ipf_interact * 100
        print(f"\n>>> SS-DMFO improves interaction by {improvement:.1f}%!")
    else:
        print(f"\n>>> SS-DMFO needs tuning for better interaction performance.")

    print(f"\nTotal execution time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
