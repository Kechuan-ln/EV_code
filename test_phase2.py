#!/usr/bin/env python3
"""Phase 2 测试脚本 - 验证交互约束的加载和评估"""

import sys
sys.path.insert(0, '/Volumes/FastACIS/Project/EVproject')

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.random import RandomBaseline
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.evaluation.metrics import MetricsCalculator


def main():
    """主测试函数"""
    print("=" * 70)
    print("PHASE 2 TEST - Spatial + Interaction Constraints")
    print("=" * 70)

    # Step 1: 加载Phase 2数据
    print("\n[Step 1] Loading Phase 2 constraints...")
    loader = ConstraintDataLoader('/Volumes/FastACIS/Project/EVproject/EV_Splatting')
    constraints = loader.load_all_constraints(phase=2)

    print(f"\nSpatial constraints: {constraints.grid_h} x {constraints.grid_w}")
    print(f"Interaction constraints:")
    print(f"  HW: {constraints.interaction.HW.nnz} non-zero entries")
    print(f"  HO: {constraints.interaction.HO.nnz} non-zero entries")
    print(f"  WO: {constraints.interaction.WO.nnz} non-zero entries")

    # Step 2: 加载用户
    print("\n[Step 2] Loading user patterns...")
    n_users = 100  # 先用小规模测试
    user_patterns = loader.load_user_patterns(n_users=n_users)
    print(f"Loaded {len(user_patterns)} users")

    # Step 3: 运行基线方法
    calculator = MetricsCalculator()

    methods = [
        ("Random", RandomBaseline()),
        ("IPF (Phase 1)", IterativeProportionalFitting(max_iter=20)),
    ]

    results_summary = {}

    for name, method in methods:
        print(f"\n{'=' * 70}")
        print(f"Testing: {name}")
        print('=' * 70)

        result = method.run(constraints, user_patterns)

        # 计算空间统计
        generated_spatial = result.compute_spatial_stats(
            user_patterns, constraints.grid_h, constraints.grid_w
        )
        generated_spatial.normalize()

        # 计算交互统计
        print("Computing interaction statistics...")
        generated_interaction = result.compute_interaction_stats(
            user_patterns, constraints.grid_h, constraints.grid_w
        )
        generated_interaction.normalize()

        print(f"Generated interaction stats:")
        print(f"  HW: {generated_interaction.HW.nnz} non-zero entries")
        print(f"  HO: {generated_interaction.HO.nnz} non-zero entries")
        print(f"  WO: {generated_interaction.WO.nnz} non-zero entries")

        # 计算所有指标
        metrics = calculator.compute_all_metrics(
            generated_spatial, constraints.spatial,
            generated_interaction, constraints.interaction
        )

        calculator.print_metrics(metrics, phase=2)

        results_summary[name] = metrics

    # Step 4: 对比总结
    print("\n\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Spatial JSD':<15} {'Interact JSD':<15} {'Total JSD':<15}")
    print("-" * 65)

    for name, metrics in results_summary.items():
        spatial_jsd = metrics['jsd_mean']
        interact_jsd = metrics.get('jsd_interaction_mean', 'N/A')
        total_jsd = metrics['jsd_total_mean']

        if isinstance(interact_jsd, float):
            print(f"{name:<20} {spatial_jsd:<15.6f} {interact_jsd:<15.6f} {total_jsd:<15.6f}")
        else:
            print(f"{name:<20} {spatial_jsd:<15.6f} {interact_jsd:<15} {total_jsd:<15.6f}")

    # Step 5: 分析
    print("\n\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    ipf_spatial = results_summary.get("IPF (Phase 1)", {}).get('jsd_mean', 1)
    ipf_interact = results_summary.get("IPF (Phase 1)", {}).get('jsd_interaction_mean', 1)

    print(f"\nIPF Phase 1 Results:")
    print(f"  Spatial JSD:     {ipf_spatial:.6f} {'(Good!)' if ipf_spatial < 0.1 else '(OK)' if ipf_spatial < 0.2 else '(Poor)'}")
    print(f"  Interaction JSD: {ipf_interact:.6f} {'(Good!)' if ipf_interact < 0.1 else '(OK)' if ipf_interact < 0.2 else '(Poor)'}")

    if ipf_spatial < 0.01 and ipf_interact > 0.3:
        print("\n>>> CONFIRMED: IPF achieves perfect spatial match but FAILS on interaction!")
        print(">>> This proves Phase 2 constraints are necessary and non-trivial.")
    elif ipf_interact < 0.1:
        print("\n>>> Interaction JSD is also low - need to investigate further.")
    else:
        print(f"\n>>> Interaction constraint adds real difficulty (JSD={ipf_interact:.4f})")


if __name__ == '__main__':
    main()
