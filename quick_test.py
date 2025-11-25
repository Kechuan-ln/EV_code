#!/usr/bin/env python3
"""快速测试各个基线方法"""

import sys
sys.path.insert(0, '/Volumes/FastACIS/Project/EVproject')

from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.random import RandomBaseline
from ssdmfo.baselines.gravity import GravityModel
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.evaluation.metrics import MetricsCalculator


def test_single_method(method, constraints, users):
    """测试单个方法"""
    print(f"\n{'=' * 60}")
    print(f"Testing: {method.name}")
    print('=' * 60)

    # 运行
    result = method.run(constraints, users)

    # 评估
    calculator = MetricsCalculator()
    generated_stats = result.compute_spatial_stats(
        users, constraints.grid_h, constraints.grid_w
    )
    generated_stats.normalize()

    metrics = calculator.compute_spatial_metrics(generated_stats, constraints.spatial)

    print(f"\nResults:")
    print(f"  JSD: H={metrics['jsd_H']:.4f}, W={metrics['jsd_W']:.4f}, O={metrics['jsd_O']:.4f}")
    print(f"  JSD mean: {metrics['jsd_mean']:.6f}")
    print(f"  Runtime: {result.runtime:.2f}s")
    print(f"  Memory: {result.memory_peak:.2f}MB")

    return metrics


def main():
    # 加载数据（小规模测试）
    print("Loading data...")
    loader = ConstraintDataLoader('/Volumes/FastACIS/Project/EVproject/EV_Splatting')
    constraints = loader.load_all_constraints(phase=1)
    users = loader.load_user_patterns(n_users=100)  # 先用100个用户快速测试

    # 测试各个方法
    methods = [
        ("Random", RandomBaseline()),
        ("Gravity", GravityModel(beta=1.5)),
        ("IPF", IterativeProportionalFitting(max_iter=20, tolerance=1e-3)),
    ]

    results = {}
    for name, method in methods:
        try:
            metrics = test_single_method(method, constraints, users)
            results[name] = metrics
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n\n" + "=" * 60)
    print("SUMMARY (100 users)")
    print("=" * 60)
    print(f"{'Method':<20} {'JSD Mean':<15} {'Status'}")
    print("-" * 60)
    for name, metrics in results.items():
        jsd = metrics['jsd_mean']
        status = "✓ Good" if jsd < 0.1 else ("~ OK" if jsd < 0.2 else "✗ Poor")
        print(f"{name:<20} {jsd:<15.6f} {status}")


if __name__ == '__main__':
    main()
