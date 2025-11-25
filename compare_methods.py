#!/usr/bin/env python3
"""方法对比脚本 - 运行所有基线方法并对比结果"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import defaultdict
import time

from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.random import RandomBaseline
from ssdmfo.baselines.gravity import GravityModel
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.evaluation.metrics import MetricsCalculator


class MethodComparison:
    """方法对比框架"""

    def __init__(self, constraints, user_patterns):
        self.constraints = constraints
        self.user_patterns = user_patterns
        self.calculator = MetricsCalculator()
        self.results = defaultdict(list)

    def add_method(self, method):
        """运行一个方法并记录结果"""
        print(f"\n{'=' * 70}")
        print(f"Testing: {method.name}")
        print('=' * 70)

        # 运行方法
        result = method.run(self.constraints, self.user_patterns)

        # 计算生成的统计
        generated_stats = result.compute_spatial_stats(
            self.user_patterns,
            self.constraints.grid_h,
            self.constraints.grid_w
        )
        generated_stats.normalize()

        # 计算指标
        metrics = self.calculator.compute_spatial_metrics(
            generated_stats, self.constraints.spatial
        )

        # 存储结果
        self.results[method.name].append({
            'runtime': result.runtime,
            'memory': result.memory_peak,
            'iterations': result.iterations,
            'jsd_H': metrics['jsd_H'],
            'jsd_W': metrics['jsd_W'],
            'jsd_O': metrics['jsd_O'],
            'jsd_mean': metrics['jsd_mean'],
            'tvd_H': metrics['tvd_H'],
            'tvd_W': metrics['tvd_W'],
            'tvd_O': metrics['tvd_O'],
            'tvd_mean': metrics['tvd_mean'],
        })

        print(f"\nResults:")
        print(f"  JSD mean: {metrics['jsd_mean']:.6f}")
        print(f"  TVD mean: {metrics['tvd_mean']:.6f}")
        print(f"  Runtime: {result.runtime:.2f}s")

        return metrics

    def run_comparison(self, methods, n_runs=3):
        """运行所有方法的对比实验"""
        print("\n" + "=" * 70)
        print("PHASE 1 METHOD COMPARISON")
        print("=" * 70)
        print(f"Users: {len(self.user_patterns)}")
        print(f"Grid: {self.constraints.grid_h} × {self.constraints.grid_w}")
        print(f"Methods: {len(methods)}")
        print(f"Runs per method: {n_runs}")
        print("=" * 70)

        for run in range(n_runs):
            print(f"\n\n{'#' * 70}")
            print(f"# RUN {run + 1}/{n_runs}")
            print('#' * 70)

            for method in methods:
                self.add_method(method)

        return self.generate_report()

    def generate_report(self):
        """生成对比报告"""
        print("\n\n" + "=" * 70)
        print("COMPARISON REPORT")
        print("=" * 70)

        # 计算统计
        summary = {}
        for method_name, runs in self.results.items():
            summary[method_name] = {
                'jsd_mean': {
                    'mean': np.mean([r['jsd_mean'] for r in runs]),
                    'std': np.std([r['jsd_mean'] for r in runs]),
                },
                'tvd_mean': {
                    'mean': np.mean([r['tvd_mean'] for r in runs]),
                    'std': np.std([r['tvd_mean'] for r in runs]),
                },
                'runtime': {
                    'mean': np.mean([r['runtime'] for r in runs]),
                    'std': np.std([r['runtime'] for r in runs]),
                },
                'memory': {
                    'mean': np.mean([r['memory'] for r in runs]),
                    'std': np.std([r['memory'] for r in runs]),
                }
            }

        # 打印表格
        print("\nMetric: JSD (Jensen-Shannon Divergence)")
        print("-" * 70)
        print(f"{'Method':<25} {'Mean':<15} {'Std':<15} {'Target':<15}")
        print("-" * 70)
        for method_name, stats in sorted(summary.items(), key=lambda x: x[1]['jsd_mean']['mean']):
            mean_val = stats['jsd_mean']['mean']
            std_val = stats['jsd_mean']['std']
            status = "✓" if mean_val < 0.1 else ("~" if mean_val < 0.2 else "✗")
            print(f"{method_name:<25} {mean_val:<15.6f} {std_val:<15.6f} {status}")

        print("\n\nMetric: Runtime (seconds)")
        print("-" * 70)
        print(f"{'Method':<25} {'Mean':<15} {'Std':<15}")
        print("-" * 70)
        for method_name, stats in sorted(summary.items(), key=lambda x: x[1]['runtime']['mean']):
            mean_val = stats['runtime']['mean']
            std_val = stats['runtime']['std']
            print(f"{method_name:<25} {mean_val:<15.2f} {std_val:<15.2f}")

        print("\n\nMetric: Memory Peak (MB)")
        print("-" * 70)
        print(f"{'Method':<25} {'Mean':<15} {'Std':<15}")
        print("-" * 70)
        for method_name, stats in sorted(summary.items(), key=lambda x: x[1]['memory']['mean']):
            mean_val = stats['memory']['mean']
            std_val = stats['memory']['std']
            print(f"{method_name:<25} {mean_val:<15.2f} {std_val:<15.2f}")

        # 评分
        print("\n\nMethod Ranking (by JSD):")
        print("-" * 70)
        ranked = sorted(summary.items(), key=lambda x: x[1]['jsd_mean']['mean'])
        for rank, (method_name, stats) in enumerate(ranked, 1):
            jsd = stats['jsd_mean']['mean']
            print(f"{rank}. {method_name}: JSD = {jsd:.6f}")

        print("\n" + "=" * 70)

        return summary


def main():
    """主函数"""
    # 加载数据
    print("Loading data...")
    loader = ConstraintDataLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting'))
    constraints = loader.load_all_constraints(phase=1)

    # 使用1000用户测试
    n_users = 1000
    user_patterns = loader.load_user_patterns(n_users=n_users)

    # 定义要测试的方法
    methods = [
        RandomBaseline(),
        GravityModel(beta=1.5),
        IterativeProportionalFitting(max_iter=50, tolerance=1e-4),
    ]

    # 运行对比
    comparison = MethodComparison(constraints, user_patterns)
    summary = comparison.run_comparison(methods, n_runs=3)

    # 保存结果
    print("\nComparison completed!")


if __name__ == '__main__':
    main()
