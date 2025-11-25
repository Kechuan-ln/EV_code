#!/usr/bin/env python3
"""Test SS-DMFO GPU-Accelerated Version

Compare CPU vs GPU performance and results.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_gpu import SSDMFOv3GPU, GPUConfig, HAS_TORCH
from ssdmfo.evaluation.metrics import MetricsCalculator


def test_method(name, method, constraints, user_patterns, calculator, top_k=50):
    """Test a single method"""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print('=' * 70)

    method_start = time.time()
    result = method.run(constraints, user_patterns)

    # Compute spatial statistics
    print("Computing spatial statistics...")
    generated_spatial = result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    generated_spatial.normalize()

    # Compute interaction statistics
    print(f"Computing interaction statistics (top_k={top_k})...")
    generated_interaction = result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w,
        top_k=top_k
    )
    generated_interaction.normalize()

    print(f"\nGenerated interaction entries:")
    print(f"  HW: {generated_interaction.HW.nnz}")
    print(f"  HO: {generated_interaction.HO.nnz}")
    print(f"  WO: {generated_interaction.WO.nnz}")

    # Compute metrics
    metrics = calculator.compute_all_metrics(
        generated_spatial, constraints.spatial,
        generated_interaction, constraints.interaction
    )

    calculator.print_metrics(metrics, phase=2)
    total_time = time.time() - method_start
    print(f"Method total time: {total_time:.1f}s")

    return metrics, generated_interaction, total_time


def main():
    """Main test function"""
    print("=" * 70)
    print("SS-DMFO GPU ACCELERATION TEST")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed. Install with:")
        print("  pip install torch")
        return

    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    total_start = time.time()

    # Load data
    print("\n[Step 1] Loading data...")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    print(f"\nGrid: {constraints.grid_h} x {constraints.grid_w}")
    print(f"Real interaction constraints:")
    print(f"  HW: {constraints.interaction.HW.nnz} non-zero entries")
    print(f"  HO: {constraints.interaction.HO.nnz} non-zero entries")
    print(f"  WO: {constraints.interaction.WO.nnz} non-zero entries")

    # Test with different user counts
    calculator = MetricsCalculator()
    results = {}

    for n_users in [100, 500, 1000]:
        print(f"\n{'#' * 70}")
        print(f"# Testing with {n_users} users")
        print('#' * 70)

        # Load users
        print(f"\n[Step 2] Loading {n_users} user patterns...")
        user_patterns = loader.load_user_patterns(n_users=n_users)
        print(f"Loaded {len(user_patterns)} users")

        # IPF baseline (only for 100 users to save time)
        if n_users == 100:
            print("\n" + "=" * 70)
            print("BASELINE: IPF Phase 1")
            print("=" * 70)
            metrics_ipf, interact_ipf, time_ipf = test_method(
                "IPF Phase 1", IterativeProportionalFitting(max_iter=20),
                constraints, user_patterns, calculator
            )
            results[f"IPF-{n_users}"] = (metrics_ipf, interact_ipf, time_ipf)

        # GPU version
        print("\n" + "=" * 70)
        print(f"GPU: SS-DMFO v3 ({n_users} users)")
        print("=" * 70)

        config = GPUConfig(
            max_iter=60,
            gpu_batch_size=min(n_users, 500),  # Batch size for GPU (avoid OOM)
            mfvi_iter=3,
            temp_init=2.0,
            temp_final=1.0,
            gumbel_scale=0.3,
            gumbel_decay=0.995,
            gumbel_final=0.05,
            interaction_freq=3,
            top_k=50,
            log_freq=5,
            early_stop_patience=8,
            phase_separation=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        gpu_method = SSDMFOv3GPU(config)
        metrics_gpu, interact_gpu, time_gpu = test_method(
            f"SS-DMFO GPU ({n_users}u)",
            gpu_method,
            constraints, user_patterns, calculator
        )
        results[f"GPU-{n_users}"] = (metrics_gpu, interact_gpu, time_gpu)

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Spatial':<10} {'Interact':<10} {'HW nnz':<10} {'Time(s)':<10}")
    print("-" * 60)

    for name, (metrics, interact, elapsed) in results.items():
        spatial = metrics['jsd_mean']
        interact_jsd = metrics.get('jsd_interaction_mean', 0)
        nnz = interact.HW.nnz
        print(f"{name:<20} {spatial:<10.4f} {interact_jsd:<10.4f} {nnz:<10} {elapsed:<10.1f}")

    # GPU memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print(f"\nTotal execution time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
