#!/usr/bin/env python3
"""Test SS-DMFO 3.0 Sparse Optimizer

Based on expert recommendations:
1. Use real constraint support set S (not top_k approximation)
2. SDDMM for outer loop (O(N·|S|) instead of O(N·G²))
3. SpMM for inner loop MFVI
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_sparse import SSDMFOSparse, SparseConfig
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
    print("SS-DMFO 3.0 SPARSE OPTIMIZER TEST")
    print("=" * 70)
    print("\nKey innovations:")
    print("  1. Use real constraint support set S (not top_k)")
    print("  2. SDDMM for O(N·|S|) interaction aggregation")
    print("  3. SpMM for efficient MFVI")
    print("  4. Gumbel noise for micro-diversity")

    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("ERROR: PyTorch not installed")
        return

    total_start = time.time()

    # Load data
    print("\n[Step 1] Loading data...")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    print(f"\nGrid: {constraints.grid_h} x {constraints.grid_w}")
    print(f"Real interaction constraints (Support Set S):")
    print(f"  HW: {constraints.interaction.HW.nnz:,} non-zero entries")
    print(f"  HO: {constraints.interaction.HO.nnz:,} non-zero entries")
    print(f"  WO: {constraints.interaction.WO.nnz:,} non-zero entries")
    total_support = (constraints.interaction.HW.nnz +
                     constraints.interaction.HO.nnz +
                     constraints.interaction.WO.nnz)
    grid_squared = (constraints.grid_h * constraints.grid_w) ** 2
    print(f"  Total: {total_support:,} / {grid_squared:,} = {100*total_support/grid_squared:.4f}%")

    calculator = MetricsCalculator()
    results = {}

    # Test with increasing user counts
    for n_users in [100, 1000, 5000]:
        print(f"\n{'#' * 70}")
        print(f"# Testing with {n_users} users")
        print('#' * 70)

        # Load users
        print(f"\n[Step 2] Loading {n_users} user patterns...")
        user_patterns = loader.load_user_patterns(n_users=n_users)
        print(f"Loaded {len(user_patterns)} users")

        # IPF baseline (only for 100 users)
        if n_users == 100:
            print("\n" + "=" * 70)
            print("BASELINE: IPF Phase 1")
            print("=" * 70)
            metrics_ipf, interact_ipf, time_ipf = test_method(
                "IPF Phase 1", IterativeProportionalFitting(max_iter=20),
                constraints, user_patterns, calculator
            )
            results[f"IPF-{n_users}"] = (metrics_ipf, interact_ipf, time_ipf)

        # Sparse SS-DMFO
        print("\n" + "=" * 70)
        print(f"SPARSE: SS-DMFO 3.0 ({n_users} users)")
        print("=" * 70)

        config = SparseConfig(
            max_iter=100,
            gpu_batch_size=min(n_users, 500),   # Reduced to avoid OOM
            sddmm_batch_size=100,               # Small batches for SDDMM
            lr_alpha=0.1,
            lr_beta=0.05,
            mfvi_iter=3,
            temp_init=2.0,
            temp_final=0.5,
            gumbel_scale=0.3,
            gumbel_decay=0.995,
            gumbel_final=0.05,
            interaction_freq=2,
            log_freq=10,
            early_stop_patience=15,
            phase_separation=True,
            phase1_ratio=0.15,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        sparse_method = SSDMFOSparse(config)
        metrics_sparse, interact_sparse, time_sparse = test_method(
            f"SS-DMFO Sparse ({n_users}u)",
            sparse_method,
            constraints, user_patterns, calculator
        )
        results[f"Sparse-{n_users}"] = (metrics_sparse, interact_sparse, time_sparse)

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Spatial':<10} {'Interact':<10} {'HW nnz':<15} {'Time(s)':<10}")
    print("-" * 70)

    for name, (metrics, interact, elapsed) in results.items():
        spatial = metrics['jsd_mean']
        interact_jsd = metrics.get('jsd_interaction_mean', 0)
        nnz = interact.HW.nnz
        print(f"{name:<20} {spatial:<10.4f} {interact_jsd:<10.4f} {nnz:<15,} {elapsed:<10.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if "IPF-100" in results and "Sparse-100" in results:
        ipf_interact = results["IPF-100"][0]['jsd_interaction_mean']
        sparse_interact = results["Sparse-100"][0]['jsd_interaction_mean']

        print(f"\nIPF vs Sparse (100 users):")
        print(f"  IPF Interaction JSD:    {ipf_interact:.4f}")
        print(f"  Sparse Interaction JSD: {sparse_interact:.4f}")

        if sparse_interact < ipf_interact:
            improvement = (ipf_interact - sparse_interact) / ipf_interact * 100
            print(f"  >>> Sparse improves by {improvement:.1f}%!")
        else:
            print(f"  >>> IPF is still better")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print(f"\nTotal execution time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
