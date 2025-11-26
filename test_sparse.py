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


def compute_support_jsd(generated_interaction, real_interaction):
    """Compute JSD on the support set (positions where real > 0)"""
    def sparse_jsd_on_support(gen, real):
        """JSD computed only on real's support set"""
        real_coo = real.tocoo()
        if real_coo.nnz == 0:
            return 0.0

        # Get values at real's support positions
        real_vals = real_coo.data + 1e-10
        gen_vals = np.array(gen[real_coo.row, real_coo.col]).flatten() + 1e-10

        # Normalize
        real_vals = real_vals / real_vals.sum()
        gen_vals = gen_vals / gen_vals.sum()

        m = 0.5 * (real_vals + gen_vals)
        jsd = 0.5 * (np.sum(real_vals * np.log(real_vals / m)) +
                     np.sum(gen_vals * np.log(gen_vals / m)))
        return jsd

    hw_jsd = sparse_jsd_on_support(generated_interaction.HW, real_interaction.HW)
    ho_jsd = sparse_jsd_on_support(generated_interaction.HO, real_interaction.HO)
    wo_jsd = sparse_jsd_on_support(generated_interaction.WO, real_interaction.WO)

    return {'HW': hw_jsd, 'HO': ho_jsd, 'WO': wo_jsd, 'mean': (hw_jsd + ho_jsd + wo_jsd) / 3}


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

    # Compute interaction statistics with higher top_k for better coverage
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

    # Compute metrics (standard top_k based)
    metrics = calculator.compute_all_metrics(
        generated_spatial, constraints.spatial,
        generated_interaction, constraints.interaction
    )

    # Also compute JSD on support set for comparison
    support_jsd = compute_support_jsd(generated_interaction, constraints.interaction)
    print(f"\nJSD on Support Set (real constraint positions):")
    print(f"  HW: {support_jsd['HW']:.4f}")
    print(f"  HO: {support_jsd['HO']:.4f}")
    print(f"  WO: {support_jsd['WO']:.4f}")
    print(f"  Mean: {support_jsd['mean']:.4f}")

    metrics['jsd_support_mean'] = support_jsd['mean']

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
            max_iter=150,
            gpu_batch_size=min(n_users, 500),
            sddmm_batch_size=100,
            lr_alpha=0.15,                      # Higher for faster spatial
            lr_beta=0.02,                       # Lower to not destabilize spatial
            mfvi_iter=3,
            temp_init=2.0,
            temp_final=0.1,                     # Much lower for convergence
            gumbel_scale=0.2,                   # Lower noise
            gumbel_decay=0.98,                  # Faster decay
            gumbel_final=0.01,                  # Very low final noise
            interaction_freq=3,
            log_freq=10,
            early_stop_patience=20,
            phase_separation=True,
            phase1_ratio=0.4,                   # Longer phase 1 for spatial
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
    print(f"\n{'Method':<20} {'Spatial':<10} {'Interact':<10} {'Support':<10} {'HW nnz':<12} {'Time(s)':<10}")
    print("-" * 80)

    for name, (metrics, interact, elapsed) in results.items():
        spatial = metrics['jsd_mean']
        interact_jsd = metrics.get('jsd_interaction_mean', 0)
        support_jsd = metrics.get('jsd_support_mean', 0)
        nnz = interact.HW.nnz
        print(f"{name:<20} {spatial:<10.4f} {interact_jsd:<10.4f} {support_jsd:<10.4f} {nnz:<12,} {elapsed:<10.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if "IPF-100" in results and "Sparse-100" in results:
        ipf_interact = results["IPF-100"][0]['jsd_interaction_mean']
        ipf_support = results["IPF-100"][0].get('jsd_support_mean', ipf_interact)
        sparse_interact = results["Sparse-100"][0]['jsd_interaction_mean']
        sparse_support = results["Sparse-100"][0].get('jsd_support_mean', sparse_interact)

        print(f"\nIPF vs Sparse (100 users):")
        print(f"  IPF:    Interact={ipf_interact:.4f}, Support={ipf_support:.4f}")
        print(f"  Sparse: Interact={sparse_interact:.4f}, Support={sparse_support:.4f}")

        # Compare on support set (more meaningful)
        if sparse_support < ipf_support:
            improvement = (ipf_support - sparse_support) / ipf_support * 100
            print(f"  >>> On Support Set: Sparse improves by {improvement:.1f}%!")
        else:
            diff = (sparse_support - ipf_support) / ipf_support * 100
            print(f"  >>> On Support Set: IPF is {diff:.1f}% better")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print(f"\nTotal execution time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
