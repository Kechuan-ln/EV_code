#!/usr/bin/env python3
"""Test SS-DMFO 4.0 G-IPF Optimizer

Key innovations over SS-DMFO 3.0:
1. Log-domain multiplicative updates (Sinkhorn-style)
2. Alternating projections instead of simultaneous gradient descent
3. No learning rate tuning - damping parameter only
4. Mathematical convergence guarantees

Algorithm:
    α = α + T * (log μ_real - log μ_gen)  [Spatial projection]
    β = β + T * (log π_real - log π_gen)  [Interaction projection]
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_gipf import SSDMFOGIPF, GIPFConfig
from ssdmfo.evaluation.metrics import MetricsCalculator


def compute_support_jsd(generated_interaction, real_interaction):
    """Compute JSD on the support set"""
    def sparse_jsd_on_support(gen, real):
        real_coo = real.tocoo()
        if real_coo.nnz == 0:
            return 0.0

        real_vals = real_coo.data + 1e-10
        gen_vals = np.array(gen[real_coo.row, real_coo.col]).flatten() + 1e-10

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

    # Support set JSD
    support_jsd = compute_support_jsd(generated_interaction, constraints.interaction)
    print(f"\nJSD on Support Set:")
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
    print("SS-DMFO 4.0 G-IPF OPTIMIZER TEST")
    print("=" * 70)
    print("\nKey innovations:")
    print("  1. Log-domain multiplicative updates (Sinkhorn-style)")
    print("  2. Alternating projections (spatial then interaction)")
    print("  3. No learning rate - damping parameter only")
    print("  4. Gauss-Seidel style updates for better convergence")

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

    calculator = MetricsCalculator()
    results = {}

    # Test with increasing user counts
    for n_users in [100, 1000]:
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

        # G-IPF SS-DMFO 4.0
        print("\n" + "=" * 70)
        print(f"G-IPF: SS-DMFO 4.0 ({n_users} users)")
        print("=" * 70)

        config = GIPFConfig(
            max_iter=200,
            gpu_batch_size=min(n_users, 500),
            sddmm_batch_size=100,
            # G-IPF specific - CONSERVATIVE damping is critical!
            alpha_damping=0.1,           # Very conservative for spatial
            beta_damping=0.02,           # Even more conservative for interaction
            # Temperature - lower is more stable
            temp_anneal=True,
            temp_init=1.0,               # Lower initial temp
            temp_final=0.3,
            # MFVI - fewer iterations to reduce instability
            mfvi_iter=2,
            mfvi_damping=0.3,
            # Gumbel - lower noise
            gumbel_scale=0.05,
            gumbel_decay=0.99,
            gumbel_final=0.01,
            # Schedule
            spatial_first_iters=50,      # Longer pure spatial phase
            interaction_freq=3,          # Less frequent interaction updates
            gauss_seidel=False,          # Disable for stability
            freeze_alpha_in_phase2=True, # Freeze alpha when optimizing interaction
            use_beta_in_final=True,      # Use optimized beta in final allocation
            # Logging
            log_freq=10,
            early_stop_patience=40,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        gipf_method = SSDMFOGIPF(config)
        metrics_gipf, interact_gipf, time_gipf = test_method(
            f"SS-DMFO 4.0 G-IPF ({n_users}u)",
            gipf_method,
            constraints, user_patterns, calculator
        )
        results[f"GIPF-{n_users}"] = (metrics_gipf, interact_gipf, time_gipf)

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

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: G-IPF vs Previous Methods")
    print("=" * 70)

    if "IPF-100" in results and "GIPF-100" in results:
        ipf_spatial = results["IPF-100"][0]['jsd_mean']
        ipf_interact = results["IPF-100"][0].get('jsd_interaction_mean', 0)
        gipf_spatial = results["GIPF-100"][0]['jsd_mean']
        gipf_interact = results["GIPF-100"][0].get('jsd_interaction_mean', 0)

        print(f"\nIPF vs G-IPF (100 users):")
        print(f"  IPF:    Spatial={ipf_spatial:.4f}, Interact={ipf_interact:.4f}")
        print(f"  G-IPF:  Spatial={gipf_spatial:.4f}, Interact={gipf_interact:.4f}")

        if gipf_spatial < 0.05:
            print(f"  >>> G-IPF achieves near-perfect spatial matching!")
        if gipf_interact < ipf_interact:
            improvement = (ipf_interact - gipf_interact) / ipf_interact * 100
            print(f"  >>> G-IPF improves interaction by {improvement:.1f}%")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print(f"\nTotal execution time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
