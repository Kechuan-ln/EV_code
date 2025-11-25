#!/usr/bin/env python3
"""Test SS-DMFO 3.0 - Fixed Version

Key improvements tested:
1. Individual user responses (no template trap)
2. Gumbel noise for diversity
3. Iterative MFVI with beta coupling
4. Temperature annealing
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.random import RandomBaseline
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_v3 import SSDMFOv3, SSDMFO3Config, create_ssdmfo_v3
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

    # Check diversity: number of non-zero entries
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
    print(f"Method total time: {time.time() - method_start:.1f}s")

    return metrics, generated_interaction


def main():
    """Main test function"""
    print("=" * 70)
    print("SS-DMFO 3.0 TEST - Fixed Version")
    print("=" * 70)
    print("\nKey fixes:")
    print("  1. Individual user responses (no template trap)")
    print("  2. Gumbel noise for diversity")
    print("  3. Iterative MFVI with beta coupling")
    print("  4. Temperature annealing")

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

    # Load users
    print("\n[Step 2] Loading user patterns...")
    n_users = 100  # Start with 100 for testing
    user_patterns = loader.load_user_patterns(n_users=n_users)
    print(f"Loaded {len(user_patterns)} users")

    # Test methods
    calculator = MetricsCalculator()
    results = {}
    interactions = {}

    # 1. IPF Phase 1 (baseline for comparison)
    print("\n" + "=" * 70)
    print("BASELINE: IPF Phase 1")
    print("=" * 70)
    results["IPF-P1"], interactions["IPF-P1"] = test_method(
        "IPF Phase 1", IterativeProportionalFitting(max_iter=20),
        constraints, user_patterns, calculator
    )

    # 2. SS-DMFO 3.0 (fast preset for initial testing)
    print("\n" + "=" * 70)
    print("NEW: SS-DMFO 3.0 (Fast Preset)")
    print("=" * 70)

    config_fast = SSDMFO3Config(
        max_iter=30,
        batch_size=50,
        mfvi_iter=3,
        temp_init=2.0,
        temp_final=0.5,
        gumbel_scale=0.2,
        gumbel_decay=0.95,
        interaction_freq=5,
        top_k=50,
        log_freq=5
    )
    results["SSDMFO-v3-fast"], interactions["SSDMFO-v3-fast"] = test_method(
        "SS-DMFO 3.0 (Fast)",
        SSDMFOv3(config_fast),
        constraints, user_patterns, calculator
    )

    # 3. SS-DMFO 3.0 (default preset)
    print("\n" + "=" * 70)
    print("NEW: SS-DMFO 3.0 (Default Preset)")
    print("=" * 70)

    config_default = SSDMFO3Config(
        max_iter=50,
        batch_size=50,
        mfvi_iter=5,
        temp_init=2.0,
        temp_final=0.3,
        gumbel_scale=0.3,
        gumbel_decay=0.97,
        interaction_freq=5,
        top_k=50,
        log_freq=10
    )
    results["SSDMFO-v3"], interactions["SSDMFO-v3"] = test_method(
        "SS-DMFO 3.0 (Default)",
        SSDMFOv3(config_default),
        constraints, user_patterns, calculator
    )

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY - SS-DMFO 3.0 vs Baseline")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Spatial':<10} {'Interact':<10} {'Total':<10} {'HW nnz':<10}")
    print("-" * 60)

    for name in ["IPF-P1", "SSDMFO-v3-fast", "SSDMFO-v3"]:
        metrics = results[name]
        interact_nnz = interactions[name].HW.nnz
        spatial = metrics['jsd_mean']
        interact = metrics.get('jsd_interaction_mean', 0)
        total = metrics['jsd_total_mean']
        print(f"{name:<20} {spatial:<10.4f} {interact:<10.4f} {total:<10.4f} {interact_nnz:<10}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    ipf_interact = results["IPF-P1"]['jsd_interaction_mean']
    v3_interact = results["SSDMFO-v3"]['jsd_interaction_mean']

    print(f"\nIPF Phase 1:")
    print(f"  Spatial JSD:     {results['IPF-P1']['jsd_mean']:.4f}")
    print(f"  Interaction JSD: {ipf_interact:.4f}")
    print(f"  HW entries:      {interactions['IPF-P1'].HW.nnz}")

    print(f"\nSS-DMFO 3.0:")
    print(f"  Spatial JSD:     {results['SSDMFO-v3']['jsd_mean']:.4f}")
    print(f"  Interaction JSD: {v3_interact:.4f}")
    print(f"  HW entries:      {interactions['SSDMFO-v3'].HW.nnz}")

    if v3_interact < ipf_interact:
        improvement = (ipf_interact - v3_interact) / ipf_interact * 100
        print(f"\n>>> SS-DMFO 3.0 improves interaction by {improvement:.1f}%!")
    else:
        print(f"\n>>> SS-DMFO 3.0 needs further tuning.")

    # Check diversity improvement
    ipf_nnz = interactions['IPF-P1'].HW.nnz
    v3_nnz = interactions['SSDMFO-v3'].HW.nnz
    if v3_nnz > ipf_nnz:
        print(f">>> Diversity improved: {ipf_nnz} -> {v3_nnz} HW entries ({v3_nnz/ipf_nnz:.1f}x)")
    else:
        print(f">>> Diversity not improved: {ipf_nnz} -> {v3_nnz} HW entries")

    print(f"\nTotal execution time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
