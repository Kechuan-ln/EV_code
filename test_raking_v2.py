#!/usr/bin/env python3
"""Test SS-DMFO 5.0 v2 with Joint Structure Preservation

Key fix: Evaluate interaction directly from joint (h,w) assignments,
not from independent P(H) × P(W).

Expected results:
- Spatial JSD: Should still be good (sampling from π* with correct marginals)
- Interaction JSD: Should approach theoretical optimum (~0.018)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_raking_v2 import (
    SSDMFO_RD_V2, RakingConfigV2, JointResult,
    evaluate_joint_result, compute_jsd, compute_support_jsd
)


def evaluate_ipf(result, constraints, user_patterns):
    """Evaluate IPF result (for comparison)"""
    gen_spatial = result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    gen_spatial.normalize()

    jsd_H = compute_jsd(gen_spatial.H, constraints.spatial.H)
    jsd_W = compute_jsd(gen_spatial.W, constraints.spatial.W)
    jsd_O = compute_jsd(gen_spatial.O, constraints.spatial.O)

    gen_interact = result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w, top_k=50
    )
    gen_interact.normalize()

    jsd_HW = compute_support_jsd(gen_interact.HW, constraints.interaction.HW)
    jsd_HO = compute_support_jsd(gen_interact.HO, constraints.interaction.HO)
    jsd_WO = compute_support_jsd(gen_interact.WO, constraints.interaction.WO)

    return {
        'spatial_H': jsd_H,
        'spatial_W': jsd_W,
        'spatial_O': jsd_O,
        'spatial_mean': (jsd_H + jsd_W + jsd_O) / 3,
        'interact_HW': jsd_HW,
        'interact_HO': jsd_HO,
        'interact_WO': jsd_WO,
        'interact_mean': (jsd_HW + jsd_HO + jsd_WO) / 3,
    }


def main():
    print("=" * 70)
    print("SS-DMFO 5.0 v2: Joint Structure Preservation TEST")
    print("=" * 70)
    print("\nKey fix: Evaluate interaction from joint (h,w) pairs directly")
    print("         Not from independent P(H) × P(W)")
    print("\nTheoretical optimum (given constraint inconsistency):")
    print("  - Interaction JSD ≈ 0.018")

    # Load data
    print("\n[Loading data...]")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    print(f"\nGrid: {constraints.grid_h} x {constraints.grid_w}")

    # Test with different user counts
    for n_users in [100, 1000, 10000]:
        print(f"\n{'#' * 70}")
        print(f"# Testing with {n_users:,} users")
        print('#' * 70)

        user_patterns = loader.load_user_patterns(n_users=n_users)
        print(f"Loaded {len(user_patterns)} users")

        results = {}

        # IPF Baseline
        if n_users == 100:
            print("\n" + "=" * 60)
            print("BASELINE: IPF")
            print("=" * 60)
            import time
            start = time.time()
            ipf = IterativeProportionalFitting(max_iter=20)
            ipf_result = ipf.run(constraints, user_patterns)
            ipf_time = time.time() - start
            ipf_metrics = evaluate_ipf(ipf_result, constraints, user_patterns)
            print(f"  Spatial JSD: {ipf_metrics['spatial_mean']:.4f}")
            print(f"  Interact JSD: {ipf_metrics['interact_mean']:.4f}")
            print(f"  Time: {ipf_time:.1f}s")
            results['IPF'] = (ipf_metrics, ipf_time)

        # SS-DMFO 5.0 v2
        print("\n" + "=" * 60)
        print(f"SS-DMFO 5.0 v2 (Joint) - {n_users:,} users")
        print("=" * 60)

        config = RakingConfigV2(
            sinkhorn_max_iter=1000,
            sinkhorn_tol=1e-8,
            verbose=True,
            log_freq=200,
        )

        import time
        start = time.time()
        rd = SSDMFO_RD_V2(config)
        result = rd.run(constraints, user_patterns)
        rd_time = time.time() - start

        metrics = evaluate_joint_result(result, constraints, user_patterns)

        print(f"\n[Results]")
        print(f"  Spatial JSD:  H={metrics['spatial_H']:.4f}, W={metrics['spatial_W']:.4f}, O={metrics['spatial_O']:.4f}")
        print(f"  Spatial Mean: {metrics['spatial_mean']:.4f}")
        print(f"  Interact JSD: HW={metrics['interact_HW']:.4f}, HO={metrics['interact_HO']:.4f}, WO={metrics['interact_WO']:.4f}")
        print(f"  Interact Mean: {metrics['interact_mean']:.4f}")
        print(f"  Time: {rd_time:.1f}s")

        results[f'RD-v2-{n_users}'] = (metrics, rd_time)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Spatial':<12} {'Interact':<12} {'Time(s)':<10}")
    print("-" * 60)

    for name, (m, t) in sorted(results.items()):
        print(f"{name:<20} {m['spatial_mean']:<12.4f} {m['interact_mean']:<12.4f} {t:<10.1f}")

    print("\n" + "=" * 70)
    print("COMPARISON vs THEORETICAL OPTIMUM")
    print("=" * 70)
    print(f"\n  Theoretical optimal interaction JSD: 0.0185")

    if 'RD-v2-10000' in results:
        actual = results['RD-v2-10000'][0]['interact_mean']
        gap = abs(actual - 0.0185) / 0.0185 * 100
        print(f"  SS-DMFO 5.0 v2 (10k users): {actual:.4f}")
        print(f"  Gap from optimum: {gap:.1f}%")

        if actual < 0.05:
            print(f"\n  >>> SUCCESS! Near-optimal interaction matching achieved!")
        elif actual < 0.1:
            print(f"\n  >>> Good progress! Interaction significantly improved.")
        else:
            print(f"\n  >>> Still needs work - check sampling/evaluation logic.")


if __name__ == '__main__':
    main()
