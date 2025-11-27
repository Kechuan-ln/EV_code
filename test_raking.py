#!/usr/bin/env python3
"""Test SS-DMFO 5.0 Raking and Decomposition optimizer

Key innovations:
1. Phase 1: Macro Sinkhorn finds optimal aggregate π* that satisfies BOTH constraints
2. Phase 2: Micro decomposition assigns users while respecting π*

Expected results:
- Spatial JSD ≈ 0 (Sinkhorn guarantees marginal matching)
- Interaction JSD = theoretical minimum (optimal given constraints)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_raking import SSDMFO_RD, RakingConfig
from ssdmfo.evaluation.metrics import MetricsCalculator


def compute_jsd(p, q):
    """Compute Jensen-Shannon Divergence"""
    p = p.flatten() + 1e-10
    q = q.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def compute_support_jsd(generated, real):
    """Compute JSD on support set only"""
    real_coo = real.tocoo()
    if real_coo.nnz == 0:
        return 0.0
    real_vals = real_coo.data + 1e-10
    gen_vals = np.array(generated[real_coo.row, real_coo.col]).flatten() + 1e-10
    real_vals = real_vals / real_vals.sum()
    gen_vals = gen_vals / gen_vals.sum()
    m = 0.5 * (real_vals + gen_vals)
    return float(0.5 * (np.sum(real_vals * np.log(real_vals / m)) +
                       np.sum(gen_vals * np.log(gen_vals / m))))


def evaluate(result, constraints, user_patterns, top_k=50):
    """Full evaluation of results"""
    # Spatial stats
    generated_spatial = result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    generated_spatial.normalize()

    jsd_H = compute_jsd(generated_spatial.H, constraints.spatial.H)
    jsd_W = compute_jsd(generated_spatial.W, constraints.spatial.W)
    jsd_O = compute_jsd(generated_spatial.O, constraints.spatial.O)
    spatial_mean = (jsd_H + jsd_W + jsd_O) / 3

    # Interaction stats
    generated_interaction = result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w, top_k=top_k
    )
    generated_interaction.normalize()

    hw_jsd = compute_support_jsd(generated_interaction.HW, constraints.interaction.HW)
    ho_jsd = compute_support_jsd(generated_interaction.HO, constraints.interaction.HO)
    wo_jsd = compute_support_jsd(generated_interaction.WO, constraints.interaction.WO)
    interact_mean = (hw_jsd + ho_jsd + wo_jsd) / 3

    return {
        'spatial_H': jsd_H,
        'spatial_W': jsd_W,
        'spatial_O': jsd_O,
        'spatial_mean': spatial_mean,
        'interact_HW': hw_jsd,
        'interact_HO': ho_jsd,
        'interact_WO': wo_jsd,
        'interact_mean': interact_mean,
    }


def main():
    print("=" * 70)
    print("SS-DMFO 5.0: Raking and Decomposition TEST")
    print("=" * 70)
    print("\nKey innovation: Two-phase approach")
    print("  Phase 1: Macro Sinkhorn → Optimal aggregate π*")
    print("  Phase 2: Micro Decomposition → Per-user allocations")
    print("\nTheoretical guarantee:")
    print("  - Spatial JSD → 0 (marginals exactly matched)")
    print("  - Interaction JSD → minimum (optimal given constraints)")

    # Load data
    print("\n[Loading data...]")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    print(f"\nGrid: {constraints.grid_h} x {constraints.grid_w}")
    print(f"Interaction support:")
    print(f"  HW: {constraints.interaction.HW.nnz:,}")
    print(f"  HO: {constraints.interaction.HO.nnz:,}")
    print(f"  WO: {constraints.interaction.WO.nnz:,}")

    # Test with different user counts
    for n_users in [100, 1000]:
        print(f"\n{'#' * 70}")
        print(f"# Testing with {n_users} users")
        print('#' * 70)

        print(f"\n[Loading {n_users} users...]")
        user_patterns = loader.load_user_patterns(n_users=n_users)
        print(f"Loaded {len(user_patterns)} users")

        results = {}

        # IPF Baseline
        if n_users == 100:
            print("\n" + "=" * 60)
            print("BASELINE: IPF")
            print("=" * 60)
            start = time.time()
            ipf = IterativeProportionalFitting(max_iter=20)
            ipf_result = ipf.run(constraints, user_patterns)
            ipf_time = time.time() - start
            ipf_metrics = evaluate(ipf_result, constraints, user_patterns)
            print(f"  Spatial JSD: {ipf_metrics['spatial_mean']:.4f}")
            print(f"  Interact JSD: {ipf_metrics['interact_mean']:.4f}")
            print(f"  Time: {ipf_time:.1f}s")
            results['IPF'] = (ipf_metrics, ipf_time)

        # SS-DMFO 5.0 with ICS
        print("\n" + "=" * 60)
        print(f"SS-DMFO 5.0 (ICS) - {n_users} users")
        print("=" * 60)

        config_ics = RakingConfig(
            decomposition_method='ics',
            ics_temp=0.5,
            ics_sort_by_variance=True,
            sinkhorn_max_iter=100,
            sinkhorn_tol=1e-6,
            verbose=True,
            log_freq=20,
        )

        start = time.time()
        rd_ics = SSDMFO_RD(config_ics)
        result_ics = rd_ics.run(constraints, user_patterns)
        ics_time = time.time() - start

        ics_metrics = evaluate(result_ics, constraints, user_patterns)
        print(f"\n[Results - ICS]")
        print(f"  Spatial JSD:  H={ics_metrics['spatial_H']:.4f}, W={ics_metrics['spatial_W']:.4f}, O={ics_metrics['spatial_O']:.4f}")
        print(f"  Spatial Mean: {ics_metrics['spatial_mean']:.4f}")
        print(f"  Interact JSD: HW={ics_metrics['interact_HW']:.4f}, HO={ics_metrics['interact_HO']:.4f}, WO={ics_metrics['interact_WO']:.4f}")
        print(f"  Interact Mean: {ics_metrics['interact_mean']:.4f}")
        print(f"  Total Time: {ics_time:.1f}s")
        results[f'RD-ICS-{n_users}'] = (ics_metrics, ics_time)

        # SS-DMFO 5.0 with IPA (only for 100 users due to memory)
        if n_users <= 100:
            print("\n" + "=" * 60)
            print(f"SS-DMFO 5.0 (IPA) - {n_users} users")
            print("=" * 60)

            config_ipa = RakingConfig(
                decomposition_method='ipa',
                ipa_temp=1.0,
                ipa_max_iter=50,
                sinkhorn_max_iter=100,
                sinkhorn_tol=1e-6,
                verbose=True,
                log_freq=20,
            )

            start = time.time()
            rd_ipa = SSDMFO_RD(config_ipa)
            result_ipa = rd_ipa.run(constraints, user_patterns)
            ipa_time = time.time() - start

            ipa_metrics = evaluate(result_ipa, constraints, user_patterns)
            print(f"\n[Results - IPA]")
            print(f"  Spatial JSD:  H={ipa_metrics['spatial_H']:.4f}, W={ipa_metrics['spatial_W']:.4f}, O={ipa_metrics['spatial_O']:.4f}")
            print(f"  Spatial Mean: {ipa_metrics['spatial_mean']:.4f}")
            print(f"  Interact JSD: HW={ipa_metrics['interact_HW']:.4f}, HO={ipa_metrics['interact_HO']:.4f}, WO={ipa_metrics['interact_WO']:.4f}")
            print(f"  Interact Mean: {ipa_metrics['interact_mean']:.4f}")
            print(f"  Total Time: {ipa_time:.1f}s")
            results[f'RD-IPA-{n_users}'] = (ipa_metrics, ipa_time)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Spatial':<12} {'Interact':<12} {'Time(s)':<10}")
    print("-" * 60)

    for name, (metrics, elapsed) in sorted(results.items()):
        print(f"{name:<20} {metrics['spatial_mean']:<12.4f} {metrics['interact_mean']:<12.4f} {elapsed:<10.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if 'IPF' in results and 'RD-ICS-100' in results:
        ipf_s, ipf_i = results['IPF'][0]['spatial_mean'], results['IPF'][0]['interact_mean']
        ics_s, ics_i = results['RD-ICS-100'][0]['spatial_mean'], results['RD-ICS-100'][0]['interact_mean']

        print(f"\nIPF baseline:")
        print(f"  Spatial: {ipf_s:.4f}, Interact: {ipf_i:.4f}")
        print(f"\nSS-DMFO 5.0 (ICS):")
        print(f"  Spatial: {ics_s:.4f}, Interact: {ics_i:.4f}")

        if ics_i < ipf_i:
            improvement = (ipf_i - ics_i) / ipf_i * 100
            print(f"\n>>> Interaction JSD improved by {improvement:.1f}%!")
        else:
            print(f"\n>>> Interaction JSD not improved (or degraded)")

        if ics_s < 0.01:
            print(f">>> Spatial JSD near-perfect!")
        elif ics_s < ipf_s * 1.1:
            print(f">>> Spatial JSD acceptable")
        else:
            print(f">>> WARNING: Spatial JSD degraded significantly")


if __name__ == '__main__':
    main()
