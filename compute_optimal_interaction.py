#!/usr/bin/env python3
"""Compute the theoretical optimal interaction JSD using Sinkhorn-scaled π*

Since constraints are inconsistent, we cannot achieve both:
- Spatial JSD = 0
- Interaction JSD = 0

This script computes what the BEST POSSIBLE interaction JSD is when we
enforce spatial constraints exactly via Sinkhorn scaling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import sparse
from ssdmfo.data.loader import ConstraintDataLoader


def compute_jsd(p, q):
    """Compute Jensen-Shannon Divergence"""
    p = p.flatten() + 1e-10
    q = q.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def sinkhorn_scale(target_coo, mu_row, mu_col, max_iter=1000, tol=1e-10):
    """Scale sparse matrix to match marginals using Sinkhorn"""
    reg = 1e-10

    target_vals = target_coo.data.copy() + reg
    rows, cols = target_coo.row, target_coo.col
    n_rows = target_coo.shape[0]
    n_cols = target_coo.shape[1]

    # Log-domain scaling factors
    log_a = np.zeros(n_rows)
    log_b = np.zeros(n_cols)

    mu_row = mu_row + reg
    mu_row = mu_row / mu_row.sum()
    mu_col = mu_col + reg
    mu_col = mu_col / mu_col.sum()

    for it in range(max_iter):
        # Row scaling
        scaled = target_vals * np.exp(log_a[rows] + log_b[cols])
        row_sums = np.zeros(n_rows)
        np.add.at(row_sums, rows, scaled)
        row_sums = np.maximum(row_sums, reg)
        log_a = log_a + np.log(mu_row / row_sums)

        # Column scaling
        scaled = target_vals * np.exp(log_a[rows] + log_b[cols])
        col_sums = np.zeros(n_cols)
        np.add.at(col_sums, cols, scaled)
        col_sums = np.maximum(col_sums, reg)
        log_b = log_b + np.log(mu_col / col_sums)

        row_err = np.abs(row_sums - mu_row).sum()
        col_err = np.abs(col_sums - mu_col).sum()

        if row_err < tol and col_err < tol:
            print(f"    Converged at iter {it+1}")
            break

    final_vals = target_vals * np.exp(log_a[rows] + log_b[cols])
    final_vals = final_vals / final_vals.sum()

    return final_vals, row_err, col_err


def compute_support_jsd(gen_vals, target_vals):
    """JSD between two distributions on same support"""
    p = gen_vals + 1e-10
    q = target_vals + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def main():
    print("=" * 70)
    print("Theoretical Optimal Interaction JSD")
    print("=" * 70)
    print("\nThis computes the BEST POSSIBLE interaction JSD when spatial")
    print("constraints are satisfied exactly via Sinkhorn scaling.")

    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    H = constraints.spatial.H.flatten()
    W = constraints.spatial.W.flatten()
    O = constraints.spatial.O.flatten()

    results = {}

    # HW
    print(f"\n--- HW Interaction ---")
    HW_coo = constraints.interaction.HW.tocoo()
    target_HW = HW_coo.data / HW_coo.data.sum()

    print("  Running Sinkhorn to match H and W marginals...")
    scaled_HW, row_err, col_err = sinkhorn_scale(HW_coo, H, W)
    print(f"    Final errors: row={row_err:.2e}, col={col_err:.2e}")

    jsd_HW = compute_support_jsd(scaled_HW, target_HW)
    print(f"  JSD(π*_HW || π_target_HW) = {jsd_HW:.6f}")
    results['HW'] = jsd_HW

    # HO
    print(f"\n--- HO Interaction ---")
    HO_coo = constraints.interaction.HO.tocoo()
    target_HO = HO_coo.data / HO_coo.data.sum()

    print("  Running Sinkhorn to match H and O marginals...")
    scaled_HO, row_err, col_err = sinkhorn_scale(HO_coo, H, O)
    print(f"    Final errors: row={row_err:.2e}, col={col_err:.2e}")

    jsd_HO = compute_support_jsd(scaled_HO, target_HO)
    print(f"  JSD(π*_HO || π_target_HO) = {jsd_HO:.6f}")
    results['HO'] = jsd_HO

    # WO
    print(f"\n--- WO Interaction ---")
    WO_coo = constraints.interaction.WO.tocoo()
    target_WO = WO_coo.data / WO_coo.data.sum()

    print("  Running Sinkhorn to match W and O marginals...")
    scaled_WO, row_err, col_err = sinkhorn_scale(WO_coo, W, O)
    print(f"    Final errors: row={row_err:.2e}, col={col_err:.2e}")

    jsd_WO = compute_support_jsd(scaled_WO, target_WO)
    print(f"  JSD(π*_WO || π_target_WO) = {jsd_WO:.6f}")
    results['WO'] = jsd_WO

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Theoretical Optimal JSD (with exact spatial matching)")
    print("=" * 70)

    mean_jsd = (results['HW'] + results['HO'] + results['WO']) / 3

    print(f"\n  HW: {results['HW']:.6f}")
    print(f"  HO: {results['HO']:.6f}")
    print(f"  WO: {results['WO']:.6f}")
    print(f"  Mean: {mean_jsd:.6f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)

    baseline_jsd = 0.6348
    print(f"\n  IPF baseline interaction JSD: {baseline_jsd:.4f}")
    print(f"  Theoretical optimal interaction JSD: {mean_jsd:.4f}")

    if mean_jsd < baseline_jsd * 0.95:
        improvement = (baseline_jsd - mean_jsd) / baseline_jsd * 100
        print(f"\n  >>> Room for improvement: {improvement:.1f}%")
        print(f"  >>> SS-DMFO 5.0 R&D can achieve this by properly decomposing π*")
    else:
        print(f"\n  >>> IPF is already near-optimal!")
        print(f"  >>> The constraint inconsistency limits achievable interaction JSD")
        print(f"  >>> No method can significantly beat IPF on interaction")


if __name__ == '__main__':
    main()
