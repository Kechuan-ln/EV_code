#!/usr/bin/env python3
"""Check consistency between spatial marginals and interaction matrices"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader


def compute_jsd(p, q):
    """Compute Jensen-Shannon Divergence"""
    p = p.flatten() + 1e-10
    q = q.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def main():
    print("=" * 70)
    print("Constraint Consistency Check")
    print("=" * 70)

    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    # Get spatial distributions
    H = constraints.spatial.H.flatten()
    W = constraints.spatial.W.flatten()
    O = constraints.spatial.O.flatten()

    # Normalize
    H = H / H.sum()
    W = W / W.sum()
    O = O / O.sum()

    print(f"\nSpatial distributions:")
    print(f"  H: sum={H.sum():.6f}, nnz={np.sum(H > 0)}")
    print(f"  W: sum={W.sum():.6f}, nnz={np.sum(W > 0)}")
    print(f"  O: sum={O.sum():.6f}, nnz={np.sum(O > 0)}")

    # Check HW interaction
    print(f"\n--- HW Interaction ---")
    HW = constraints.interaction.HW
    HW_dense = HW.toarray()
    HW_dense = HW_dense / (HW_dense.sum() + 1e-10)

    # Row marginal (should match H)
    HW_row_sum = HW_dense.sum(axis=1)
    HW_col_sum = HW_dense.sum(axis=0)

    jsd_row = compute_jsd(HW_row_sum, H)
    jsd_col = compute_jsd(HW_col_sum, W)

    print(f"  HW row marginal vs H: JSD = {jsd_row:.6f}")
    print(f"  HW col marginal vs W: JSD = {jsd_col:.6f}")
    print(f"  Row sum: {HW_row_sum.sum():.6f}, H sum: {H.sum():.6f}")
    print(f"  Col sum: {HW_col_sum.sum():.6f}, W sum: {W.sum():.6f}")

    # Check HO interaction
    print(f"\n--- HO Interaction ---")
    HO = constraints.interaction.HO
    HO_dense = HO.toarray()
    HO_dense = HO_dense / (HO_dense.sum() + 1e-10)

    HO_row_sum = HO_dense.sum(axis=1)
    HO_col_sum = HO_dense.sum(axis=0)

    jsd_row = compute_jsd(HO_row_sum, H)
    jsd_col = compute_jsd(HO_col_sum, O)

    print(f"  HO row marginal vs H: JSD = {jsd_row:.6f}")
    print(f"  HO col marginal vs O: JSD = {jsd_col:.6f}")

    # Check WO interaction
    print(f"\n--- WO Interaction ---")
    WO = constraints.interaction.WO
    WO_dense = WO.toarray()
    WO_dense = WO_dense / (WO_dense.sum() + 1e-10)

    WO_row_sum = WO_dense.sum(axis=1)
    WO_col_sum = WO_dense.sum(axis=0)

    jsd_row = compute_jsd(WO_row_sum, W)
    jsd_col = compute_jsd(WO_col_sum, O)

    print(f"  WO row marginal vs W: JSD = {jsd_row:.6f}")
    print(f"  WO col marginal vs O: JSD = {jsd_col:.6f}")

    # Analysis
    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    if jsd_row > 0.01 or jsd_col > 0.01:
        print("\n⚠️  CONSTRAINT INCONSISTENCY DETECTED!")
        print("The interaction matrices are NOT consistent with spatial marginals.")
        print("This means there is NO joint distribution that can satisfy both constraints.")
        print("\nImplication:")
        print("  - Sinkhorn cannot converge (finds best compromise)")
        print("  - Perfect spatial + perfect interaction is IMPOSSIBLE")
        print("  - We must trade off between spatial and interaction matching")
    else:
        print("\n✓ Constraints are consistent")
        print("A joint distribution satisfying both exists.")


if __name__ == '__main__':
    main()
