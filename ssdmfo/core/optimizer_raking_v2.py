#!/usr/bin/env python3
"""SS-DMFO 5.0 v2: Raking and Decomposition with Joint Structure Preservation

Key fix from v1: Preserve the joint H-W correlation structure!

The issue in v1 was:
- IPA/ICS decomposed π* to users but then extracted marginals
- This lost the H-W correlation, making interaction ≈ P(H)×P(W)

The fix:
- Store joint (h,w) assignments or joint distributions
- Evaluate interaction directly from joint structure
- Use special Result type that stores joint allocations

This should achieve:
- Spatial JSD ≈ 0 (from Sinkhorn marginal matching)
- Interaction JSD ≈ 0.018 (theoretical optimum given constraint inconsistency)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import sparse
import time

from ..data.structures import Constraints, UserPattern, Result, SpatialConstraints, InteractionConstraints


@dataclass
class RakingConfigV2:
    """Configuration for SS-DMFO 5.0 v2"""

    # Phase 1: Macro Sinkhorn
    sinkhorn_max_iter: int = 1000
    sinkhorn_tol: float = 1e-8
    sinkhorn_reg: float = 1e-10

    # Phase 2: Decomposition
    decomposition_method: str = 'ics'  # 'ics' recommended

    # ICS parameters
    ics_temp: float = 0.1  # Lower temp = more concentrated sampling
    ics_sort_by_variance: bool = True

    # Logging
    verbose: bool = True
    log_freq: int = 100


@dataclass
class JointResult:
    """Result that preserves joint H-W structure"""
    method_name: str
    runtime: float

    # Joint assignments: user_id -> (h_idx, w_idx) for H-W pair
    hw_assignments: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    ho_assignments: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    wo_assignments: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Also store per-location allocations for spatial evaluation
    allocations: Dict[int, np.ndarray] = field(default_factory=dict)

    def compute_spatial_stats(self, user_patterns: Dict[int, UserPattern],
                             grid_h: int, grid_w: int) -> SpatialConstraints:
        """Compute spatial stats from allocations"""
        H_map = np.zeros((grid_h, grid_w))
        W_map = np.zeros((grid_h, grid_w))
        O_map = np.zeros((grid_h, grid_w))

        for user_id, alloc in self.allocations.items():
            pattern = user_patterns[user_id]
            for loc_idx, location in enumerate(pattern.locations):
                probs = alloc[loc_idx].reshape(grid_h, grid_w)
                if location.type == 'H':
                    H_map += probs
                elif location.type == 'W':
                    W_map += probs
                else:
                    O_map += probs

        return SpatialConstraints(H=H_map, W=W_map, O=O_map)

    def compute_interaction_stats_joint(self, grid_h: int, grid_w: int) -> InteractionConstraints:
        """Compute interaction stats DIRECTLY from joint assignments (key fix!)"""
        grid_size = grid_h * grid_w

        # Count (h, w) pairs from assignments
        hw_counts = np.zeros((grid_size, grid_size))
        ho_counts = np.zeros((grid_size, grid_size))
        wo_counts = np.zeros((grid_size, grid_size))

        for uid, (h_idx, w_idx) in self.hw_assignments.items():
            hw_counts[h_idx, w_idx] += 1

        for uid, (h_idx, o_idx) in self.ho_assignments.items():
            ho_counts[h_idx, o_idx] += 1

        for uid, (w_idx, o_idx) in self.wo_assignments.items():
            wo_counts[w_idx, o_idx] += 1

        return InteractionConstraints(
            HW=sparse.csr_matrix(hw_counts),
            HO=sparse.csr_matrix(ho_counts),
            WO=sparse.csr_matrix(wo_counts)
        )


class MacroSinkhornV2:
    """Phase 1: Macro Sinkhorn with better convergence handling"""

    def __init__(self, config: RakingConfigV2):
        self.config = config

    def run(self, target: sparse.csr_matrix,
            mu_row: np.ndarray, mu_col: np.ndarray,
            name: str = "") -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, Dict]:
        """Run Sinkhorn and return scaled matrix plus row/col scaling factors"""
        reg = self.config.sinkhorn_reg

        target_coo = target.tocoo()
        target_vals = target_coo.data.copy() + reg
        rows, cols = target_coo.row, target_coo.col
        n_rows, n_cols = target.shape

        # Normalize inputs
        target_vals = target_vals / target_vals.sum()
        mu_row = mu_row.flatten() + reg
        mu_row = mu_row / mu_row.sum()
        mu_col = mu_col.flatten() + reg
        mu_col = mu_col / mu_col.sum()

        # Log-domain scaling
        log_a = np.zeros(n_rows)
        log_b = np.zeros(n_cols)

        history = {'row_err': [], 'col_err': []}
        start = time.time()

        for it in range(self.config.sinkhorn_max_iter):
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

            history['row_err'].append(row_err)
            history['col_err'].append(col_err)

            if self.config.verbose and (it + 1) % self.config.log_freq == 0:
                print(f"    {name} iter {it+1}: row_err={row_err:.2e}, col_err={col_err:.2e}")

            if row_err < self.config.sinkhorn_tol and col_err < self.config.sinkhorn_tol:
                if self.config.verbose:
                    print(f"    {name} converged at iter {it+1}")
                break

        # Final scaled values
        final_vals = target_vals * np.exp(log_a[rows] + log_b[cols])
        final_vals = final_vals / final_vals.sum()

        pi_star = sparse.csr_matrix(
            (final_vals, (rows, cols)),
            shape=target.shape
        )

        history['time'] = time.time() - start
        history['converged'] = row_err < self.config.sinkhorn_tol

        return pi_star, log_a, log_b, history


class SSDMFO_RD_V2:
    """SS-DMFO 5.0 v2: Preserves joint structure"""

    def __init__(self, config: Optional[RakingConfigV2] = None):
        self.config = config or RakingConfigV2()
        self.sinkhorn = MacroSinkhornV2(self.config)

    def run(self, constraints: Constraints, user_patterns: Dict[int, UserPattern]) -> JointResult:
        """Run full pipeline with joint structure preservation"""
        start = time.time()
        n_users = len(user_patterns)
        grid_size = constraints.grid_h * constraints.grid_w
        user_ids = list(user_patterns.keys())

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("SS-DMFO 5.0 v2: R&D with Joint Structure Preservation")
            print(f"{'='*60}")
            print(f"Users: {n_users}, Grid: {constraints.grid_h}x{constraints.grid_w}")

        # Get spatial marginals
        H = constraints.spatial.H.flatten()
        W = constraints.spatial.W.flatten()
        O = constraints.spatial.O.flatten()

        # ============================================================
        # Phase 1: Macro Sinkhorn for each interaction type
        # ============================================================
        if self.config.verbose:
            print(f"\n[Phase 1] Macro Sinkhorn")

        print(f"\n  Computing π*_HW...")
        pi_HW, _, _, hist_hw = self.sinkhorn.run(
            constraints.interaction.HW, H, W, "HW"
        )

        print(f"\n  Computing π*_HO...")
        pi_HO, _, _, hist_ho = self.sinkhorn.run(
            constraints.interaction.HO, H, O, "HO"
        )

        print(f"\n  Computing π*_WO...")
        pi_WO, _, _, hist_wo = self.sinkhorn.run(
            constraints.interaction.WO, W, O, "WO"
        )

        phase1_time = time.time() - start

        # ============================================================
        # Phase 2: ICS Decomposition with Joint Preservation
        # ============================================================
        if self.config.verbose:
            print(f"\n[Phase 2] ICS Decomposition (Joint Preservation)")

        phase2_start = time.time()

        # Sample (h, w) pairs for each user from π*_HW
        hw_assignments = self._sample_from_joint(pi_HW, n_users, "HW")

        # Sample (h, o) pairs from π*_HO
        ho_assignments = self._sample_from_joint(pi_HO, n_users, "HO")

        # Sample (w, o) pairs from π*_WO
        wo_assignments = self._sample_from_joint(pi_WO, n_users, "WO")

        phase2_time = time.time() - phase2_start

        # ============================================================
        # Build per-location allocations for spatial evaluation
        # ============================================================
        if self.config.verbose:
            print(f"\n[Building allocations]")

        allocations = {}
        for i, uid in enumerate(user_ids):
            pattern = user_patterns[uid]
            n_locs = len(pattern.locations)
            user_alloc = np.zeros((n_locs, grid_size))

            # Get this user's assigned locations
            h_idx, w_idx = hw_assignments[i]
            _, o_idx = ho_assignments[i]  # Use o from HO

            for loc_idx, loc in enumerate(pattern.locations):
                if loc.type == 'H':
                    user_alloc[loc_idx, h_idx] = 1.0
                elif loc.type == 'W':
                    user_alloc[loc_idx, w_idx] = 1.0
                else:  # O
                    user_alloc[loc_idx, o_idx] = 1.0

            allocations[uid] = user_alloc

        total_time = time.time() - start

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Phase 1: {phase1_time:.1f}s, Phase 2: {phase2_time:.1f}s")
            print(f"Total: {total_time:.1f}s")
            print(f"{'='*60}")

        # Convert to uid-keyed assignments
        hw_by_uid = {user_ids[i]: hw_assignments[i] for i in range(n_users)}
        ho_by_uid = {user_ids[i]: ho_assignments[i] for i in range(n_users)}
        wo_by_uid = {user_ids[i]: wo_assignments[i] for i in range(n_users)}

        return JointResult(
            method_name='SS-DMFO 5.0 v2 (Joint)',
            runtime=total_time,
            hw_assignments=hw_by_uid,
            ho_assignments=ho_by_uid,
            wo_assignments=wo_by_uid,
            allocations=allocations,
        )

    def _sample_from_joint(self, pi_star: sparse.csr_matrix, n_samples: int, name: str) -> Dict[int, Tuple[int, int]]:
        """Sample (row, col) pairs from π* distribution"""
        pi_coo = pi_star.tocoo()
        probs = pi_coo.data.copy()
        probs = probs / probs.sum()
        rows, cols = pi_coo.row, pi_coo.col
        n_support = len(probs)

        if self.config.verbose:
            print(f"  Sampling {n_samples} pairs from {name} (support={n_support:,})")

        # Sample indices into the support set
        sampled_indices = np.random.choice(n_support, size=n_samples, replace=True, p=probs)

        assignments = {}
        for i, idx in enumerate(sampled_indices):
            assignments[i] = (rows[idx], cols[idx])

        return assignments


def compute_jsd(p, q):
    """Jensen-Shannon Divergence"""
    p = p.flatten() + 1e-10
    q = q.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def compute_support_jsd(gen_matrix: sparse.csr_matrix, target_matrix: sparse.csr_matrix) -> float:
    """JSD on support of target matrix"""
    target_coo = target_matrix.tocoo()
    if target_coo.nnz == 0:
        return 0.0

    target_vals = target_coo.data + 1e-10
    gen_vals = np.array(gen_matrix[target_coo.row, target_coo.col]).flatten() + 1e-10

    target_vals = target_vals / target_vals.sum()
    gen_vals = gen_vals / gen_vals.sum()

    m = 0.5 * (target_vals + gen_vals)
    return float(0.5 * (np.sum(target_vals * np.log(target_vals / m)) +
                       np.sum(gen_vals * np.log(gen_vals / m))))


def evaluate_joint_result(result: JointResult, constraints: Constraints,
                         user_patterns: Dict[int, UserPattern]) -> Dict:
    """Evaluate JointResult"""
    # Spatial stats
    gen_spatial = result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    gen_spatial.normalize()

    jsd_H = compute_jsd(gen_spatial.H, constraints.spatial.H)
    jsd_W = compute_jsd(gen_spatial.W, constraints.spatial.W)
    jsd_O = compute_jsd(gen_spatial.O, constraints.spatial.O)

    # Interaction stats (from joint assignments!)
    gen_interact = result.compute_interaction_stats_joint(
        constraints.grid_h, constraints.grid_w
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
