#!/usr/bin/env python3
"""SS-DMFO 5.0: Raking and Decomposition (R&D) Optimizer

This module implements a fundamentally different approach than SS-DMFO 3.0/4.0:
- Phase 1: Macro-level Matrix Raking (Sinkhorn) to find optimal aggregate distribution
- Phase 2: Micro-level Decomposition to assign individuals while respecting heterogeneity

Key insight: MFVI fails because beta's influence is too weak (~0.001 vs alpha ~1.0).
Instead, we directly optimize the aggregate distribution, then decompose to individuals.

Reference: Expert team feedback on SS-DMFO 4.0 G-IPF failure analysis.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import sparse
import time

from ..data.structures import Constraints, UserPattern, Result


@dataclass
class RakingConfig:
    """Configuration for SS-DMFO 5.0 Raking optimizer"""

    # Phase 1: Macro Sinkhorn
    sinkhorn_max_iter: int = 100
    sinkhorn_tol: float = 1e-6
    sinkhorn_reg: float = 1e-10  # Regularization for numerical stability

    # Phase 2: Decomposition method
    decomposition_method: str = 'ics'  # 'ipa' or 'ics'

    # IPA (Iterative Proportional Allocation) parameters
    ipa_max_iter: int = 50
    ipa_tol: float = 1e-4
    ipa_temp: float = 1.0  # Temperature for cost-based weighting

    # ICS (Iterative Capacity Sampling) parameters
    ics_temp: float = 0.5  # Temperature for sampling
    ics_sort_by_variance: bool = True  # Sort users by cost variance

    # Cost function parameters
    cost_type: str = 'distance'  # 'distance', 'frequency', or 'combined'
    distance_weight: float = 0.5  # Weight for distance in combined cost

    # Logging
    log_freq: int = 10
    verbose: bool = True


class MacroSinkhorn:
    """Phase 1: Macro-level Sinkhorn iteration (Matrix Raking)

    Solves the optimal transport problem:
        min_{π} D_KL(π || π_target)
        s.t. Σ_w π(h,w) = μ_H(h)
             Σ_h π(h,w) = μ_W(w)

    This guarantees:
    - Spatial marginals exactly match target
    - Joint distribution is as close as possible to target interaction
    """

    def __init__(self, config: RakingConfig):
        self.config = config

    def run(self,
            target_interaction: sparse.csr_matrix,
            marginal_row: np.ndarray,
            marginal_col: np.ndarray) -> Tuple[sparse.csr_matrix, Dict]:
        """Run macro Sinkhorn to find optimal aggregate distribution

        Args:
            target_interaction: Sparse matrix of target interaction (e.g., HW)
            marginal_row: Target row marginal (e.g., μ_H)
            marginal_col: Target column marginal (e.g., μ_W)

        Returns:
            Optimal sparse distribution π* and convergence info
        """
        reg = self.config.sinkhorn_reg

        # Convert to COO for efficient element-wise operations
        target_coo = target_interaction.tocoo()

        # Initialize scaling factors
        n_rows = target_interaction.shape[0]
        n_cols = target_interaction.shape[1]

        # Row and column scaling factors (log-domain for stability)
        log_a = np.zeros(n_rows)  # Row scaling
        log_b = np.zeros(n_cols)  # Column scaling

        # Target values (regularized)
        target_vals = target_coo.data + reg
        target_vals = target_vals / target_vals.sum()

        # Marginals (regularized and normalized)
        mu_row = marginal_row.flatten() + reg
        mu_row = mu_row / mu_row.sum()

        mu_col = marginal_col.flatten() + reg
        mu_col = mu_col / mu_col.sum()

        history = {'row_err': [], 'col_err': [], 'time': []}
        start_time = time.time()

        for it in range(self.config.sinkhorn_max_iter):
            # Current scaled values: π(i,j) = a[i] * target[i,j] * b[j]
            # In log domain: log π = log_a[row] + log(target) + log_b[col]

            # Step 1: Row scaling to match μ_row
            # Compute current row sums
            row_sums = np.zeros(n_rows)
            scaled_vals = target_vals * np.exp(log_a[target_coo.row] + log_b[target_coo.col])
            np.add.at(row_sums, target_coo.row, scaled_vals)
            row_sums = np.maximum(row_sums, reg)

            # Update row scaling: a_new = a * (μ_row / row_sum)
            log_a = log_a + np.log(mu_row / row_sums)

            row_err = np.abs(row_sums - mu_row).sum()

            # Step 2: Column scaling to match μ_col
            col_sums = np.zeros(n_cols)
            scaled_vals = target_vals * np.exp(log_a[target_coo.row] + log_b[target_coo.col])
            np.add.at(col_sums, target_coo.col, scaled_vals)
            col_sums = np.maximum(col_sums, reg)

            # Update column scaling
            log_b = log_b + np.log(mu_col / col_sums)

            col_err = np.abs(col_sums - mu_col).sum()

            history['row_err'].append(row_err)
            history['col_err'].append(col_err)
            history['time'].append(time.time() - start_time)

            if self.config.verbose and (it + 1) % self.config.log_freq == 0:
                print(f"  Sinkhorn iter {it+1}: row_err={row_err:.2e}, col_err={col_err:.2e}")

            # Check convergence
            if row_err < self.config.sinkhorn_tol and col_err < self.config.sinkhorn_tol:
                if self.config.verbose:
                    print(f"  Sinkhorn converged at iteration {it+1}")
                break

        # Construct final optimal distribution
        final_vals = target_vals * np.exp(log_a[target_coo.row] + log_b[target_coo.col])

        # Normalize to ensure it's a proper distribution
        final_vals = final_vals / final_vals.sum()

        pi_star = sparse.csr_matrix(
            (final_vals, (target_coo.row, target_coo.col)),
            shape=target_interaction.shape
        )

        history['converged'] = it < self.config.sinkhorn_max_iter - 1
        history['iterations'] = it + 1

        return pi_star, history


class MicroDecomposer:
    """Phase 2: Micro-level decomposition of aggregate distribution to individuals"""

    def __init__(self, config: RakingConfig):
        self.config = config

    def compute_user_costs(self,
                          user_pattern: UserPattern,
                          support_rows: np.ndarray,
                          support_cols: np.ndarray,
                          grid_h: int,
                          grid_w: int,
                          loc_type_row: str,
                          loc_type_col: str) -> np.ndarray:
        """Compute cost for each (row, col) configuration for a user

        Args:
            user_pattern: User's life pattern
            support_rows: Row indices in support set
            support_cols: Column indices in support set
            grid_h, grid_w: Grid dimensions
            loc_type_row: Location type for rows (e.g., 'H')
            loc_type_col: Location type for columns (e.g., 'W')

        Returns:
            Cost array of shape (n_support,)
        """
        n_support = len(support_rows)

        if self.config.cost_type == 'distance':
            # Distance-based cost: penalize far (h, w) pairs
            # Convert flat indices to 2D coordinates
            row_y, row_x = support_rows // grid_w, support_rows % grid_w
            col_y, col_x = support_cols // grid_w, support_cols % grid_w

            # Euclidean distance (normalized by grid diagonal)
            dist = np.sqrt((row_y - col_y)**2 + (row_x - col_x)**2)
            max_dist = np.sqrt(grid_h**2 + grid_w**2)
            costs = dist / max_dist

        elif self.config.cost_type == 'frequency':
            # Frequency-based cost: prefer configurations that match user's activity pattern
            # This requires information about how often user visits different location types
            # For now, use uniform (will be refined later)
            costs = np.ones(n_support)

        else:  # combined
            # Combined distance and frequency
            row_y, row_x = support_rows // grid_w, support_rows % grid_w
            col_y, col_x = support_cols // grid_w, support_cols % grid_w
            dist = np.sqrt((row_y - col_y)**2 + (row_x - col_x)**2)
            max_dist = np.sqrt(grid_h**2 + grid_w**2)

            dist_cost = dist / max_dist
            freq_cost = np.ones(n_support)  # Placeholder

            w = self.config.distance_weight
            costs = w * dist_cost + (1 - w) * freq_cost

        return costs

    def decompose_ipa(self,
                      pi_star: sparse.csr_matrix,
                      user_patterns: Dict[int, UserPattern],
                      grid_h: int,
                      grid_w: int,
                      loc_type_row: str,
                      loc_type_col: str) -> Dict[int, np.ndarray]:
        """Iterative Proportional Allocation (IPA) - Probabilistic decomposition

        Finds J_i(h,w) for each user such that:
        - Σ_i J_i(h,w) = π*(h,w) for all (h,w)
        - Σ_{h,w} J_i(h,w) = 1 for all i
        - J_i minimizes cost-weighted distance to uniform
        """
        n_users = len(user_patterns)
        user_ids = list(user_patterns.keys())

        # Get support set
        pi_coo = pi_star.tocoo()
        support_rows = pi_coo.row
        support_cols = pi_coo.col
        target_vals = pi_coo.data
        n_support = len(support_rows)

        if self.config.verbose:
            print(f"  IPA: {n_users} users, {n_support} support positions")

        # Initialize J_i based on costs
        # J_i(h,w) ∝ exp(-C_i(h,w) / T) * π*(h,w)
        J = np.zeros((n_users, n_support))

        for i, uid in enumerate(user_ids):
            costs = self.compute_user_costs(
                user_patterns[uid], support_rows, support_cols,
                grid_h, grid_w, loc_type_row, loc_type_col
            )
            J[i] = np.exp(-costs / self.config.ipa_temp) * target_vals
            J[i] = J[i] / (J[i].sum() + 1e-10)

        # Sinkhorn iteration between users and configurations
        for it in range(self.config.ipa_max_iter):
            # Step 1: Scale to match target configuration capacities
            # Target: Σ_i J_i(h,w) = π*(h,w) * n_users
            config_sums = J.sum(axis=0)
            scale_factors = (target_vals * n_users) / (config_sums + 1e-10)
            J = J * scale_factors[np.newaxis, :]

            config_err = np.abs(J.sum(axis=0) - target_vals * n_users).sum()

            # Step 2: Scale to ensure each user has total probability 1
            user_sums = J.sum(axis=1, keepdims=True)
            J = J / (user_sums + 1e-10)

            user_err = np.abs(J.sum(axis=1) - 1).sum()

            if self.config.verbose and (it + 1) % self.config.log_freq == 0:
                print(f"    IPA iter {it+1}: config_err={config_err:.4f}, user_err={user_err:.4f}")

            if config_err < self.config.ipa_tol and user_err < self.config.ipa_tol:
                if self.config.verbose:
                    print(f"    IPA converged at iteration {it+1}")
                break

        # Convert to per-user allocations
        # J_i gives probability over support set, need to convert to full grid
        grid_size = grid_h * grid_w
        allocations = {}

        for i, uid in enumerate(user_ids):
            # Create sparse allocation for this user
            # For now, we only handle the interaction pair, not individual locations
            # This will be integrated with the full allocation later
            alloc = np.zeros(grid_size)
            # Marginal over rows (e.g., H locations)
            np.add.at(alloc, support_rows, J[i])
            alloc = alloc / (alloc.sum() + 1e-10)
            allocations[uid] = alloc

        return allocations

    def decompose_ics(self,
                      pi_star: sparse.csr_matrix,
                      user_patterns: Dict[int, UserPattern],
                      grid_h: int,
                      grid_w: int,
                      loc_type_row: str,
                      loc_type_col: str) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, np.ndarray]]:
        """Iterative Capacity Sampling (ICS) - Discrete decomposition

        Greedily assigns each user to a configuration (h, w) while respecting capacities.

        Returns:
            assignments: Dict mapping user_id to (h, w) configuration
            allocations: Dict mapping user_id to probability distribution (from assignments)
        """
        n_users = len(user_patterns)
        user_ids = list(user_patterns.keys())

        # Get support set
        pi_coo = pi_star.tocoo()
        support_rows = pi_coo.row
        support_cols = pi_coo.col
        target_vals = pi_coo.data
        n_support = len(support_rows)

        # Initialize capacities (round to integers)
        capacities = np.round(target_vals * n_users).astype(int)

        # Ensure total capacity equals number of users
        total_cap = capacities.sum()
        if total_cap < n_users:
            # Add to largest capacities
            deficit = n_users - total_cap
            top_indices = np.argsort(-target_vals)[:deficit]
            capacities[top_indices] += 1
        elif total_cap > n_users:
            # Remove from smallest non-zero capacities
            surplus = total_cap - n_users
            nonzero = np.where(capacities > 0)[0]
            sorted_nonzero = nonzero[np.argsort(target_vals[nonzero])]
            for idx in sorted_nonzero[:surplus]:
                if capacities[idx] > 0:
                    capacities[idx] -= 1

        if self.config.verbose:
            print(f"  ICS: {n_users} users, {n_support} positions, total capacity={capacities.sum()}")

        # Compute costs for all users
        all_costs = np.zeros((n_users, n_support))
        for i, uid in enumerate(user_ids):
            all_costs[i] = self.compute_user_costs(
                user_patterns[uid], support_rows, support_cols,
                grid_h, grid_w, loc_type_row, loc_type_col
            )

        # Order users (optionally by cost variance for better assignment quality)
        if self.config.ics_sort_by_variance:
            cost_vars = all_costs.var(axis=1)
            user_order = np.argsort(-cost_vars)  # High variance first (more constrained)
        else:
            user_order = np.arange(n_users)

        # Greedy assignment
        assignments = {}
        remaining_cap = capacities.copy()

        for idx, i in enumerate(user_order):
            uid = user_ids[i]
            costs = all_costs[i]

            # Compute sampling probability based on cost and remaining capacity
            valid_mask = remaining_cap > 0
            if not valid_mask.any():
                # No capacity left, assign uniformly to any position
                j = np.random.randint(n_support)
            else:
                probs = np.zeros(n_support)
                probs[valid_mask] = remaining_cap[valid_mask] * np.exp(-costs[valid_mask] / self.config.ics_temp)
                probs = probs / (probs.sum() + 1e-10)

                # Sample
                j = np.random.choice(n_support, p=probs)

            # Assign
            h_idx = support_rows[j]
            w_idx = support_cols[j]
            assignments[uid] = (h_idx, w_idx)
            remaining_cap[j] = max(0, remaining_cap[j] - 1)

            if self.config.verbose and (idx + 1) % (n_users // 10 + 1) == 0:
                pct = (idx + 1) / n_users * 100
                used_cap = capacities.sum() - remaining_cap.sum()
                print(f"    ICS progress: {pct:.0f}%, assigned={used_cap}/{capacities.sum()}")

        # Convert discrete assignments to probability distributions
        grid_size = grid_h * grid_w
        allocations = {}
        for uid, (h_idx, w_idx) in assignments.items():
            alloc = np.zeros(grid_size)
            alloc[h_idx] = 1.0  # Discrete assignment to row index
            allocations[uid] = alloc

        return assignments, allocations


class SSDMFO_RD:
    """SS-DMFO 5.0: Raking and Decomposition optimizer

    Two-phase approach:
    1. Macro Sinkhorn: Find optimal aggregate distribution π*
    2. Micro Decomposition: Assign users to configurations respecting π*
    """

    def __init__(self, config: Optional[RakingConfig] = None):
        self.config = config or RakingConfig()
        self.macro_solver = MacroSinkhorn(self.config)
        self.micro_solver = MicroDecomposer(self.config)

    def run(self, constraints: Constraints, user_patterns: Dict[int, UserPattern]) -> Result:
        """Run the full Raking and Decomposition pipeline

        Args:
            constraints: Spatial and interaction constraints
            user_patterns: Dictionary of user life patterns

        Returns:
            Result with per-user allocations
        """
        start_time = time.time()
        n_users = len(user_patterns)
        grid_size = constraints.grid_h * constraints.grid_w

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("SS-DMFO 5.0: Raking and Decomposition")
            print(f"{'='*60}")
            print(f"Users: {n_users}, Grid: {constraints.grid_h}x{constraints.grid_w}")

        # ================================================================
        # Phase 1: Macro Sinkhorn for each interaction type
        # ================================================================
        if self.config.verbose:
            print(f"\n[Phase 1] Macro Sinkhorn (Matrix Raking)")

        pi_stars = {}

        # H-W interaction
        if self.config.verbose:
            print(f"\n  Processing H-W interaction...")
        pi_hw, hist_hw = self.macro_solver.run(
            constraints.interaction.HW,
            constraints.spatial.H.flatten(),
            constraints.spatial.W.flatten()
        )
        pi_stars['HW'] = pi_hw

        # H-O interaction
        if self.config.verbose:
            print(f"\n  Processing H-O interaction...")
        pi_ho, hist_ho = self.macro_solver.run(
            constraints.interaction.HO,
            constraints.spatial.H.flatten(),
            constraints.spatial.O.flatten()
        )
        pi_stars['HO'] = pi_ho

        # W-O interaction
        if self.config.verbose:
            print(f"\n  Processing W-O interaction...")
        pi_wo, hist_wo = self.macro_solver.run(
            constraints.interaction.WO,
            constraints.spatial.W.flatten(),
            constraints.spatial.O.flatten()
        )
        pi_stars['WO'] = pi_wo

        phase1_time = time.time() - start_time
        if self.config.verbose:
            print(f"\n  Phase 1 completed in {phase1_time:.1f}s")

        # ================================================================
        # Phase 2: Micro Decomposition
        # ================================================================
        if self.config.verbose:
            print(f"\n[Phase 2] Micro Decomposition ({self.config.decomposition_method.upper()})")

        phase2_start = time.time()

        # For now, we use H-W interaction as the primary constraint
        # and derive H, W allocations from it

        if self.config.decomposition_method == 'ipa':
            # IPA gives probabilistic allocations
            h_allocations = self.micro_solver.decompose_ipa(
                pi_stars['HW'], user_patterns,
                constraints.grid_h, constraints.grid_w,
                'H', 'W'
            )

            # Also need W allocations (marginal over columns)
            pi_hw_coo = pi_stars['HW'].tocoo()
            w_allocations = {}
            for uid in user_patterns:
                w_alloc = np.zeros(grid_size)
                # This is simplified - in full implementation, need proper joint handling
                w_alloc = constraints.spatial.W.flatten()
                w_allocations[uid] = w_alloc

        else:  # ics
            assignments, h_allocations = self.micro_solver.decompose_ics(
                pi_stars['HW'], user_patterns,
                constraints.grid_h, constraints.grid_w,
                'H', 'W'
            )

            # Get W allocations from assignments
            w_allocations = {}
            for uid, (h_idx, w_idx) in assignments.items():
                w_alloc = np.zeros(grid_size)
                w_alloc[w_idx] = 1.0
                w_allocations[uid] = w_alloc

        phase2_time = time.time() - phase2_start
        if self.config.verbose:
            print(f"\n  Phase 2 completed in {phase2_time:.1f}s")

        # ================================================================
        # Construct final allocations
        # ================================================================
        # Each user needs allocations for their semantic locations
        allocations = {}

        for uid, pattern in user_patterns.items():
            n_locs = len(pattern.locations)
            user_alloc = np.zeros((n_locs, grid_size))

            h_alloc = h_allocations.get(uid, constraints.spatial.H.flatten())
            w_alloc = w_allocations.get(uid, constraints.spatial.W.flatten())
            o_alloc = constraints.spatial.O.flatten()  # O uses spatial prior for now

            for loc_idx, loc in enumerate(pattern.locations):
                if loc.type == 'H':
                    user_alloc[loc_idx] = h_alloc
                elif loc.type == 'W':
                    user_alloc[loc_idx] = w_alloc
                else:  # O
                    user_alloc[loc_idx] = o_alloc

                # Ensure normalized
                if user_alloc[loc_idx].sum() > 0:
                    user_alloc[loc_idx] = user_alloc[loc_idx] / user_alloc[loc_idx].sum()

            allocations[uid] = user_alloc

        total_time = time.time() - start_time
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Total time: {total_time:.1f}s")
            print(f"{'='*60}")

        return Result(
            method_name=f'SS-DMFO 5.0 R&D ({self.config.decomposition_method.upper()})',
            allocations=allocations,
            runtime=total_time,
            iterations=hist_hw.get('iterations', 0) + hist_ho.get('iterations', 0) + hist_wo.get('iterations', 0),
        )
