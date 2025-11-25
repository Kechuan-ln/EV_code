"""SS-DMFO Dual Optimizer - Vectorized Version

Core algorithm flow:
1. Initialize potentials alpha, beta
2. Loop until convergence:
   a. Given potentials, compute optimal response Q_i for each user via MFVI
   b. Aggregate responses to get generated statistics mu_gen, pi_gen
   c. Compute gradient = generated statistics - real statistics
   d. Update potentials via GRADIENT ASCENT (dual maximization)
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from scipy import sparse

from ..data.structures import (
    Constraints, UserPattern, Result,
    SpatialConstraints, InteractionConstraints
)
from ..baselines.base import BaseMethod
from .potentials import DualPotentials, PotentialsWithMomentum
from .mean_field import FastMeanFieldSolver


class SSDMFOOptimizer(BaseMethod):
    """SS-DMFO Dual Optimizer"""

    def __init__(self,
                 phase: int = 2,
                 max_iter: int = 100,
                 lr: float = 0.1,
                 temperature: float = 1.0,
                 tolerance: float = 1e-4,
                 log_freq: int = 10,
                 use_adam: bool = True):
        """
        Args:
            phase: optimization phase (1=spatial, 2=spatial+interaction)
            max_iter: maximum iterations
            lr: learning rate
            temperature: MFVI temperature parameter
            tolerance: convergence tolerance
            log_freq: logging frequency
            use_adam: whether to use Adam optimizer
        """
        super().__init__(f"SS-DMFO(phase={phase})")
        self.phase = phase
        self.max_iter = max_iter
        self.lr = lr
        self.temperature = temperature
        self.tolerance = tolerance
        self.log_freq = log_freq
        self.use_adam = use_adam

        # MFVI solver
        self.mf_solver = FastMeanFieldSolver(
            temperature=temperature,
            max_iter=1,  # Use single iteration for speed
            damping=0.5
        )

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """Run SS-DMFO optimization"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        # Normalize constraints
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        # Initialize potentials
        print(f"Initializing potentials (phase={self.phase})...")
        potentials = DualPotentials.initialize(
            grid_h, grid_w, phase=self.phase, init_scale=0.01
        )

        # Optimizer
        if self.use_adam:
            optimizer = PotentialsWithMomentum(potentials)
        else:
            optimizer = None

        # Optimization loop
        print(f"Starting optimization (max_iter={self.max_iter}, lr={self.lr})...")
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # Step 1: Compute responses for all users
            responses = self.mf_solver.compute_all_responses_fast(
                user_patterns, potentials, constraints, self.phase
            )

            # Step 2: Aggregate statistics
            gen_spatial = self._aggregate_spatial(
                responses, user_patterns, grid_h, grid_w
            )

            # Step 3: Compute gradients (gen - real)
            # For DUAL ASCENT, we want alpha += lr * (gen - real)
            # Since update_alpha does alpha -= lr * grad, we negate
            gradients = self._compute_spatial_gradients(
                gen_spatial, constraints.spatial
            )
            # NEGATE for gradient ascent
            gradients = {k: -v for k, v in gradients.items()}

            # Step 4: Compute loss (monitoring)
            loss = self._compute_loss(gen_spatial, constraints.spatial)

            # Step 5: Update potentials
            if self.use_adam:
                optimizer.step(gradients, self.lr)
            else:
                for loc_type, grad in gradients.items():
                    potentials.update_alpha(loc_type, grad, self.lr)

            # Logging
            if iteration % self.log_freq == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: Loss = {loss:.6f}")

            # Convergence check
            if abs(prev_loss - loss) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break

            prev_loss = loss

        # Final responses as allocations
        final_responses = self.mf_solver.compute_all_responses_fast(
            user_patterns, potentials, constraints, self.phase
        )

        return final_responses

    def _aggregate_spatial(self,
                           responses: Dict[int, np.ndarray],
                           user_patterns: Dict[int, UserPattern],
                           grid_h: int, grid_w: int) -> SpatialConstraints:
        """Aggregate spatial statistics (vectorized)"""
        grid_size = grid_h * grid_w
        H_flat = np.zeros(grid_size)
        W_flat = np.zeros(grid_size)
        O_flat = np.zeros(grid_size)

        for user_id, Q in responses.items():
            pattern = user_patterns[user_id]
            for loc_idx, loc in enumerate(pattern.locations):
                if loc.type == 'H':
                    H_flat += Q[loc_idx]
                elif loc.type == 'W':
                    W_flat += Q[loc_idx]
                elif loc.type == 'O':
                    O_flat += Q[loc_idx]

        result = SpatialConstraints(
            H=H_flat.reshape(grid_h, grid_w),
            W=W_flat.reshape(grid_h, grid_w),
            O=O_flat.reshape(grid_h, grid_w)
        )
        result.normalize()
        return result

    def _aggregate_interaction_vectorized(self,
                                          responses: Dict[int, np.ndarray],
                                          user_patterns: Dict[int, UserPattern],
                                          grid_size: int,
                                          top_k: int = 30) -> InteractionConstraints:
        """Aggregate interaction statistics (VECTORIZED VERSION)

        Key optimizations:
        1. Batch all users' top-k indices extraction
        2. Use NumPy outer products instead of nested loops
        3. Accumulate sparse matrices efficiently
        """
        # Collect all location distributions by type
        all_H = []  # List of (user_id, loc_idx, Q_vector)
        all_W = []
        all_O = []

        for user_id, Q in responses.items():
            pattern = user_patterns[user_id]
            for loc_idx, loc in enumerate(pattern.locations):
                if loc.type == 'H':
                    all_H.append((user_id, loc_idx, Q[loc_idx]))
                elif loc.type == 'W':
                    all_W.append((user_id, loc_idx, Q[loc_idx]))
                elif loc.type == 'O':
                    all_O.append((user_id, loc_idx, Q[loc_idx]))

        # Compute interactions using vectorized operations
        HW = self._compute_pairwise_interaction_fast(all_H, all_W, user_patterns, grid_size, top_k)
        HO = self._compute_pairwise_interaction_fast(all_H, all_O, user_patterns, grid_size, top_k)
        WO = self._compute_pairwise_interaction_fast(all_W, all_O, user_patterns, grid_size, top_k)

        result = InteractionConstraints(HW=HW, HO=HO, WO=WO)
        result.normalize()
        return result

    def _compute_pairwise_interaction_fast(self,
                                           locs1: List[Tuple],
                                           locs2: List[Tuple],
                                           user_patterns: Dict[int, UserPattern],
                                           grid_size: int,
                                           top_k: int) -> sparse.csr_matrix:
        """Compute pairwise interaction matrix (vectorized)

        For each user, compute outer product of their type1 and type2 locations.
        """
        if not locs1 or not locs2:
            return sparse.csr_matrix((grid_size, grid_size))

        # Group by user
        user_locs1 = {}
        user_locs2 = {}

        for user_id, loc_idx, q in locs1:
            if user_id not in user_locs1:
                user_locs1[user_id] = []
            user_locs1[user_id].append(q)

        for user_id, loc_idx, q in locs2:
            if user_id not in user_locs2:
                user_locs2[user_id] = []
            user_locs2[user_id].append(q)

        # Find users with both types
        common_users = set(user_locs1.keys()) & set(user_locs2.keys())

        if not common_users:
            return sparse.csr_matrix((grid_size, grid_size))

        # Accumulate sparse entries
        rows = []
        cols = []
        data = []

        for user_id in common_users:
            qs1 = user_locs1[user_id]
            qs2 = user_locs2[user_id]

            # Stack all Q vectors for this user
            Q1 = np.vstack(qs1)  # (n1, grid_size)
            Q2 = np.vstack(qs2)  # (n2, grid_size)

            # Get top-k indices for each distribution
            # Sum across locations of same type for efficiency
            q1_sum = Q1.sum(axis=0)  # (grid_size,)
            q2_sum = Q2.sum(axis=0)  # (grid_size,)

            # Top-k for each
            if top_k < grid_size:
                idx1 = np.argpartition(q1_sum, -top_k)[-top_k:]
                idx2 = np.argpartition(q2_sum, -top_k)[-top_k:]
            else:
                idx1 = np.where(q1_sum > 1e-10)[0]
                idx2 = np.where(q2_sum > 1e-10)[0]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            # Get probabilities at these indices
            p1 = q1_sum[idx1]
            p2 = q2_sum[idx2]

            # Normalize
            p1 = p1 / (p1.sum() + 1e-10)
            p2 = p2 / (p2.sum() + 1e-10)

            # Outer product: p1[:, None] * p2[None, :]
            outer = np.outer(p1, p2)  # (top_k, top_k)

            # Convert to sparse entries
            for i, i1 in enumerate(idx1):
                for j, i2 in enumerate(idx2):
                    if outer[i, j] > 1e-15:
                        rows.append(i1)
                        cols.append(i2)
                        data.append(outer[i, j])

        if not data:
            return sparse.csr_matrix((grid_size, grid_size))

        # Build sparse matrix (automatically sums duplicates)
        return sparse.csr_matrix((data, (rows, cols)), shape=(grid_size, grid_size))

    def _aggregate_interaction(self,
                               responses: Dict[int, np.ndarray],
                               user_patterns: Dict[int, UserPattern],
                               grid_size: int,
                               top_k: int = 30) -> InteractionConstraints:
        """Aggregate interaction statistics - wrapper for vectorized version"""
        return self._aggregate_interaction_vectorized(
            responses, user_patterns, grid_size, top_k
        )

    def _compute_spatial_gradients(self,
                                   gen: SpatialConstraints,
                                   real: SpatialConstraints) -> Dict[str, np.ndarray]:
        """Compute spatial distribution gradients

        Gradient = generated distribution - real distribution
        """
        return {
            'H': gen.H - real.H,
            'W': gen.W - real.W,
            'O': gen.O - real.O
        }

    def _compute_interaction_gradients(self,
                                       gen: InteractionConstraints,
                                       real: InteractionConstraints) -> Dict[str, sparse.csr_matrix]:
        """Compute interaction distribution gradients"""
        return {
            'HW': gen.HW - real.HW,
            'HO': gen.HO - real.HO,
            'WO': gen.WO - real.WO
        }

    def _compute_loss(self,
                      gen: SpatialConstraints,
                      real: SpatialConstraints) -> float:
        """Compute total loss (for monitoring)"""
        def jsd(p, q):
            p = p.flatten() + 1e-10
            q = q.flatten() + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p/m)) + np.sum(q * np.log(q/m)))

        loss_H = jsd(gen.H, real.H)
        loss_W = jsd(gen.W, real.W)
        loss_O = jsd(gen.O, real.O)

        return (loss_H + loss_W + loss_O) / 3


class SSDMFOPhase2(SSDMFOOptimizer):
    """Phase 2 specialized optimizer (jointly optimize spatial and interaction constraints)

    Key improvements:
    - Updates both alpha (spatial) and beta (interaction) potentials
    - Computes interaction loss less frequently for speed
    - Uses proper gradient ascent for dual optimization
    - Vectorized interaction computation
    """

    def __init__(self,
                 max_iter: int = 200,
                 lr: float = 0.1,
                 lr_beta: float = 0.01,
                 temperature: float = 1.0,
                 interaction_weight: float = 0.5,
                 interaction_freq: int = 5,
                 top_k: int = 20,
                 **kwargs):
        """
        Args:
            lr_beta: learning rate for beta potentials
            interaction_weight: weight for interaction constraints
            interaction_freq: compute interaction every N iterations
            top_k: number of top probability cells to consider
        """
        super().__init__(phase=2, max_iter=max_iter, lr=lr,
                         temperature=temperature, **kwargs)
        self.lr_beta = lr_beta
        self.interaction_weight = interaction_weight
        self.interaction_freq = interaction_freq
        self.top_k = top_k
        self.name = f"SS-DMFO-P2(T={temperature})"

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """Phase 2 optimization: jointly consider spatial and interaction constraints"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        # Normalize
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        # Initialize
        print(f"Initializing SS-DMFO Phase 2...")
        potentials = DualPotentials.initialize(grid_h, grid_w, phase=2)

        if self.use_adam:
            optimizer = PotentialsWithMomentum(potentials)

        print(f"Running optimization (max_iter={self.max_iter}, lr={self.lr})...")
        prev_loss = float('inf')
        interaction_loss = 0.0

        for iteration in range(self.max_iter):
            iter_start = time.time()

            # Compute responses (Phase 2 considers interactions via beta)
            responses = self.mf_solver.compute_all_responses_fast(
                user_patterns, potentials, constraints, phase=2
            )

            # Aggregate spatial statistics
            gen_spatial = self._aggregate_spatial(
                responses, user_patterns, grid_h, grid_w
            )

            # Compute spatial gradients (negated for gradient ascent)
            spatial_grads = self._compute_spatial_gradients(gen_spatial, constraints.spatial)
            spatial_grads = {k: -v for k, v in spatial_grads.items()}

            # Compute spatial loss
            spatial_loss = self._compute_loss(gen_spatial, constraints.spatial)

            # Compute interaction every N iterations (expensive)
            if constraints.interaction is not None and iteration % self.interaction_freq == 0:
                gen_interaction = self._aggregate_interaction_vectorized(
                    responses, user_patterns, grid_size, top_k=self.top_k
                )
                interaction_loss = self._compute_interaction_loss_fast(
                    gen_interaction, constraints.interaction
                )

                # Update beta potentials
                interaction_grads = self._compute_interaction_gradients(
                    gen_interaction, constraints.interaction
                )
                self._update_beta_potentials(potentials, interaction_grads, self.lr_beta)

            total_loss = spatial_loss + self.interaction_weight * interaction_loss

            # Update alpha potentials
            if self.use_adam:
                optimizer.step(spatial_grads, self.lr)
            else:
                for loc_type, grad in spatial_grads.items():
                    potentials.update_alpha(loc_type, grad, self.lr)

            # Logging
            iter_time = time.time() - iter_start
            if iteration % self.log_freq == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: Spatial={spatial_loss:.4f}, "
                      f"Interact={interaction_loss:.4f}, Total={total_loss:.4f} "
                      f"({iter_time:.2f}s)")

            # Convergence
            if abs(prev_loss - total_loss) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break
            prev_loss = total_loss

        return self.mf_solver.compute_all_responses_fast(
            user_patterns, potentials, constraints, phase=2
        )

    def _update_beta_potentials(self, potentials: DualPotentials,
                                 grads: Dict[str, sparse.csr_matrix],
                                 lr: float):
        """Update beta potentials via gradient ascent

        For dual ascent: beta += lr * (gen - real)
        Since update_beta does beta -= lr * grad, we negate
        """
        for key in ['HW', 'HO', 'WO']:
            if key in grads:
                # Negate for gradient ascent
                potentials.update_beta(key[0], key[1], -grads[key], lr)

    def _compute_interaction_loss_fast(self,
                                       gen: InteractionConstraints,
                                       real: InteractionConstraints) -> float:
        """Compute interaction JSD loss (optimized sparse version)"""
        def sparse_jsd_fast(p: sparse.csr_matrix, q: sparse.csr_matrix) -> float:
            # Get union of non-zero indices efficiently
            p_coo = p.tocoo()
            q_coo = q.tocoo()

            # Use sets for fast intersection
            p_indices = set(zip(p_coo.row.tolist(), p_coo.col.tolist()))
            q_indices = set(zip(q_coo.row.tolist(), q_coo.col.tolist()))
            all_indices = p_indices | q_indices

            if not all_indices:
                return 0.0

            # Convert to arrays for fast lookup
            indices = list(all_indices)
            n = len(indices)

            # Extract values
            p_vals = np.array([p[r, c] for r, c in indices]) + 1e-10
            q_vals = np.array([q[r, c] for r, c in indices]) + 1e-10

            # Normalize
            p_vals = p_vals / p_vals.sum()
            q_vals = q_vals / q_vals.sum()

            m = 0.5 * (p_vals + q_vals)
            return 0.5 * (np.sum(p_vals * np.log(p_vals/m)) +
                         np.sum(q_vals * np.log(q_vals/m)))

        losses = []
        if gen.HW.nnz > 0 or real.HW.nnz > 0:
            losses.append(sparse_jsd_fast(gen.HW, real.HW))
        if gen.HO.nnz > 0 or real.HO.nnz > 0:
            losses.append(sparse_jsd_fast(gen.HO, real.HO))
        if gen.WO.nnz > 0 or real.WO.nnz > 0:
            losses.append(sparse_jsd_fast(gen.WO, real.WO))

        return np.mean(losses) if losses else 0.0
