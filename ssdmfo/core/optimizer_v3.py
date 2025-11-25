"""SS-DMFO 3.0 Optimizer - Fixed Version

Key fixes based on expert feedback:
1. Fix template trap: Each user gets individual Q_i (not shared templates)
2. Stochastic optimization: Batch processing to control memory
3. Iterative MFVI: Proper beta coupling between locations
4. Gumbel noise injection: Break symmetry for diversity
5. Temperature annealing: Control exploration vs exploitation

Author: SS-DMFO Team
Date: 2025-11-25
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from scipy import sparse
from dataclasses import dataclass

from ..data.structures import (
    Constraints, UserPattern, Result,
    SpatialConstraints, InteractionConstraints
)
from ..baselines.base import BaseMethod
from .potentials import DualPotentials, PotentialsWithMomentum


@dataclass
class SSDMFO3Config:
    """Configuration for SS-DMFO 3.0"""
    # Optimization
    max_iter: int = 100
    lr_alpha: float = 0.1
    lr_beta: float = 0.01
    tolerance: float = 1e-4

    # MFVI
    mfvi_iter: int = 5          # Iterations for MFVI convergence
    mfvi_damping: float = 0.5   # Damping factor

    # Temperature annealing (SLOWER decay to prevent rebound)
    temp_init: float = 2.0      # High temperature (exploration)
    temp_final: float = 1.0     # Higher final temp to maintain diversity
    temp_decay: str = 'exponential'  # 'linear' or 'exponential'

    # Gumbel noise (maintain diversity)
    gumbel_scale: float = 0.1   # Scale of Gumbel noise
    gumbel_decay: float = 0.995 # SLOWER decay to maintain diversity
    gumbel_final: float = 0.05  # Minimum Gumbel scale (never go to 0)

    # Stochastic optimization
    batch_size: int = 50        # Users per batch

    # Logging
    log_freq: int = 10

    # Interaction
    interaction_freq: int = 5   # Compute interaction every N iters
    top_k: int = 50             # Top-k for interaction (increased from 30)

    # Early stopping for interaction
    early_stop_patience: int = 10  # Stop if no improvement for N interaction checks
    phase_separation: bool = True  # Phase 1: spatial only, Phase 2: both


class MeanFieldSolverV3:
    """MFVI Solver V3 - With individual responses and Gumbel noise

    OPTIMIZED: Vectorized operations, cached alpha, fast softmax
    """

    def __init__(self, config: SSDMFO3Config):
        self.config = config
        # Cache for flattened alpha (avoid repeated flatten)
        self._alpha_cache = None
        self._alpha_version = -1

    def _get_alpha_flat(self, potentials: DualPotentials) -> Dict[str, np.ndarray]:
        """Get cached flattened alpha potentials"""
        # Simple version check using id (potentials object identity)
        pot_id = id(potentials)
        if self._alpha_cache is None or self._alpha_version != pot_id:
            self._alpha_cache = {
                'H': potentials.alpha_H.ravel(),  # ravel is faster than flatten when possible
                'W': potentials.alpha_W.ravel(),
                'O': potentials.alpha_O.ravel()
            }
            self._alpha_version = pot_id
        return self._alpha_cache

    def invalidate_cache(self):
        """Call after potentials are updated"""
        self._alpha_cache = None

    def compute_user_response_individual(self,
                                         user: UserPattern,
                                         potentials: DualPotentials,
                                         grid_size: int,
                                         temperature: float,
                                         gumbel_scale: float,
                                         use_beta: bool = True) -> np.ndarray:
        """Compute individual response Q_i for a single user - OPTIMIZED"""
        n_locs = len(user.locations)
        loc_types = [loc.type for loc in user.locations]

        # Use cached alpha
        alpha_flat = self._get_alpha_flat(potentials)

        # VECTORIZED: Initialize all Q at once
        Q = np.empty((n_locs, grid_size))

        # Generate all Gumbel noise at once
        all_gumbel = np.random.gumbel(0, gumbel_scale, (n_locs, grid_size))

        # Vectorized softmax initialization
        for i, lt in enumerate(loc_types):
            log_q = -alpha_flat[lt] / temperature + all_gumbel[i]
            log_q -= log_q.max()  # Numerical stability
            Q[i] = np.exp(log_q)
            Q[i] /= Q[i].sum() + 1e-10

        # Iterative MFVI with beta coupling (only if beta has content)
        if use_beta and self.config.mfvi_iter > 0:
            # Check if any beta has non-zero entries
            has_beta = (potentials.beta_HW is not None and potentials.beta_HW.nnz > 0) or \
                      (potentials.beta_HO is not None and potentials.beta_HO.nnz > 0) or \
                      (potentials.beta_WO is not None and potentials.beta_WO.nnz > 0)
            if has_beta:
                Q = self._iterative_mfvi_fast(
                    Q, loc_types, potentials, alpha_flat,
                    grid_size, temperature, gumbel_scale
                )

        return Q

    def _iterative_mfvi_fast(self,
                             Q: np.ndarray,
                             loc_types: List[str],
                             potentials: DualPotentials,
                             alpha_flat: Dict[str, np.ndarray],
                             grid_size: int,
                             temperature: float,
                             gumbel_scale: float) -> np.ndarray:
        """OPTIMIZED Iterative MFVI - reduced redundant computation"""
        n_locs = len(loc_types)
        damping = self.config.mfvi_damping
        inv_temp = 1.0 / temperature  # Precompute

        # Precompute which beta matrices to use for each (i,j) pair
        beta_lookup = {}
        for i in range(n_locs):
            for j in range(n_locs):
                if i != j:
                    beta = potentials.get_beta(loc_types[i], loc_types[j])
                    if beta is not None and beta.nnz > 0:
                        beta_lookup[(i, j)] = beta

        # If no beta interactions, skip MFVI entirely
        if not beta_lookup:
            return Q

        for mfvi_iter in range(self.config.mfvi_iter):
            Q_old = Q.copy()
            noise_scale = gumbel_scale * (0.5 ** mfvi_iter)

            # Generate all noise at once for this iteration
            all_gumbel = np.random.gumbel(0, noise_scale, (n_locs, grid_size))

            for i in range(n_locs):
                # Start with alpha field
                field = alpha_flat[loc_types[i]].copy()

                # Add beta interactions (using precomputed lookup)
                for j in range(n_locs):
                    if (i, j) in beta_lookup:
                        field += beta_lookup[(i, j)].dot(Q_old[j])

                # Fast softmax
                log_q = (-field + all_gumbel[i]) * inv_temp
                log_q -= log_q.max()
                q_new = np.exp(log_q)
                q_new *= (1.0 / (q_new.sum() + 1e-10))  # Faster than /=

                # Damped update
                Q[i] = damping * q_new + (1 - damping) * Q_old[i]

        return Q

    def compute_batch_responses(self,
                                user_batch: Dict[int, UserPattern],
                                potentials: DualPotentials,
                                grid_size: int,
                                temperature: float,
                                gumbel_scale: float,
                                use_beta: bool = True) -> Dict[int, np.ndarray]:
        """Compute responses for a batch of users - OPTIMIZED"""
        # Precompute alpha_flat ONCE for entire batch
        alpha_flat = {
            'H': potentials.alpha_H.ravel(),
            'W': potentials.alpha_W.ravel(),
            'O': potentials.alpha_O.ravel()
        }

        # Check beta status ONCE
        has_beta = use_beta and self.config.mfvi_iter > 0 and (
            (potentials.beta_HW is not None and potentials.beta_HW.nnz > 0) or
            (potentials.beta_HO is not None and potentials.beta_HO.nnz > 0) or
            (potentials.beta_WO is not None and potentials.beta_WO.nnz > 0)
        )

        responses = {}
        inv_temp = 1.0 / temperature

        for user_id, user in user_batch.items():
            Q = self._compute_user_fast(
                user, potentials, alpha_flat, grid_size,
                temperature, inv_temp, gumbel_scale, has_beta
            )
            responses[user_id] = Q
        return responses

    def _compute_user_fast(self,
                           user: UserPattern,
                           potentials: DualPotentials,
                           alpha_flat: Dict[str, np.ndarray],
                           grid_size: int,
                           temperature: float,
                           inv_temp: float,
                           gumbel_scale: float,
                           has_beta: bool) -> np.ndarray:
        """Fast single-user computation with precomputed alpha"""
        n_locs = len(user.locations)
        loc_types = [loc.type for loc in user.locations]

        # Initialize Q with vectorized Gumbel noise
        Q = np.empty((n_locs, grid_size))
        all_gumbel = np.random.gumbel(0, gumbel_scale, (n_locs, grid_size))

        for i, lt in enumerate(loc_types):
            log_q = -alpha_flat[lt] * inv_temp + all_gumbel[i]
            log_q -= log_q.max()
            Q[i] = np.exp(log_q)
            Q[i] *= (1.0 / (Q[i].sum() + 1e-10))

        # MFVI only if beta has content
        if has_beta:
            Q = self._iterative_mfvi_fast(
                Q, loc_types, potentials, alpha_flat,
                grid_size, temperature, gumbel_scale
            )

        return Q


class SSDMFOv3(BaseMethod):
    """SS-DMFO 3.0 Optimizer

    Key improvements:
    1. Individual user responses (no template trap)
    2. Stochastic optimization (batch processing)
    3. Iterative MFVI with beta coupling
    4. Gumbel noise for diversity
    5. Temperature annealing
    """

    def __init__(self, config: Optional[SSDMFO3Config] = None):
        self.config = config or SSDMFO3Config()
        super().__init__(f"SS-DMFO-v3")
        self.mf_solver = MeanFieldSolverV3(self.config)

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """Run SS-DMFO 3.0 optimization with stochastic updates"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w
        n_users = len(user_patterns)

        # Normalize constraints
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        # Initialize potentials
        print(f"[SS-DMFO 3.0] Initializing...")
        print(f"  Users: {n_users}, Grid: {grid_h}x{grid_w}={grid_size}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  MFVI iterations: {self.config.mfvi_iter}")
        print(f"  Temperature: {self.config.temp_init} -> {self.config.temp_final}")
        print(f"  Gumbel scale: {self.config.gumbel_scale} -> {self.config.gumbel_final}")
        print(f"  Phase separation: {self.config.phase_separation}")

        potentials = DualPotentials.initialize(grid_h, grid_w, phase=2)
        optimizer = PotentialsWithMomentum(potentials)

        # User list for batching
        user_ids = list(user_patterns.keys())
        n_batches = (n_users + self.config.batch_size - 1) // self.config.batch_size

        # Early stopping tracking
        best_interaction_loss = float('inf')
        best_potentials_state = None
        no_improve_count = 0
        best_iter = 0

        # Optimization loop
        print(f"\n[SS-DMFO 3.0] Starting optimization (max_iter={self.config.max_iter})...")
        prev_loss = float('inf')
        gumbel_scale = self.config.gumbel_scale

        # Determine phase transition point
        phase1_iters = self.config.max_iter // 3 if self.config.phase_separation else 0

        for iteration in range(self.config.max_iter):
            iter_start = time.time()

            # Phase separation: Phase 1 = spatial only, Phase 2 = both
            in_phase1 = self.config.phase_separation and iteration < phase1_iters
            use_beta = not in_phase1 and iteration >= 5

            # Temperature annealing (SLOWER - only start after phase1)
            if in_phase1:
                temperature = self.config.temp_init
            elif self.config.temp_decay == 'exponential':
                phase2_progress = (iteration - phase1_iters) / max(self.config.max_iter - phase1_iters - 1, 1)
                temperature = self.config.temp_init * (
                    self.config.temp_final / self.config.temp_init
                ) ** phase2_progress
            else:  # linear
                phase2_progress = (iteration - phase1_iters) / max(self.config.max_iter - phase1_iters - 1, 1)
                temperature = self.config.temp_init - (
                    self.config.temp_init - self.config.temp_final
                ) * phase2_progress

            # Gumbel noise decay (with minimum floor)
            gumbel_scale = max(
                self.config.gumbel_final,
                self.config.gumbel_scale * (self.config.gumbel_decay ** iteration)
            )

            # Shuffle users for stochastic optimization
            np.random.shuffle(user_ids)

            # Accumulators for gradients
            spatial_accum = {
                'H': np.zeros((grid_h, grid_w)),
                'W': np.zeros((grid_h, grid_w)),
                'O': np.zeros((grid_h, grid_w))
            }

            # Process users in batches
            all_responses = {}

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, n_users)
                batch_user_ids = user_ids[start_idx:end_idx]

                # Get batch users
                batch_users = {uid: user_patterns[uid] for uid in batch_user_ids}

                # Compute responses for this batch
                batch_responses = self.mf_solver.compute_batch_responses(
                    batch_users, potentials, grid_size,
                    temperature, gumbel_scale, use_beta
                )

                # Accumulate spatial statistics
                for user_id, Q in batch_responses.items():
                    pattern = user_patterns[user_id]
                    for loc_idx, loc in enumerate(pattern.locations):
                        probs = Q[loc_idx].reshape(grid_h, grid_w)
                        spatial_accum[loc.type] += probs

                # Store responses for interaction computation
                all_responses.update(batch_responses)

            # Normalize accumulated spatial stats
            gen_spatial = SpatialConstraints(
                H=spatial_accum['H'],
                W=spatial_accum['W'],
                O=spatial_accum['O']
            )
            gen_spatial.normalize()

            # Compute spatial gradients (negated for gradient ascent)
            spatial_grads = {
                'H': -(gen_spatial.H - constraints.spatial.H),
                'W': -(gen_spatial.W - constraints.spatial.W),
                'O': -(gen_spatial.O - constraints.spatial.O)
            }

            # Compute spatial loss
            spatial_loss = self._compute_jsd(gen_spatial, constraints.spatial)

            # Compute interaction (less frequently, and only in phase 2)
            interaction_loss = 0.0
            if constraints.interaction is not None and not in_phase1 and iteration % self.config.interaction_freq == 0:
                gen_interaction = self._aggregate_interaction_fast(
                    all_responses, user_patterns, grid_size
                )
                interaction_loss = self._compute_interaction_jsd(
                    gen_interaction, constraints.interaction
                )

                # Early stopping: track best interaction state
                if interaction_loss < best_interaction_loss - 0.001:
                    best_interaction_loss = interaction_loss
                    best_potentials_state = potentials.copy_state()
                    no_improve_count = 0
                    best_iter = iteration
                else:
                    no_improve_count += 1

                # Update beta potentials
                if use_beta:
                    self._update_beta(potentials, gen_interaction,
                                     constraints.interaction, self.config.lr_beta)

            total_loss = spatial_loss + 0.5 * interaction_loss

            # Update alpha potentials (reduce learning rate in phase 2)
            alpha_lr = self.config.lr_alpha if in_phase1 else self.config.lr_alpha * 0.5
            optimizer.step(spatial_grads, alpha_lr)

            # Logging
            iter_time = time.time() - iter_start
            phase_str = "P1" if in_phase1 else "P2"
            if iteration % self.config.log_freq == 0 or iteration < 5:
                print(f"  Iter {iteration:3d} [{phase_str}]: Spatial={spatial_loss:.4f}, "
                      f"Interact={interaction_loss:.4f}, T={temperature:.2f}, "
                      f"Gumbel={gumbel_scale:.3f} ({iter_time:.1f}s)")

            # Early stopping for interaction (only in phase 2)
            if not in_phase1 and no_improve_count >= self.config.early_stop_patience:
                print(f"  Early stopping: no interaction improvement for {no_improve_count} checks")
                print(f"  Restoring best state from iter {best_iter} (interact={best_interaction_loss:.4f})")
                if best_potentials_state is not None:
                    potentials.restore_state(best_potentials_state)
                break

            # Convergence check
            if abs(prev_loss - total_loss) < self.config.tolerance and iteration > 20:
                print(f"  Converged at iteration {iteration}")
                break
            prev_loss = total_loss

        # Final pass: use best potentials with moderate diversity
        print(f"\n[SS-DMFO 3.0] Computing final allocations...")
        print(f"  Best interaction: {best_interaction_loss:.4f} at iter {best_iter}")

        # Use optimized batch method for final pass
        final_responses = self.mf_solver.compute_batch_responses(
            user_patterns, potentials, grid_size,
            self.config.temp_final,
            self.config.gumbel_final,  # Keep some noise for diversity!
            use_beta=True
        )

        return final_responses

    def _aggregate_interaction_fast(self,
                                    responses: Dict[int, np.ndarray],
                                    user_patterns: Dict[int, UserPattern],
                                    grid_size: int) -> InteractionConstraints:
        """Aggregate interaction statistics with improved coverage"""
        top_k = self.config.top_k

        # Collect by type
        user_locs = {uid: {'H': [], 'W': [], 'O': []}
                     for uid in responses.keys()}

        for user_id, Q in responses.items():
            pattern = user_patterns[user_id]
            for loc_idx, loc in enumerate(pattern.locations):
                user_locs[user_id][loc.type].append(Q[loc_idx])

        # Compute pairwise interactions
        hw_data, hw_rows, hw_cols = [], [], []
        ho_data, ho_rows, ho_cols = [], [], []
        wo_data, wo_rows, wo_cols = [], [], []

        for user_id in responses.keys():
            H_locs = user_locs[user_id]['H']
            W_locs = user_locs[user_id]['W']
            O_locs = user_locs[user_id]['O']

            # HW interactions
            if H_locs and W_locs:
                self._add_pairwise(H_locs, W_locs, top_k, grid_size,
                                  hw_data, hw_rows, hw_cols)
            # HO interactions
            if H_locs and O_locs:
                self._add_pairwise(H_locs, O_locs, top_k, grid_size,
                                  ho_data, ho_rows, ho_cols)
            # WO interactions
            if W_locs and O_locs:
                self._add_pairwise(W_locs, O_locs, top_k, grid_size,
                                  wo_data, wo_rows, wo_cols)

        # Build sparse matrices
        HW = sparse.csr_matrix((hw_data, (hw_rows, hw_cols)),
                               shape=(grid_size, grid_size)) if hw_data else \
             sparse.csr_matrix((grid_size, grid_size))
        HO = sparse.csr_matrix((ho_data, (ho_rows, ho_cols)),
                               shape=(grid_size, grid_size)) if ho_data else \
             sparse.csr_matrix((grid_size, grid_size))
        WO = sparse.csr_matrix((wo_data, (wo_rows, wo_cols)),
                               shape=(grid_size, grid_size)) if wo_data else \
             sparse.csr_matrix((grid_size, grid_size))

        result = InteractionConstraints(HW=HW, HO=HO, WO=WO)
        result.normalize()
        return result

    def _add_pairwise(self, locs1: List[np.ndarray], locs2: List[np.ndarray],
                      top_k: int, grid_size: int,
                      data: List, rows: List, cols: List):
        """Add pairwise interactions - VECTORIZED version"""
        # Sum distributions for each type
        q1 = np.sum(locs1, axis=0)
        q2 = np.sum(locs2, axis=0)

        # Get top-k indices
        if top_k < grid_size:
            idx1 = np.argpartition(q1, -top_k)[-top_k:]
            idx2 = np.argpartition(q2, -top_k)[-top_k:]
        else:
            idx1 = np.where(q1 > 1e-10)[0]
            idx2 = np.where(q2 > 1e-10)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            return

        # Get probabilities and normalize
        p1 = q1[idx1]
        p2 = q2[idx2]
        p1 = p1 / (p1.sum() + 1e-10)
        p2 = p2 / (p2.sum() + 1e-10)

        # VECTORIZED: Compute outer product and extract non-zero entries
        outer = np.outer(p1, p2)  # (len(idx1), len(idx2))
        mask = outer > 1e-15

        # Use meshgrid to get all index combinations
        i_mesh, j_mesh = np.meshgrid(idx1, idx2, indexing='ij')

        # Extract matching entries (numpy operations, no Python loop)
        data.extend(outer[mask].tolist())
        rows.extend(i_mesh[mask].tolist())
        cols.extend(j_mesh[mask].tolist())

    def _update_beta(self, potentials: DualPotentials,
                     gen: InteractionConstraints,
                     real: InteractionConstraints,
                     lr: float):
        """Update beta potentials via gradient ascent"""
        # Gradient = gen - real (negated for ascent)
        for pair, gen_mat, real_mat in [
            ('HW', gen.HW, real.HW),
            ('HO', gen.HO, real.HO),
            ('WO', gen.WO, real.WO)
        ]:
            if gen_mat.nnz > 0 or real_mat.nnz > 0:
                grad = gen_mat - real_mat
                potentials.update_beta(pair[0], pair[1], -grad, lr)

    def _compute_jsd(self, gen: SpatialConstraints,
                     real: SpatialConstraints) -> float:
        """Compute JSD for spatial distributions"""
        def jsd(p, q):
            p = p.flatten() + 1e-10
            q = q.flatten() + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p/m)) + np.sum(q * np.log(q/m)))

        return (jsd(gen.H, real.H) + jsd(gen.W, real.W) + jsd(gen.O, real.O)) / 3

    def _compute_interaction_jsd(self, gen: InteractionConstraints,
                                  real: InteractionConstraints) -> float:
        """Compute JSD for interaction distributions (fast version)"""
        def sparse_jsd(p, q):
            # Sample-based estimation for speed
            p_coo = p.tocoo()
            q_coo = q.tocoo()

            if p_coo.nnz == 0 and q_coo.nnz == 0:
                return 0.0

            # Get indices
            p_idx = set(zip(p_coo.row.tolist(), p_coo.col.tolist()))
            q_idx = set(zip(q_coo.row.tolist(), q_coo.col.tolist()))

            # Sample if too large
            all_idx = list(p_idx | q_idx)
            if len(all_idx) > 10000:
                all_idx = [all_idx[i] for i in
                          np.random.choice(len(all_idx), 10000, replace=False)]

            if not all_idx:
                return 0.0

            p_vals = np.array([p[r, c] for r, c in all_idx]) + 1e-10
            q_vals = np.array([q[r, c] for r, c in all_idx]) + 1e-10

            p_vals = p_vals / p_vals.sum()
            q_vals = q_vals / q_vals.sum()

            m = 0.5 * (p_vals + q_vals)
            return 0.5 * (np.sum(p_vals * np.log(p_vals/m)) +
                         np.sum(q_vals * np.log(q_vals/m)))

        losses = []
        if gen.HW.nnz > 0 or real.HW.nnz > 0:
            losses.append(sparse_jsd(gen.HW, real.HW))
        if gen.HO.nnz > 0 or real.HO.nnz > 0:
            losses.append(sparse_jsd(gen.HO, real.HO))
        if gen.WO.nnz > 0 or real.WO.nnz > 0:
            losses.append(sparse_jsd(gen.WO, real.WO))

        return np.mean(losses) if losses else 0.0


# Convenience function
def create_ssdmfo_v3(preset: str = 'default') -> SSDMFOv3:
    """Create SS-DMFO 3.0 optimizer with preset configurations"""
    if preset == 'fast':
        config = SSDMFO3Config(
            max_iter=50,
            batch_size=100,
            mfvi_iter=3,
            interaction_freq=10,
            top_k=30
        )
    elif preset == 'accurate':
        config = SSDMFO3Config(
            max_iter=200,
            batch_size=50,
            mfvi_iter=10,
            interaction_freq=3,
            top_k=100,
            temp_init=3.0,
            temp_final=0.3
        )
    else:  # default
        config = SSDMFO3Config()

    return SSDMFOv3(config)
