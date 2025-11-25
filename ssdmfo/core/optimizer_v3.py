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

    # Temperature annealing
    temp_init: float = 2.0      # High temperature (exploration)
    temp_final: float = 0.5     # Low temperature (exploitation)
    temp_decay: str = 'exponential'  # 'linear' or 'exponential'

    # Gumbel noise
    gumbel_scale: float = 0.1   # Scale of Gumbel noise
    gumbel_decay: float = 0.99  # Decay per iteration

    # Stochastic optimization
    batch_size: int = 50        # Users per batch

    # Logging
    log_freq: int = 10

    # Interaction
    interaction_freq: int = 5   # Compute interaction every N iters
    top_k: int = 50             # Top-k for interaction (increased from 30)


class MeanFieldSolverV3:
    """MFVI Solver V3 - With individual responses and Gumbel noise"""

    def __init__(self, config: SSDMFO3Config):
        self.config = config

    def compute_user_response_individual(self,
                                         user: UserPattern,
                                         potentials: DualPotentials,
                                         grid_size: int,
                                         temperature: float,
                                         gumbel_scale: float,
                                         use_beta: bool = True) -> np.ndarray:
        """Compute individual response Q_i for a single user

        Key differences from V2:
        1. Each user gets INDEPENDENT Gumbel noise
        2. Iterative MFVI with beta coupling
        3. No template sharing

        Args:
            user: User pattern
            potentials: Current potentials
            grid_size: Number of grid cells
            temperature: Current temperature
            gumbel_scale: Scale of Gumbel noise
            use_beta: Whether to use beta (interaction) potentials

        Returns:
            Q: (n_locations, grid_size) - individual distribution for this user
        """
        n_locs = len(user.locations)
        loc_types = [loc.type for loc in user.locations]

        # Get alpha potentials (flattened)
        alpha_flat = {
            'H': potentials.alpha_H.flatten(),
            'W': potentials.alpha_W.flatten(),
            'O': potentials.alpha_O.flatten()
        }

        # Initialize Q with Gumbel noise (KEY: individual noise per user)
        Q = np.zeros((n_locs, grid_size))
        for i, lt in enumerate(loc_types):
            # Gumbel noise for diversity (different for each user!)
            gumbel = np.random.gumbel(0, gumbel_scale, grid_size)
            log_q = -alpha_flat[lt] / temperature + gumbel
            log_q -= log_q.max()
            q = np.exp(log_q)
            q /= q.sum() + 1e-10
            Q[i] = q

        # Iterative MFVI with beta coupling
        if use_beta and self.config.mfvi_iter > 0:
            Q = self._iterative_mfvi(
                Q, loc_types, potentials, alpha_flat,
                grid_size, temperature, gumbel_scale
            )

        return Q

    def _iterative_mfvi(self,
                        Q: np.ndarray,
                        loc_types: List[str],
                        potentials: DualPotentials,
                        alpha_flat: Dict[str, np.ndarray],
                        grid_size: int,
                        temperature: float,
                        gumbel_scale: float) -> np.ndarray:
        """Iterative MFVI with beta coupling

        Update rule:
        Q(l) <- softmax((-alpha_c - sum_{l'} beta_{cc'} @ Q(l') + gumbel) / T)
        """
        n_locs = len(loc_types)

        for mfvi_iter in range(self.config.mfvi_iter):
            Q_old = Q.copy()

            for i in range(n_locs):
                loc_type = loc_types[i]

                # Start with alpha field
                field = alpha_flat[loc_type].copy()

                # Add beta interaction with other locations
                for j in range(n_locs):
                    if i == j:
                        continue
                    other_type = loc_types[j]
                    beta = potentials.get_beta(loc_type, other_type)
                    if beta is not None and beta.nnz > 0:
                        # Sparse matrix-vector multiplication
                        field += beta.dot(Q_old[j])

                # Add Gumbel noise (decreasing with MFVI iterations)
                noise_scale = gumbel_scale * (0.5 ** mfvi_iter)
                gumbel = np.random.gumbel(0, noise_scale, grid_size)

                # Softmax with temperature
                log_q = (-field + gumbel) / temperature
                log_q -= log_q.max()
                q_new = np.exp(log_q)
                q_new /= q_new.sum() + 1e-10

                # Damped update
                Q[i] = self.config.mfvi_damping * q_new + (1 - self.config.mfvi_damping) * Q_old[i]

        return Q

    def compute_batch_responses(self,
                                user_batch: Dict[int, UserPattern],
                                potentials: DualPotentials,
                                grid_size: int,
                                temperature: float,
                                gumbel_scale: float,
                                use_beta: bool = True) -> Dict[int, np.ndarray]:
        """Compute responses for a batch of users"""
        responses = {}
        for user_id, user in user_batch.items():
            Q = self.compute_user_response_individual(
                user, potentials, grid_size,
                temperature, gumbel_scale, use_beta
            )
            responses[user_id] = Q
        return responses


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
        print(f"  Gumbel scale: {self.config.gumbel_scale}")

        potentials = DualPotentials.initialize(grid_h, grid_w, phase=2)
        optimizer = PotentialsWithMomentum(potentials)

        # User list for batching
        user_ids = list(user_patterns.keys())
        n_batches = (n_users + self.config.batch_size - 1) // self.config.batch_size

        # Optimization loop
        print(f"\n[SS-DMFO 3.0] Starting optimization (max_iter={self.config.max_iter})...")
        prev_loss = float('inf')
        gumbel_scale = self.config.gumbel_scale

        for iteration in range(self.config.max_iter):
            iter_start = time.time()

            # Temperature annealing
            if self.config.temp_decay == 'exponential':
                progress = iteration / max(self.config.max_iter - 1, 1)
                temperature = self.config.temp_init * (
                    self.config.temp_final / self.config.temp_init
                ) ** progress
            else:  # linear
                temperature = self.config.temp_init - (
                    self.config.temp_init - self.config.temp_final
                ) * iteration / max(self.config.max_iter - 1, 1)

            # Gumbel noise decay
            gumbel_scale = self.config.gumbel_scale * (self.config.gumbel_decay ** iteration)

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
            use_beta = (iteration >= 5)  # Start using beta after warmup

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

            # Compute interaction (less frequently)
            interaction_loss = 0.0
            if constraints.interaction is not None and iteration % self.config.interaction_freq == 0:
                gen_interaction = self._aggregate_interaction_fast(
                    all_responses, user_patterns, grid_size
                )
                interaction_loss = self._compute_interaction_jsd(
                    gen_interaction, constraints.interaction
                )

                # Update beta potentials
                if use_beta:
                    self._update_beta(potentials, gen_interaction,
                                     constraints.interaction, self.config.lr_beta)

            total_loss = spatial_loss + 0.5 * interaction_loss

            # Update alpha potentials
            optimizer.step(spatial_grads, self.config.lr_alpha)

            # Logging
            iter_time = time.time() - iter_start
            if iteration % self.config.log_freq == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: Spatial={spatial_loss:.4f}, "
                      f"Interact={interaction_loss:.4f}, T={temperature:.2f}, "
                      f"Gumbel={gumbel_scale:.3f} ({iter_time:.1f}s)")

            # Convergence check
            if abs(prev_loss - total_loss) < self.config.tolerance and iteration > 20:
                print(f"  Converged at iteration {iteration}")
                break
            prev_loss = total_loss

        # Final pass to get responses
        print("\n[SS-DMFO 3.0] Computing final allocations...")
        final_responses = {}
        for user_id, user in user_patterns.items():
            Q = self.mf_solver.compute_user_response_individual(
                user, potentials, grid_size,
                self.config.temp_final, 0.0, use_beta=True
            )
            final_responses[user_id] = Q

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
        """Add pairwise interactions between two location sets"""
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

        # Compute outer product entries
        for i, i1 in enumerate(idx1):
            for j, i2 in enumerate(idx2):
                val = p1[i] * p2[j]
                if val > 1e-15:
                    rows.append(i1)
                    cols.append(i2)
                    data.append(val)

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
