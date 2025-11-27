"""SS-DMFO 4.0 G-IPF Optimizer (Generalized Iterative Proportional Fitting)

Based on expert team recommendations:
1. Replace gradient descent (Adam) with G-IPF log-domain alternating projections
2. Use multiplicative updates (log-domain addition) instead of additive updates
3. Alternating projections: first satisfy spatial constraints, then interaction constraints
4. Keep SpMM/SDDMM sparse acceleration from SS-DMFO 3.0

Key insight: G-IPF guarantees convergence to global optimum (under entropy regularization)
and requires NO learning rate tuning.

Algorithm (Log-Domain G-IPF):
    α_new = α_old + T * (log μ_real - log μ_gen)
    β_new = β_old + T * (log π_real - log π_gen)

This is equivalent to multiplicative scaling in probability space:
    μ_new ∝ μ_old * (μ_real / μ_gen)

Author: SS-DMFO Team
Date: 2025-11-27
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from scipy import sparse
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. GPU acceleration disabled.")

from ..data.structures import (
    Constraints, UserPattern, Result,
    SpatialConstraints, InteractionConstraints
)
from ..baselines.base import BaseMethod


@dataclass
class GIPFConfig:
    """Configuration for G-IPF SS-DMFO 4.0"""
    # Optimization
    max_iter: int = 200
    tolerance: float = 1e-5

    # G-IPF specific - IMPORTANT: Use small damping for stability
    # The log-ratio updates can be very large, so we need small damping
    damping: float = 0.1           # Damping factor λ ∈ (0, 1] for stability
    alpha_damping: float = 0.1     # Conservative for spatial (was 0.8 - too aggressive!)
    beta_damping: float = 0.05     # Very conservative for interaction

    # Temperature (entropy regularization strength)
    temperature: float = 1.0       # Fixed temperature for G-IPF
    temp_anneal: bool = True       # Whether to anneal temperature
    temp_init: float = 1.0         # Lower initial temperature for stability
    temp_final: float = 0.5        # Final temperature if annealing

    # MFVI for micro response
    mfvi_iter: int = 5             # More iterations for better convergence
    mfvi_damping: float = 0.5

    # Gumbel noise for diversity
    gumbel_scale: float = 0.1      # Lower initial noise for G-IPF
    gumbel_decay: float = 0.99
    gumbel_final: float = 0.01

    # Batch sizes
    gpu_batch_size: int = 500
    sddmm_batch_size: int = 200

    # Update schedule
    spatial_first_iters: int = 20  # Pure spatial updates before introducing interaction
    interaction_freq: int = 1      # Update interaction every N iterations (G-IPF: every iter)
    gauss_seidel: bool = True      # Gauss-Seidel style: re-run MFVI between α and β updates
    freeze_alpha_in_phase2: bool = True  # Freeze alpha when optimizing interaction (prevents spatial degradation)

    # Logging
    log_freq: int = 10

    # Early stopping
    early_stop_patience: int = 30

    # Device
    device: str = 'cuda'
    dtype: str = 'float32'


class SupportSet:
    """Sparse support set S extracted from real constraints"""

    def __init__(self, real_interaction: InteractionConstraints, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        # Extract support sets with log values for G-IPF
        self.HW_coords, self.HW_values, self.HW_log = self._extract_support(real_interaction.HW)
        self.HO_coords, self.HO_values, self.HO_log = self._extract_support(real_interaction.HO)
        self.WO_coords, self.WO_values, self.WO_log = self._extract_support(real_interaction.WO)

        print(f"[SupportSet] Extracted:")
        print(f"  HW: {len(self.HW_coords[0])} positions")
        print(f"  HO: {len(self.HO_coords[0])} positions")
        print(f"  WO: {len(self.WO_coords[0])} positions")

    def _extract_support(self, sp_matrix: sparse.spmatrix) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Extract (row, col) coordinates, values, and log values"""
        coo = sp_matrix.tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64)).to(self.device)
        cols = torch.from_numpy(coo.col.astype(np.int64)).to(self.device)

        # Normalize values
        values = coo.data.astype(np.float32)
        values = values / (values.sum() + 1e-10)

        values_tensor = torch.from_numpy(values).to(self.device, self.dtype)
        log_values = torch.log(values_tensor + 1e-10)

        return (rows, cols), values_tensor, log_values


class GIPFPotentials:
    """Dual potentials for G-IPF with log-domain updates"""

    def __init__(self, grid_h: int, grid_w: int, support: SupportSet,
                 device: torch.device, dtype: torch.dtype):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_size = grid_h * grid_w
        self.device = device
        self.dtype = dtype
        self.support = support

        # First-order potentials (dense, initialized to 0)
        self.alpha_H = torch.zeros(self.grid_size, device=device, dtype=dtype)
        self.alpha_W = torch.zeros(self.grid_size, device=device, dtype=dtype)
        self.alpha_O = torch.zeros(self.grid_size, device=device, dtype=dtype)

        # Second-order potentials (sparse, on support set)
        self.beta_HW = torch.zeros(len(support.HW_coords[0]), device=device, dtype=dtype)
        self.beta_HO = torch.zeros(len(support.HO_coords[0]), device=device, dtype=dtype)
        self.beta_WO = torch.zeros(len(support.WO_coords[0]), device=device, dtype=dtype)

    def get_beta_sparse(self, pair: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get beta values and coordinates for a pair type"""
        if pair == 'HW':
            return self.support.HW_coords, self.beta_HW
        elif pair == 'HO':
            return self.support.HO_coords, self.beta_HO
        elif pair == 'WO':
            return self.support.WO_coords, self.beta_WO
        raise ValueError(f"Unknown pair: {pair}")

    def to_sparse_tensor(self, pair: str) -> torch.Tensor:
        """Convert beta to sparse tensor for SpMM"""
        coords, values = self.get_beta_sparse(pair)
        rows, cols = coords
        indices = torch.stack([rows, cols])
        return torch.sparse_coo_tensor(
            indices, values, (self.grid_size, self.grid_size),
            device=self.device, dtype=self.dtype
        ).coalesce()

    def copy_state(self) -> Dict:
        return {
            'alpha_H': self.alpha_H.clone(),
            'alpha_W': self.alpha_W.clone(),
            'alpha_O': self.alpha_O.clone(),
            'beta_HW': self.beta_HW.clone(),
            'beta_HO': self.beta_HO.clone(),
            'beta_WO': self.beta_WO.clone(),
        }

    def restore_state(self, state: Dict):
        self.alpha_H = state['alpha_H'].clone()
        self.alpha_W = state['alpha_W'].clone()
        self.alpha_O = state['alpha_O'].clone()
        self.beta_HW = state['beta_HW'].clone()
        self.beta_HO = state['beta_HO'].clone()
        self.beta_WO = state['beta_WO'].clone()


class BatchedUserData:
    """Preprocessed user data for GPU batch processing"""

    def __init__(self, user_patterns: Dict[int, UserPattern], device: torch.device):
        self.device = device
        self.user_ids = list(user_patterns.keys())
        self.n_users = len(self.user_ids)

        self.max_locs = max(len(u.locations) for u in user_patterns.values())

        self.loc_types = torch.full((self.n_users, self.max_locs), -1,
                                    dtype=torch.long, device=device)
        self.n_locs = torch.zeros(self.n_users, dtype=torch.long, device=device)

        type_map = {'H': 0, 'W': 1, 'O': 2}

        for u_idx, user_id in enumerate(self.user_ids):
            user = user_patterns[user_id]
            n = len(user.locations)
            self.n_locs[u_idx] = n
            for l_idx, loc in enumerate(user.locations):
                self.loc_types[u_idx, l_idx] = type_map[loc.type]

        self.H_mask = (self.loc_types == 0)
        self.W_mask = (self.loc_types == 1)
        self.O_mask = (self.loc_types == 2)
        self.valid_mask = (self.loc_types >= 0)


class SSDMFOGIPF(BaseMethod):
    """SS-DMFO 4.0 with G-IPF (Generalized Iterative Proportional Fitting)

    Key innovations:
    1. Log-domain multiplicative updates (Sinkhorn-style)
    2. Alternating projections between spatial and interaction constraints
    3. Damping for stability in complex MMOT problems
    4. No learning rate tuning required
    """

    def __init__(self, config: Optional[GIPFConfig] = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required. Install with: pip install torch")

        self.config = config or GIPFConfig()
        super().__init__("SS-DMFO-GIPF")

        if self.config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[G-IPF] Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("[G-IPF] Using CPU")

        self.dtype = torch.float32 if self.config.dtype == 'float32' else torch.float64

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """Main optimization with G-IPF"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w
        n_users = len(user_patterns)

        # Normalize constraints
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        print(f"[SS-DMFO 4.0 G-IPF] Initializing...")
        print(f"  Device: {self.device}")
        print(f"  Users: {n_users}, Grid: {grid_h}x{grid_w}={grid_size}")
        print(f"  Damping: α={self.config.alpha_damping}, β={self.config.beta_damping}")

        # Move spatial constraints to GPU with log values
        target_H = torch.from_numpy(constraints.spatial.H.flatten()).to(self.device, self.dtype)
        target_W = torch.from_numpy(constraints.spatial.W.flatten()).to(self.device, self.dtype)
        target_O = torch.from_numpy(constraints.spatial.O.flatten()).to(self.device, self.dtype)

        # Normalize
        target_H = target_H / (target_H.sum() + 1e-10)
        target_W = target_W / (target_W.sum() + 1e-10)
        target_O = target_O / (target_O.sum() + 1e-10)

        # Log values for G-IPF updates
        log_target_H = torch.log(target_H + 1e-10)
        log_target_W = torch.log(target_W + 1e-10)
        log_target_O = torch.log(target_O + 1e-10)

        # Extract support set
        if constraints.interaction is None:
            raise ValueError("Interaction constraints required for G-IPF")

        support = SupportSet(constraints.interaction, self.device, self.dtype)

        # Initialize potentials
        potentials = GIPFPotentials(grid_h, grid_w, support, self.device, self.dtype)

        # Preprocess user data
        print(f"[G-IPF] Preprocessing user data...")
        gpu_batch_size = self.config.gpu_batch_size
        user_id_list = list(user_patterns.keys())
        n_batches = (n_users + gpu_batch_size - 1) // gpu_batch_size

        user_data_batches = []
        for batch_idx in range(n_batches):
            start = batch_idx * gpu_batch_size
            end = min(start + gpu_batch_size, n_users)
            batch_ids = user_id_list[start:end]
            batch_patterns = {uid: user_patterns[uid] for uid in batch_ids}
            user_data_batches.append(BatchedUserData(batch_patterns, self.device))

        print(f"  GPU batch size: {gpu_batch_size}, num batches: {n_batches}")
        print(f"  Max locations per user: {max(b.max_locs for b in user_data_batches)}")

        # Early stopping
        best_total_loss = float('inf')
        best_state = None
        no_improve_count = 0
        best_iter = 0

        # History for monitoring
        spatial_history = []
        interaction_history = []

        print(f"\n[G-IPF] Starting optimization (max_iter={self.config.max_iter})...")
        print(f"  Pure spatial phase: first {self.config.spatial_first_iters} iterations")
        print(f"  Freeze alpha in phase 2: {self.config.freeze_alpha_in_phase2}")
        print(f"  Gauss-Seidel updates: {self.config.gauss_seidel}")

        last_spatial_loss = 0.0
        last_interaction_loss = 0.0

        for iteration in range(self.config.max_iter):
            iter_start = time.time()

            # Phase control
            use_beta = iteration >= self.config.spatial_first_iters

            # Temperature annealing
            if self.config.temp_anneal:
                progress = iteration / max(self.config.max_iter - 1, 1)
                temperature = self.config.temp_init * (
                    self.config.temp_final / self.config.temp_init
                ) ** progress
            else:
                temperature = self.config.temperature

            # Gumbel noise decay
            gumbel_scale = max(
                self.config.gumbel_final,
                self.config.gumbel_scale * (self.config.gumbel_decay ** iteration)
            )

            # ============================================
            # STEP 1: FORWARD PASS (Micro Response)
            # ============================================
            gen_H = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
            gen_W = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
            gen_O = torch.zeros(grid_size, device=self.device, dtype=self.dtype)

            compute_interaction = use_beta and (iteration % self.config.interaction_freq == 0)

            if compute_interaction:
                HW_gen = torch.zeros(len(support.HW_coords[0]), device=self.device, dtype=self.dtype)
                HO_gen = torch.zeros(len(support.HO_coords[0]), device=self.device, dtype=self.dtype)
                WO_gen = torch.zeros(len(support.WO_coords[0]), device=self.device, dtype=self.dtype)

            for user_data in user_data_batches:
                Q = self._batch_forward(
                    user_data, potentials, grid_size,
                    temperature, gumbel_scale, use_beta
                )

                # Aggregate spatial
                batch_H, batch_W, batch_O = self._aggregate_spatial(Q, user_data)
                gen_H += batch_H
                gen_W += batch_W
                gen_O += batch_O

                # SDDMM for interaction
                if compute_interaction:
                    hw_batch, ho_batch, wo_batch = self._sddmm_aggregate(Q, user_data, support)
                    HW_gen += hw_batch
                    HO_gen += ho_batch
                    WO_gen += wo_batch

                del Q
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Normalize spatial
            gen_H = gen_H / (gen_H.sum() + 1e-10)
            gen_W = gen_W / (gen_W.sum() + 1e-10)
            gen_O = gen_O / (gen_O.sum() + 1e-10)

            # Log of generated distribution
            log_gen_H = torch.log(gen_H + 1e-10)
            log_gen_W = torch.log(gen_W + 1e-10)
            log_gen_O = torch.log(gen_O + 1e-10)

            # Compute spatial JSD
            last_spatial_loss = self._compute_jsd(gen_H, target_H, gen_W, target_W, gen_O, target_O)
            spatial_history.append(last_spatial_loss)

            # ============================================
            # STEP 2: G-IPF ALPHA UPDATE (Spatial Projection)
            # ============================================
            # Since Q ∝ exp(-α / T), higher α = lower probability
            # To increase probability where gen < target, we need to DECREASE α
            # Correct for Q ∝ exp(-α/T): α_new = α_old - T * (log μ_real - log μ_gen)
            # When gen < target: log_target - log_gen > 0, so we subtract → α decreases → Q increases ✓

            # Option to freeze alpha in phase 2 to prevent spatial degradation
            should_update_alpha = not use_beta or not self.config.freeze_alpha_in_phase2

            if should_update_alpha:
                alpha_update_H = temperature * (log_target_H - log_gen_H)
                alpha_update_W = temperature * (log_target_W - log_gen_W)
                alpha_update_O = temperature * (log_target_O - log_gen_O)

                damping = self.config.alpha_damping
                # SUBTRACT because Q ∝ exp(-α/T)
                potentials.alpha_H = potentials.alpha_H - damping * alpha_update_H
                potentials.alpha_W = potentials.alpha_W - damping * alpha_update_W
                potentials.alpha_O = potentials.alpha_O - damping * alpha_update_O

            # ============================================
            # STEP 3: GAUSS-SEIDEL RE-COMPUTATION (Optional)
            # ============================================
            # Re-run forward pass after alpha update before beta update
            if self.config.gauss_seidel and compute_interaction:
                gen_H = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
                gen_W = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
                gen_O = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
                HW_gen = torch.zeros(len(support.HW_coords[0]), device=self.device, dtype=self.dtype)
                HO_gen = torch.zeros(len(support.HO_coords[0]), device=self.device, dtype=self.dtype)
                WO_gen = torch.zeros(len(support.WO_coords[0]), device=self.device, dtype=self.dtype)

                for user_data in user_data_batches:
                    Q = self._batch_forward(
                        user_data, potentials, grid_size,
                        temperature, gumbel_scale, use_beta
                    )

                    batch_H, batch_W, batch_O = self._aggregate_spatial(Q, user_data)
                    gen_H += batch_H
                    gen_W += batch_W
                    gen_O += batch_O

                    hw_batch, ho_batch, wo_batch = self._sddmm_aggregate(Q, user_data, support)
                    HW_gen += hw_batch
                    HO_gen += ho_batch
                    WO_gen += wo_batch

                    del Q
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            # ============================================
            # STEP 4: G-IPF BETA UPDATE (Interaction Projection)
            # ============================================
            if compute_interaction:
                # Normalize
                HW_gen = HW_gen / (HW_gen.sum() + 1e-10)
                HO_gen = HO_gen / (HO_gen.sum() + 1e-10)
                WO_gen = WO_gen / (WO_gen.sum() + 1e-10)

                # Log of generated
                log_HW_gen = torch.log(HW_gen + 1e-10)
                log_HO_gen = torch.log(HO_gen + 1e-10)
                log_WO_gen = torch.log(WO_gen + 1e-10)

                # Compute interaction JSD
                last_interaction_loss = self._interaction_jsd_sparse(HW_gen, HO_gen, WO_gen, support)
                interaction_history.append(last_interaction_loss)

                # G-IPF update for beta
                # Beta is used in MFVI as: field += β·mean_field, then Q ∝ exp(-field/T)
                # Higher β at (h,w) → higher field → lower Q (similar to alpha)
                # So when gen < real at (h,w), we want β to DECREASE to encourage co-occurrence
                # β_new = β_old - T * (log π_real - log π_gen)
                beta_damping = self.config.beta_damping

                beta_update_HW = temperature * (support.HW_log - log_HW_gen)
                beta_update_HO = temperature * (support.HO_log - log_HO_gen)
                beta_update_WO = temperature * (support.WO_log - log_WO_gen)

                # SUBTRACT because higher β → lower probability (like alpha)
                potentials.beta_HW = potentials.beta_HW - beta_damping * beta_update_HW
                potentials.beta_HO = potentials.beta_HO - beta_damping * beta_update_HO
                potentials.beta_WO = potentials.beta_WO - beta_damping * beta_update_WO

                # Early stopping based on total loss
                total_loss = last_spatial_loss + 0.5 * last_interaction_loss
                if total_loss < best_total_loss - 0.001:
                    best_total_loss = total_loss
                    best_state = potentials.copy_state()
                    no_improve_count = 0
                    best_iter = iteration
                else:
                    no_improve_count += 1
            else:
                # Pure spatial phase
                if last_spatial_loss < best_total_loss - 0.001:
                    best_total_loss = last_spatial_loss
                    best_state = potentials.copy_state()
                    no_improve_count = 0
                    best_iter = iteration

            # Logging
            iter_time = time.time() - iter_start
            if not use_beta:
                phase_str = "Spatial"
            elif self.config.freeze_alpha_in_phase2:
                phase_str = "Interact"  # Alpha frozen, only optimizing interaction
            else:
                phase_str = "Full"

            if iteration % self.config.log_freq == 0 or iteration < 5:
                interact_str = f"{last_interaction_loss:.4f}" if last_interaction_loss > 0 else "---"
                print(f"  Iter {iteration:3d} [{phase_str}]: Spatial={last_spatial_loss:.4f}, "
                      f"Interact={interact_str}, T={temperature:.2f}, "
                      f"Gumbel={gumbel_scale:.3f} ({iter_time:.1f}s)")

            # Early stopping
            if use_beta and no_improve_count >= self.config.early_stop_patience:
                print(f"  Early stopping at iter {iteration}")
                if best_state is not None:
                    potentials.restore_state(best_state)
                break

            # Convergence check
            if last_spatial_loss < self.config.tolerance:
                print(f"  Spatial converged at iter {iteration}")

        # ============================================
        # FINAL PASS: Multiple Imputation
        # ============================================
        print(f"\n[G-IPF] Computing final allocations...")
        print(f"  Best total loss: {best_total_loss:.4f} at iter {best_iter}")
        print(f"  Final Spatial: {last_spatial_loss:.4f}, Interaction: {last_interaction_loss:.4f}")

        # Restore best state
        if best_state is not None:
            potentials.restore_state(best_state)

        # Multiple imputation for stable statistics
        n_samples = 10
        final_responses = {}

        # Use lower temperature for sharper final distributions
        final_temp = self.config.temp_final if self.config.temp_anneal else self.config.temperature * 0.5

        for user_data in user_data_batches:
            Q_sum = None

            for sample_idx in range(n_samples):
                Q_sample = self._batch_forward(
                    user_data, potentials, grid_size,
                    temperature=final_temp,
                    gumbel_scale=self.config.gumbel_final,
                    use_beta=False  # Skip MFVI for final pass
                )

                if Q_sum is None:
                    Q_sum = Q_sample
                else:
                    Q_sum = Q_sum + Q_sample

                del Q_sample

            # Average and normalize
            Q_final = Q_sum / n_samples
            Q_final = Q_final / (Q_final.sum(dim=-1, keepdim=True) + 1e-10)

            Q_np = Q_final.cpu().numpy()
            for u_idx, user_id in enumerate(user_data.user_ids):
                n = user_data.n_locs[u_idx].item()
                final_responses[user_id] = Q_np[u_idx, :n, :]

            del Q_final, Q_sum

        return final_responses

    def _batch_forward(self,
                       user_data: BatchedUserData,
                       potentials: GIPFPotentials,
                       grid_size: int,
                       temperature: float,
                       gumbel_scale: float,
                       use_beta: bool) -> torch.Tensor:
        """Batch forward pass with Gumbel noise"""
        n_users = user_data.n_users
        max_locs = user_data.max_locs
        inv_temp = 1.0 / temperature

        # Stack alpha for vectorized lookup
        alpha_stack = torch.stack([potentials.alpha_H, potentials.alpha_W, potentials.alpha_O])

        loc_types_clamped = user_data.loc_types.clamp(min=0)
        alpha_per_loc = alpha_stack[loc_types_clamped]

        # Gumbel noise for diversity
        if gumbel_scale > 1e-6:
            gumbel = torch.distributions.Gumbel(0, gumbel_scale).sample(
                (n_users, max_locs, grid_size)
            ).to(self.device, self.dtype)
            log_Q = -alpha_per_loc * inv_temp + gumbel
        else:
            log_Q = -alpha_per_loc * inv_temp

        Q = F.softmax(log_Q, dim=-1)
        Q = Q * user_data.valid_mask.unsqueeze(-1).float()

        # MFVI with SpMM
        if use_beta and self.config.mfvi_iter > 0:
            Q = self._mfvi_spmm(Q, user_data, potentials, alpha_per_loc, temperature, gumbel_scale)

        return Q

    def _mfvi_spmm(self,
                   Q: torch.Tensor,
                   user_data: BatchedUserData,
                   potentials: GIPFPotentials,
                   alpha_per_loc: torch.Tensor,
                   temperature: float,
                   gumbel_scale: float) -> torch.Tensor:
        """MFVI using SpMM for local field computation"""
        n_users, max_locs, grid_size = Q.shape
        damping = self.config.mfvi_damping
        inv_temp = 1.0 / temperature

        # Get sparse beta tensors
        beta_HW = potentials.to_sparse_tensor('HW')
        beta_HO = potentials.to_sparse_tensor('HO')
        beta_WO = potentials.to_sparse_tensor('WO')

        has_HW = beta_HW._nnz() > 0
        has_HO = beta_HO._nnz() > 0
        has_WO = beta_WO._nnz() > 0

        if not (has_HW or has_HO or has_WO):
            return Q

        # Pre-compute masks
        H_mask_float = user_data.H_mask.unsqueeze(-1).float()
        W_mask_float = user_data.W_mask.unsqueeze(-1).float()
        O_mask_float = user_data.O_mask.unsqueeze(-1).float()
        valid_mask_float = user_data.valid_mask.unsqueeze(-1).float()

        for mfvi_iter in range(self.config.mfvi_iter):
            noise_scale = gumbel_scale * (0.5 ** mfvi_iter)

            # Sum Q by type
            H_sum = (Q * H_mask_float).sum(dim=1)
            W_sum = (Q * W_mask_float).sum(dim=1)
            O_sum = (Q * O_mask_float).sum(dim=1)

            # Compute field via SpMM
            field = alpha_per_loc.clone()

            if has_HW:
                HW_contrib = torch.sparse.mm(beta_HW, W_sum.T).T
                field = field + H_mask_float * HW_contrib.unsqueeze(1)
                del HW_contrib
                WH_contrib = torch.sparse.mm(beta_HW.T, H_sum.T).T
                field = field + W_mask_float * WH_contrib.unsqueeze(1)
                del WH_contrib

            if has_HO:
                HO_contrib = torch.sparse.mm(beta_HO, O_sum.T).T
                field = field + H_mask_float * HO_contrib.unsqueeze(1)
                del HO_contrib
                OH_contrib = torch.sparse.mm(beta_HO.T, H_sum.T).T
                field = field + O_mask_float * OH_contrib.unsqueeze(1)
                del OH_contrib

            if has_WO:
                WO_contrib = torch.sparse.mm(beta_WO, O_sum.T).T
                field = field + W_mask_float * WO_contrib.unsqueeze(1)
                del WO_contrib
                OW_contrib = torch.sparse.mm(beta_WO.T, W_sum.T).T
                field = field + O_mask_float * OW_contrib.unsqueeze(1)
                del OW_contrib

            del H_sum, W_sum, O_sum

            # Gumbel noise
            if noise_scale > 1e-6:
                gumbel = torch.distributions.Gumbel(0, noise_scale).sample(
                    (n_users, max_locs, grid_size)
                ).to(self.device, self.dtype)
                log_Q = -field * inv_temp + gumbel
                del gumbel
            else:
                log_Q = -field * inv_temp
            del field

            Q_new = F.softmax(log_Q, dim=-1)
            del log_Q

            # Damped update
            Q = damping * Q_new + (1 - damping) * Q
            del Q_new
            Q = Q * valid_mask_float

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return Q

    def _aggregate_spatial(self, Q: torch.Tensor, user_data: BatchedUserData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate spatial statistics"""
        gen_H = (Q * user_data.H_mask.unsqueeze(-1).float()).sum(dim=(0, 1))
        gen_W = (Q * user_data.W_mask.unsqueeze(-1).float()).sum(dim=(0, 1))
        gen_O = (Q * user_data.O_mask.unsqueeze(-1).float()).sum(dim=(0, 1))
        return gen_H, gen_W, gen_O

    def _sddmm_aggregate(self, Q: torch.Tensor, user_data: BatchedUserData,
                         support: SupportSet) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SDDMM: Compute π_gen only at support positions"""
        n_users = user_data.n_users
        mini_batch = self.config.sddmm_batch_size

        # Aggregate Q by type
        H_agg = (Q * user_data.H_mask.unsqueeze(-1).float()).sum(dim=1)
        W_agg = (Q * user_data.W_mask.unsqueeze(-1).float()).sum(dim=1)
        O_agg = (Q * user_data.O_mask.unsqueeze(-1).float()).sum(dim=1)

        # Normalize per user
        H_agg = H_agg / (H_agg.sum(dim=1, keepdim=True) + 1e-10)
        W_agg = W_agg / (W_agg.sum(dim=1, keepdim=True) + 1e-10)
        O_agg = O_agg / (O_agg.sum(dim=1, keepdim=True) + 1e-10)

        # Support coordinates
        HW_rows, HW_cols = support.HW_coords
        HO_rows, HO_cols = support.HO_coords
        WO_rows, WO_cols = support.WO_coords

        HW_gen = torch.zeros(len(HW_rows), device=self.device, dtype=self.dtype)
        HO_gen = torch.zeros(len(HO_rows), device=self.device, dtype=self.dtype)
        WO_gen = torch.zeros(len(WO_rows), device=self.device, dtype=self.dtype)

        # Process in mini-batches
        n_mini_batches = (n_users + mini_batch - 1) // mini_batch

        for mb in range(n_mini_batches):
            start = mb * mini_batch
            end = min(start + mini_batch, n_users)

            H_batch = H_agg[start:end]
            W_batch = W_agg[start:end]
            O_batch = O_agg[start:end]

            # SDDMM
            HW_gen += (H_batch[:, HW_rows] * W_batch[:, HW_cols]).sum(dim=0)
            HO_gen += (H_batch[:, HO_rows] * O_batch[:, HO_cols]).sum(dim=0)
            WO_gen += (W_batch[:, WO_rows] * O_batch[:, WO_cols]).sum(dim=0)

            del H_batch, W_batch, O_batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Average
        HW_gen = HW_gen / n_users
        HO_gen = HO_gen / n_users
        WO_gen = WO_gen / n_users

        del H_agg, W_agg, O_agg

        return HW_gen, HO_gen, WO_gen

    def _compute_jsd(self, gen_H, target_H, gen_W, target_W, gen_O, target_O) -> float:
        """Compute spatial JSD"""
        def jsd(p, q):
            p = p + 1e-10
            q = q + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            m = 0.5 * (p + q)
            return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))

        jsd_H = jsd(gen_H, target_H)
        jsd_W = jsd(gen_W, target_W)
        jsd_O = jsd(gen_O, target_O)

        return ((jsd_H + jsd_W + jsd_O) / 3).item()

    def _interaction_jsd_sparse(self, HW_gen: torch.Tensor, HO_gen: torch.Tensor,
                                WO_gen: torch.Tensor, support: SupportSet) -> float:
        """Compute JSD on support set"""
        def sparse_jsd(gen, real):
            gen = gen + 1e-10
            real = real + 1e-10
            gen = gen / gen.sum()
            real = real / real.sum()
            m = 0.5 * (gen + real)
            return 0.5 * (torch.sum(gen * torch.log(gen / m)) + torch.sum(real * torch.log(real / m)))

        losses = []
        if len(HW_gen) > 0:
            losses.append(sparse_jsd(HW_gen, support.HW_values).item())
        if len(HO_gen) > 0:
            losses.append(sparse_jsd(HO_gen, support.HO_values).item())
        if len(WO_gen) > 0:
            losses.append(sparse_jsd(WO_gen, support.WO_values).item())

        return np.mean(losses) if losses else 0.0


def create_ssdmfo_gipf(preset: str = 'default') -> SSDMFOGIPF:
    """Create G-IPF SS-DMFO optimizer"""
    if preset == 'fast':
        config = GIPFConfig(
            max_iter=100,
            gpu_batch_size=2000,
            mfvi_iter=3,
            spatial_first_iters=10
        )
    elif preset == 'accurate':
        config = GIPFConfig(
            max_iter=300,
            gpu_batch_size=500,
            mfvi_iter=5,
            spatial_first_iters=30,
            damping=0.8,
            temp_init=2.0,
            temp_final=0.3
        )
    elif preset == 'stable':
        # More conservative settings for stability
        config = GIPFConfig(
            max_iter=200,
            alpha_damping=0.5,      # More conservative
            beta_damping=0.3,       # Much more conservative
            mfvi_iter=5,
            gauss_seidel=True,
            temp_anneal=True,
            temp_init=3.0,
            temp_final=0.5
        )
    else:
        config = GIPFConfig()

    return SSDMFOGIPF(config)
