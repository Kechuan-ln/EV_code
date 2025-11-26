"""SS-DMFO 3.0 Sparse Optimizer

Based on expert team recommendations:
1. SDDMM (Sampled Dense-Dense Matrix Multiplication) for outer loop
2. SpMM (Sparse Matrix-Dense Matrix Multiplication) for inner loop (MFVI)
3. Use real constraint support set S instead of top_k approximation

Key insight: We only need to compute π_gen at positions where π_real is non-zero.
This reduces complexity from O(N·G²) to O(N·|S|), where |S| ≈ 1M << G² ≈ 1.7B

Author: SS-DMFO Team
Date: 2025-11-26
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
class SparseConfig:
    """Configuration for Sparse SS-DMFO 3.0"""
    # Optimization
    max_iter: int = 100
    lr_alpha: float = 0.1
    lr_beta: float = 0.05
    tolerance: float = 1e-4

    # MFVI
    mfvi_iter: int = 5
    mfvi_damping: float = 0.5

    # Temperature annealing
    temp_init: float = 2.0
    temp_final: float = 0.5

    # Gumbel noise (critical for diversity)
    gumbel_scale: float = 0.3
    gumbel_decay: float = 0.995
    gumbel_final: float = 0.05

    # Batch size for GPU processing (users per batch in forward pass)
    gpu_batch_size: int = 500  # Reduced to avoid OOM

    # Mini-batch for SDDMM (users per mini-batch for interaction)
    sddmm_batch_size: int = 200  # Small to fit support set indexing

    # Logging
    log_freq: int = 5

    # Phase control
    interaction_freq: int = 2
    phase_separation: bool = True
    phase1_ratio: float = 0.2

    # Early stopping
    early_stop_patience: int = 15

    # GPU settings
    device: str = 'cuda'
    dtype: str = 'float32'


class SupportSet:
    """Sparse support set S extracted from real constraints

    Contains the (row, col) coordinates where real interaction > 0.
    All optimization happens only on these coordinates.
    """

    def __init__(self, real_interaction: InteractionConstraints, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        # Extract support sets from sparse matrices
        self.HW_coords, self.HW_values = self._extract_support(real_interaction.HW)
        self.HO_coords, self.HO_values = self._extract_support(real_interaction.HO)
        self.WO_coords, self.WO_values = self._extract_support(real_interaction.WO)

        print(f"[SupportSet] Extracted:")
        print(f"  HW: {len(self.HW_coords[0])} positions")
        print(f"  HO: {len(self.HO_coords[0])} positions")
        print(f"  WO: {len(self.WO_coords[0])} positions")

    def _extract_support(self, sp_matrix: sparse.spmatrix) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract (row, col) coordinates and values from sparse matrix"""
        coo = sp_matrix.tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64)).to(self.device)
        cols = torch.from_numpy(coo.col.astype(np.int64)).to(self.device)
        values = torch.from_numpy(coo.data.astype(np.float32 if self.dtype == torch.float32 else np.float64)).to(self.device)
        return (rows, cols), values


class SparsePotentials:
    """Dual potentials with sparse beta on support set S"""

    def __init__(self, grid_h: int, grid_w: int, support: SupportSet,
                 device: torch.device, dtype: torch.dtype):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_size = grid_h * grid_w
        self.device = device
        self.dtype = dtype
        self.support = support

        # First-order potentials (dense, on GPU)
        self.alpha_H = torch.zeros(self.grid_size, device=device, dtype=dtype)
        self.alpha_W = torch.zeros(self.grid_size, device=device, dtype=dtype)
        self.alpha_O = torch.zeros(self.grid_size, device=device, dtype=dtype)

        # Second-order potentials (sparse, same structure as support set)
        # Initialize to zero on support positions
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


class SparseAdamOptimizer:
    """Adam optimizer supporting both dense (alpha) and sparse (beta) updates"""

    def __init__(self, potentials: SparsePotentials, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.potentials = potentials
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        device = potentials.device
        dtype = potentials.dtype
        grid_size = potentials.grid_size

        # Dense moments for alpha
        self.m_alpha = {
            'H': torch.zeros(grid_size, device=device, dtype=dtype),
            'W': torch.zeros(grid_size, device=device, dtype=dtype),
            'O': torch.zeros(grid_size, device=device, dtype=dtype),
        }
        self.v_alpha = {
            'H': torch.zeros(grid_size, device=device, dtype=dtype),
            'W': torch.zeros(grid_size, device=device, dtype=dtype),
            'O': torch.zeros(grid_size, device=device, dtype=dtype),
        }

        # Sparse moments for beta (same size as support set)
        self.m_beta = {
            'HW': torch.zeros_like(potentials.beta_HW),
            'HO': torch.zeros_like(potentials.beta_HO),
            'WO': torch.zeros_like(potentials.beta_WO),
        }
        self.v_beta = {
            'HW': torch.zeros_like(potentials.beta_HW),
            'HO': torch.zeros_like(potentials.beta_HO),
            'WO': torch.zeros_like(potentials.beta_WO),
        }

    def step_alpha(self, grad_H: torch.Tensor, grad_W: torch.Tensor, grad_O: torch.Tensor, lr: float):
        """Update alpha potentials"""
        self.t += 1

        for loc_type, grad in [('H', grad_H), ('W', grad_W), ('O', grad_O)]:
            self.m_alpha[loc_type] = self.beta1 * self.m_alpha[loc_type] + (1 - self.beta1) * grad
            self.v_alpha[loc_type] = self.beta2 * self.v_alpha[loc_type] + (1 - self.beta2) * grad ** 2

            m_hat = self.m_alpha[loc_type] / (1 - self.beta1 ** self.t)
            v_hat = self.v_alpha[loc_type] / (1 - self.beta2 ** self.t)

            update = lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            if loc_type == 'H':
                self.potentials.alpha_H -= update
            elif loc_type == 'W':
                self.potentials.alpha_W -= update
            elif loc_type == 'O':
                self.potentials.alpha_O -= update

    def step_beta(self, grad_HW: torch.Tensor, grad_HO: torch.Tensor, grad_WO: torch.Tensor, lr: float):
        """Update beta potentials (sparse, on support set)"""
        for pair, grad in [('HW', grad_HW), ('HO', grad_HO), ('WO', grad_WO)]:
            if len(grad) == 0:
                continue

            self.m_beta[pair] = self.beta1 * self.m_beta[pair] + (1 - self.beta1) * grad
            self.v_beta[pair] = self.beta2 * self.v_beta[pair] + (1 - self.beta2) * grad ** 2

            m_hat = self.m_beta[pair] / (1 - self.beta1 ** max(self.t, 1))
            v_hat = self.v_beta[pair] / (1 - self.beta2 ** max(self.t, 1))

            update = lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            if pair == 'HW':
                self.potentials.beta_HW -= update
            elif pair == 'HO':
                self.potentials.beta_HO -= update
            elif pair == 'WO':
                self.potentials.beta_WO -= update


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


class SSDMFOSparse(BaseMethod):
    """SS-DMFO 3.0 with Sparse Tensor Acceleration

    Key innovations from expert team:
    1. Use real constraint support set S instead of top_k
    2. SDDMM for aggregating interaction statistics (O(N·|S|) instead of O(N·G²))
    3. SpMM for MFVI local field computation
    4. Gumbel noise injection for micro-diversity
    """

    def __init__(self, config: Optional[SparseConfig] = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required. Install with: pip install torch")

        self.config = config or SparseConfig()
        super().__init__("SS-DMFO-Sparse")

        if self.config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[Sparse] Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("[Sparse] Using CPU")

        self.dtype = torch.float32 if self.config.dtype == 'float32' else torch.float64

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """Main optimization with sparse acceleration"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w
        n_users = len(user_patterns)

        # Normalize constraints
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        print(f"[SS-DMFO Sparse] Initializing...")
        print(f"  Device: {self.device}")
        print(f"  Users: {n_users}, Grid: {grid_h}x{grid_w}={grid_size}")

        # Move spatial constraints to GPU
        target_H = torch.from_numpy(constraints.spatial.H.flatten()).to(self.device, self.dtype)
        target_W = torch.from_numpy(constraints.spatial.W.flatten()).to(self.device, self.dtype)
        target_O = torch.from_numpy(constraints.spatial.O.flatten()).to(self.device, self.dtype)

        # Extract support set from real interaction constraints
        if constraints.interaction is None:
            raise ValueError("Interaction constraints required for sparse optimization")

        support = SupportSet(constraints.interaction, self.device, self.dtype)

        # Initialize potentials with support structure
        potentials = SparsePotentials(grid_h, grid_w, support, self.device, self.dtype)

        # Initialize optimizer
        optimizer = SparseAdamOptimizer(potentials)

        # Preprocess user data into batches
        print(f"[SS-DMFO Sparse] Preprocessing user data...")
        gpu_batch_size = self.config.gpu_batch_size
        user_id_list = list(user_patterns.keys())
        n_batches = (n_users + gpu_batch_size - 1) // gpu_batch_size
        print(f"  GPU batch size: {gpu_batch_size}, num batches: {n_batches}")

        user_data_batches = []
        for batch_idx in range(n_batches):
            start = batch_idx * gpu_batch_size
            end = min(start + gpu_batch_size, n_users)
            batch_ids = user_id_list[start:end]
            batch_patterns = {uid: user_patterns[uid] for uid in batch_ids}
            user_data_batches.append(BatchedUserData(batch_patterns, self.device))

        print(f"  Max locations per user: {max(b.max_locs for b in user_data_batches)}")

        # Early stopping
        best_interaction_loss = float('inf')
        last_interaction_loss = 0.0
        best_state = None
        no_improve_count = 0
        best_iter = 0

        # Phase separation
        phase1_iters = int(self.config.max_iter * self.config.phase1_ratio) if self.config.phase_separation else 0

        print(f"\n[SS-DMFO Sparse] Starting optimization (max_iter={self.config.max_iter}, phase1={phase1_iters})...")
        print(f"  Support set sizes: HW={len(support.HW_coords[0])}, HO={len(support.HO_coords[0])}, WO={len(support.WO_coords[0])}")

        for iteration in range(self.config.max_iter):
            iter_start = time.time()

            # Phase control
            in_phase1 = self.config.phase_separation and iteration < phase1_iters
            use_beta = not in_phase1 and iteration >= 5

            # Temperature annealing
            if in_phase1:
                temperature = self.config.temp_init
            else:
                phase2_progress = (iteration - phase1_iters) / max(self.config.max_iter - phase1_iters - 1, 1)
                temperature = self.config.temp_init * (
                    self.config.temp_final / self.config.temp_init
                ) ** phase2_progress

            # Gumbel noise decay
            gumbel_scale = max(
                self.config.gumbel_final,
                self.config.gumbel_scale * (self.config.gumbel_decay ** iteration)
            )

            # ============================================
            # FORWARD PASS: Process users in batches
            # ============================================
            gen_H = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
            gen_W = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
            gen_O = torch.zeros(grid_size, device=self.device, dtype=self.dtype)

            compute_interaction = (not in_phase1 and iteration % self.config.interaction_freq == 0)

            # Aggregators for SDDMM
            if compute_interaction:
                # Initialize aggregated Q values at support positions
                HW_gen = torch.zeros(len(support.HW_coords[0]), device=self.device, dtype=self.dtype)
                HO_gen = torch.zeros(len(support.HO_coords[0]), device=self.device, dtype=self.dtype)
                WO_gen = torch.zeros(len(support.WO_coords[0]), device=self.device, dtype=self.dtype)

            for batch_idx, user_data in enumerate(user_data_batches):
                # Forward pass for this batch
                Q = self._batch_forward(
                    user_data, potentials, grid_size,
                    temperature, gumbel_scale, use_beta
                )

                # Aggregate spatial statistics
                batch_H, batch_W, batch_O = self._aggregate_spatial(Q, user_data)
                gen_H += batch_H
                gen_W += batch_W
                gen_O += batch_O

                # SDDMM: Aggregate interaction at support positions
                if compute_interaction:
                    hw_batch, ho_batch, wo_batch = self._sddmm_aggregate(
                        Q, user_data, support
                    )
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

            # Compute spatial loss
            spatial_loss = self._compute_jsd(gen_H, target_H, gen_W, target_W, gen_O, target_O)

            # Update alpha
            grad_H = -(gen_H - target_H)
            grad_W = -(gen_W - target_W)
            grad_O = -(gen_O - target_O)
            optimizer.step_alpha(grad_H, grad_W, grad_O, self.config.lr_alpha)

            # ============================================
            # INTERACTION UPDATE (using SDDMM results)
            # ============================================
            if compute_interaction:
                # Normalize generated interaction
                HW_gen = HW_gen / (HW_gen.sum() + 1e-10)
                HO_gen = HO_gen / (HO_gen.sum() + 1e-10)
                WO_gen = WO_gen / (WO_gen.sum() + 1e-10)

                # Compute interaction JSD on support set
                last_interaction_loss = self._interaction_jsd_sparse(
                    HW_gen, HO_gen, WO_gen, support
                )

                # Early stopping
                if last_interaction_loss < best_interaction_loss - 0.001:
                    best_interaction_loss = last_interaction_loss
                    best_state = potentials.copy_state()
                    no_improve_count = 0
                    best_iter = iteration
                else:
                    no_improve_count += 1

                # Update beta on support set
                if use_beta:
                    grad_HW = HW_gen - support.HW_values
                    grad_HO = HO_gen - support.HO_values
                    grad_WO = WO_gen - support.WO_values
                    optimizer.step_beta(grad_HW, grad_HO, grad_WO, self.config.lr_beta)

            # Logging
            iter_time = time.time() - iter_start
            phase_str = "P1" if in_phase1 else "P2"
            if iteration % self.config.log_freq == 0 or iteration < 5:
                interact_str = f"{last_interaction_loss:.4f}" if last_interaction_loss > 0 else "---"
                print(f"  Iter {iteration:3d} [{phase_str}]: Spatial={spatial_loss:.4f}, "
                      f"Interact={interact_str}, T={temperature:.2f}, "
                      f"Gumbel={gumbel_scale:.3f} ({iter_time:.1f}s)")

            # Early stopping
            if not in_phase1 and no_improve_count >= self.config.early_stop_patience:
                print(f"  Early stopping at iter {iteration}")
                if best_state is not None:
                    potentials.restore_state(best_state)
                break

        # Final pass
        print(f"\n[SS-DMFO Sparse] Computing final allocations...")
        print(f"  Best interaction: {best_interaction_loss:.4f} at iter {best_iter}")

        final_responses = {}
        for user_data in user_data_batches:
            Q_final = self._batch_forward(
                user_data, potentials, grid_size,
                self.config.temp_final, self.config.gumbel_final, use_beta=True
            )

            Q_np = Q_final.cpu().numpy()
            for u_idx, user_id in enumerate(user_data.user_ids):
                n = user_data.n_locs[u_idx].item()
                final_responses[user_id] = Q_np[u_idx, :n, :]

            del Q_final

        return final_responses

    def _batch_forward(self,
                       user_data: BatchedUserData,
                       potentials: SparsePotentials,
                       grid_size: int,
                       temperature: float,
                       gumbel_scale: float,
                       use_beta: bool) -> torch.Tensor:
        """Batch forward pass with Gumbel noise injection"""
        n_users = user_data.n_users
        max_locs = user_data.max_locs
        inv_temp = 1.0 / temperature

        # Stack alpha for vectorized lookup
        alpha_stack = torch.stack([potentials.alpha_H, potentials.alpha_W, potentials.alpha_O])

        loc_types_clamped = user_data.loc_types.clamp(min=0)
        alpha_per_loc = alpha_stack[loc_types_clamped]

        # Gumbel noise for diversity (critical!)
        gumbel = torch.distributions.Gumbel(0, gumbel_scale).sample(
            (n_users, max_locs, grid_size)
        ).to(self.device, self.dtype)

        log_Q = -alpha_per_loc * inv_temp + gumbel
        Q = F.softmax(log_Q, dim=-1)
        Q = Q * user_data.valid_mask.unsqueeze(-1).float()

        # MFVI with SpMM
        if use_beta and self.config.mfvi_iter > 0:
            Q = self._mfvi_spmm(Q, user_data, potentials, alpha_per_loc, temperature, gumbel_scale)

        return Q

    def _mfvi_spmm(self,
                   Q: torch.Tensor,
                   user_data: BatchedUserData,
                   potentials: SparsePotentials,
                   alpha_per_loc: torch.Tensor,
                   temperature: float,
                   gumbel_scale: float) -> torch.Tensor:
        """MFVI using SpMM for local field computation - memory optimized"""
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

        # Pre-compute masks once
        H_mask_float = user_data.H_mask.unsqueeze(-1).float()
        W_mask_float = user_data.W_mask.unsqueeze(-1).float()
        O_mask_float = user_data.O_mask.unsqueeze(-1).float()
        valid_mask_float = user_data.valid_mask.unsqueeze(-1).float()

        for mfvi_iter in range(self.config.mfvi_iter):
            noise_scale = gumbel_scale * (0.5 ** mfvi_iter)

            # Sum Q by type (memory: 3 * n_users * grid_size)
            H_sum = (Q * H_mask_float).sum(dim=1)
            W_sum = (Q * W_mask_float).sum(dim=1)
            O_sum = (Q * O_mask_float).sum(dim=1)

            # Compute field contributions via SpMM
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

            # Generate Gumbel noise and update Q in-place style
            gumbel = torch.distributions.Gumbel(0, max(noise_scale, 1e-6)).sample(
                (n_users, max_locs, grid_size)
            ).to(self.device, self.dtype)

            log_Q = -field * inv_temp + gumbel
            del field, gumbel

            Q_new = F.softmax(log_Q, dim=-1)
            del log_Q

            # Damped update
            Q = damping * Q_new + (1 - damping) * Q
            del Q_new
            Q = Q * valid_mask_float

            # Clear cache periodically
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
        """SDDMM: Compute π_gen only at support positions - memory optimized

        This is the key innovation: Instead of computing full outer product O(G²),
        we only compute at positions where π_real > 0, which is O(|S|).

        Memory optimization: Process users in mini-batches to avoid OOM
        when indexing large support sets.
        """
        n_users = user_data.n_users
        mini_batch = self.config.sddmm_batch_size

        # Aggregate Q by type for each user
        H_agg = (Q * user_data.H_mask.unsqueeze(-1).float()).sum(dim=1)  # (n_users, grid_size)
        W_agg = (Q * user_data.W_mask.unsqueeze(-1).float()).sum(dim=1)
        O_agg = (Q * user_data.O_mask.unsqueeze(-1).float()).sum(dim=1)

        # Normalize per user
        H_agg = H_agg / (H_agg.sum(dim=1, keepdim=True) + 1e-10)
        W_agg = W_agg / (W_agg.sum(dim=1, keepdim=True) + 1e-10)
        O_agg = O_agg / (O_agg.sum(dim=1, keepdim=True) + 1e-10)

        # Initialize accumulators
        HW_rows, HW_cols = support.HW_coords
        HO_rows, HO_cols = support.HO_coords
        WO_rows, WO_cols = support.WO_coords

        HW_gen = torch.zeros(len(HW_rows), device=self.device, dtype=self.dtype)
        HO_gen = torch.zeros(len(HO_rows), device=self.device, dtype=self.dtype)
        WO_gen = torch.zeros(len(WO_rows), device=self.device, dtype=self.dtype)

        # Process users in mini-batches to avoid OOM
        n_mini_batches = (n_users + mini_batch - 1) // mini_batch

        for mb in range(n_mini_batches):
            start = mb * mini_batch
            end = min(start + mini_batch, n_users)

            H_batch = H_agg[start:end]  # (batch, grid_size)
            W_batch = W_agg[start:end]
            O_batch = O_agg[start:end]

            # SDDMM for this batch: Gather-Multiply-Sum
            # Memory: batch_size * |S| per interaction type
            HW_gen += (H_batch[:, HW_rows] * W_batch[:, HW_cols]).sum(dim=0)
            HO_gen += (H_batch[:, HO_rows] * O_batch[:, HO_cols]).sum(dim=0)
            WO_gen += (W_batch[:, WO_rows] * O_batch[:, WO_cols]).sum(dim=0)

            del H_batch, W_batch, O_batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Average across all users
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
        """Compute JSD on support set (sparse)"""
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


def create_ssdmfo_sparse(preset: str = 'default') -> SSDMFOSparse:
    """Create sparse SS-DMFO optimizer"""
    if preset == 'fast':
        config = SparseConfig(
            max_iter=50,
            gpu_batch_size=2000,
            mfvi_iter=3,
            interaction_freq=3
        )
    elif preset == 'accurate':
        config = SparseConfig(
            max_iter=150,
            gpu_batch_size=1000,
            mfvi_iter=5,
            interaction_freq=2,
            temp_init=2.0,
            temp_final=0.3
        )
    else:
        config = SparseConfig()

    return SSDMFOSparse(config)
