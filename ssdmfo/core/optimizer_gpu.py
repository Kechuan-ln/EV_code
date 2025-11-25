"""SS-DMFO 3.0 GPU-Accelerated Optimizer

GPU acceleration using PyTorch for:
1. Batch user processing (all users in parallel)
2. Sparse matrix operations (SpMM)
3. Vectorized softmax and Gumbel noise

Requirements:
    pip install torch

Author: SS-DMFO Team
Date: 2025-11-25
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
class GPUConfig:
    """Configuration for GPU-accelerated SS-DMFO"""
    # Optimization
    max_iter: int = 100
    lr_alpha: float = 0.15          # Increased for faster spatial convergence
    lr_beta: float = 0.05           # Increased for better interaction optimization
    tolerance: float = 1e-4

    # MFVI
    mfvi_iter: int = 5
    mfvi_damping: float = 0.5

    # Temperature annealing
    temp_init: float = 2.0
    temp_final: float = 0.3         # Lower for sharper final distributions

    # Gumbel noise
    gumbel_scale: float = 0.3
    gumbel_decay: float = 0.99      # Faster decay
    gumbel_final: float = 0.01      # Lower floor for precise convergence

    # Batch size for GPU processing (to avoid OOM)
    # Adjust based on GPU memory: 500 for 8GB, 1000 for 16GB, 2000 for 24GB
    gpu_batch_size: int = 500

    # Logging
    log_freq: int = 5

    # Interaction
    interaction_freq: int = 2       # More frequent interaction updates
    top_k: int = 500                # Important cells count (500×500=250K interactions)

    # Early stopping
    early_stop_patience: int = 15   # More patience for interaction improvement
    phase_separation: bool = True
    phase1_ratio: float = 0.25      # Shorter phase 1 (was 1/3)

    # GPU settings
    device: str = 'cuda'  # 'cuda' or 'cpu'
    dtype: str = 'float32'  # 'float32' or 'float64'


class GPUPotentials:
    """GPU-resident dual potentials"""

    def __init__(self, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid_size = grid_h * grid_w
        self.device = device
        self.dtype = dtype

        # First-order potentials (on GPU)
        self.alpha_H = torch.zeros(self.grid_size, device=device, dtype=dtype)
        self.alpha_W = torch.zeros(self.grid_size, device=device, dtype=dtype)
        self.alpha_O = torch.zeros(self.grid_size, device=device, dtype=dtype)

        # Second-order potentials (sparse, on GPU)
        self.beta_HW = None
        self.beta_HO = None
        self.beta_WO = None

    def get_alpha(self, loc_type: str) -> torch.Tensor:
        if loc_type == 'H':
            return self.alpha_H
        elif loc_type == 'W':
            return self.alpha_W
        else:
            return self.alpha_O

    def get_beta(self, type1: str, type2: str) -> Optional[torch.Tensor]:
        key = ''.join(sorted([type1, type2]))
        if key == 'HW':
            return self.beta_HW
        elif key == 'HO':
            return self.beta_HO
        elif key == 'OW':
            return self.beta_WO
        return None

    def set_beta(self, type1: str, type2: str, beta: torch.Tensor):
        key = ''.join(sorted([type1, type2]))
        if key == 'HW':
            self.beta_HW = beta
        elif key == 'HO':
            self.beta_HO = beta
        elif key == 'OW':
            self.beta_WO = beta

    def copy_state(self) -> Dict:
        state = {
            'alpha_H': self.alpha_H.clone(),
            'alpha_W': self.alpha_W.clone(),
            'alpha_O': self.alpha_O.clone(),
        }
        if self.beta_HW is not None:
            state['beta_HW'] = self.beta_HW.clone()
        if self.beta_HO is not None:
            state['beta_HO'] = self.beta_HO.clone()
        if self.beta_WO is not None:
            state['beta_WO'] = self.beta_WO.clone()
        return state

    def restore_state(self, state: Dict):
        self.alpha_H = state['alpha_H'].clone()
        self.alpha_W = state['alpha_W'].clone()
        self.alpha_O = state['alpha_O'].clone()
        if 'beta_HW' in state:
            self.beta_HW = state['beta_HW'].clone()
        if 'beta_HO' in state:
            self.beta_HO = state['beta_HO'].clone()
        if 'beta_WO' in state:
            self.beta_WO = state['beta_WO'].clone()


class GPUAdamOptimizer:
    """Adam optimizer for GPU potentials - matches CPU behavior"""

    def __init__(self, potentials: GPUPotentials, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.potentials = potentials
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        device = potentials.device
        dtype = potentials.dtype
        grid_size = potentials.grid_size

        # First moment (momentum)
        self.m = {
            'H': torch.zeros(grid_size, device=device, dtype=dtype),
            'W': torch.zeros(grid_size, device=device, dtype=dtype),
            'O': torch.zeros(grid_size, device=device, dtype=dtype),
        }

        # Second moment (RMSprop)
        self.v = {
            'H': torch.zeros(grid_size, device=device, dtype=dtype),
            'W': torch.zeros(grid_size, device=device, dtype=dtype),
            'O': torch.zeros(grid_size, device=device, dtype=dtype),
        }

    def step(self, grad_H: torch.Tensor, grad_W: torch.Tensor, grad_O: torch.Tensor, lr: float):
        """Execute one Adam update step"""
        self.t += 1

        for loc_type, grad in [('H', grad_H), ('W', grad_W), ('O', grad_O)]:
            # Update first moment
            self.m[loc_type] = self.beta1 * self.m[loc_type] + (1 - self.beta1) * grad

            # Update second moment
            self.v[loc_type] = self.beta2 * self.v[loc_type] + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = self.m[loc_type] / (1 - self.beta1 ** self.t)
            v_hat = self.v[loc_type] / (1 - self.beta2 ** self.t)

            # Compute update
            update = lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            # Apply update
            if loc_type == 'H':
                self.potentials.alpha_H -= update
            elif loc_type == 'W':
                self.potentials.alpha_W -= update
            elif loc_type == 'O':
                self.potentials.alpha_O -= update


class BatchedUserData:
    """Preprocessed user data for GPU batch processing"""

    def __init__(self, user_patterns: Dict[int, UserPattern], device: torch.device):
        self.device = device
        self.user_ids = list(user_patterns.keys())
        self.n_users = len(self.user_ids)

        # Find max locations per user
        self.max_locs = max(len(u.locations) for u in user_patterns.values())

        # Create batched tensors
        # loc_types[u, l] = type index (0=H, 1=W, 2=O, -1=padding)
        self.loc_types = torch.full((self.n_users, self.max_locs), -1,
                                    dtype=torch.long, device=device)
        # n_locs[u] = number of locations for user u
        self.n_locs = torch.zeros(self.n_users, dtype=torch.long, device=device)

        type_map = {'H': 0, 'W': 1, 'O': 2}

        for u_idx, user_id in enumerate(self.user_ids):
            user = user_patterns[user_id]
            n = len(user.locations)
            self.n_locs[u_idx] = n
            for l_idx, loc in enumerate(user.locations):
                self.loc_types[u_idx, l_idx] = type_map[loc.type]

        # Masks for each type
        self.H_mask = (self.loc_types == 0)  # (n_users, max_locs)
        self.W_mask = (self.loc_types == 1)
        self.O_mask = (self.loc_types == 2)
        self.valid_mask = (self.loc_types >= 0)


class SSDMFOv3GPU(BaseMethod):
    """GPU-Accelerated SS-DMFO 3.0

    Key optimizations:
    1. All users processed in parallel on GPU
    2. Batched sparse matrix operations
    3. Vectorized Gumbel-softmax
    4. Fused CUDA kernels via PyTorch
    """

    def __init__(self, config: Optional[GPUConfig] = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU acceleration. Install with: pip install torch")

        self.config = config or GPUConfig()
        super().__init__("SS-DMFO-v3-GPU")

        # Setup device
        if self.config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("[GPU] CUDA not available, using CPU")

        self.dtype = torch.float32 if self.config.dtype == 'float32' else torch.float64

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """GPU-accelerated optimization"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w
        n_users = len(user_patterns)

        # Normalize constraints
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        print(f"[SS-DMFO GPU] Initializing...")
        print(f"  Device: {self.device}")
        print(f"  Users: {n_users}, Grid: {grid_h}x{grid_w}={grid_size}")
        print(f"  Temperature: {self.config.temp_init} -> {self.config.temp_final}")
        print(f"  Gumbel scale: {self.config.gumbel_scale} -> {self.config.gumbel_final}")

        # Move constraints to GPU
        target_H = torch.from_numpy(constraints.spatial.H.flatten()).to(self.device, self.dtype)
        target_W = torch.from_numpy(constraints.spatial.W.flatten()).to(self.device, self.dtype)
        target_O = torch.from_numpy(constraints.spatial.O.flatten()).to(self.device, self.dtype)

        # Initialize potentials on GPU
        potentials = GPUPotentials(grid_h, grid_w, self.device, self.dtype)

        # Initialize Adam optimizer (CRITICAL: matches CPU version behavior)
        optimizer = GPUAdamOptimizer(potentials)

        # Preprocess user data - split into batches to avoid OOM
        print(f"[SS-DMFO GPU] Preprocessing user data...")
        gpu_batch_size = self.config.gpu_batch_size
        user_id_list = list(user_patterns.keys())
        n_batches = (n_users + gpu_batch_size - 1) // gpu_batch_size
        print(f"  GPU batch size: {gpu_batch_size}, num batches: {n_batches}")

        # Create batched user data objects
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
        last_interaction_loss = 0.0  # Track last computed interaction for display
        best_state = None
        no_improve_count = 0
        best_iter = 0

        # Phase separation - use configurable ratio
        phase1_iters = int(self.config.max_iter * self.config.phase1_ratio) if self.config.phase_separation else 0

        print(f"\n[SS-DMFO GPU] Starting optimization (max_iter={self.config.max_iter}, phase1={phase1_iters})...")

        for iteration in range(self.config.max_iter):
            iter_start = time.time()

            # Phase separation
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
            # GPU BATCH FORWARD PASS (process in batches to avoid OOM)
            # ============================================
            gen_H = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
            gen_W = torch.zeros(grid_size, device=self.device, dtype=self.dtype)
            gen_O = torch.zeros(grid_size, device=self.device, dtype=self.dtype)

            # Determine if we need interaction this iteration
            compute_interaction = (constraints.interaction is not None and
                                   not in_phase1 and
                                   iteration % self.config.interaction_freq == 0)

            # Store Q tensors for GPU interaction computation
            all_Q_gpu = [] if compute_interaction else None

            for batch_idx, user_data in enumerate(user_data_batches):
                Q = self._batch_forward_gpu(
                    user_data, potentials, grid_size,
                    temperature, gumbel_scale, use_beta
                )

                # Aggregate spatial statistics for this batch
                batch_H, batch_W, batch_O = self._aggregate_spatial_gpu(Q, user_data, grid_size)
                gen_H += batch_H
                gen_W += batch_W
                gen_O += batch_O

                # Keep Q on GPU for interaction computation
                if compute_interaction:
                    all_Q_gpu.append((user_data, Q))
                else:
                    del Q
                    torch.cuda.empty_cache()

            # Normalize
            gen_H = gen_H / (gen_H.sum() + 1e-10)
            gen_W = gen_W / (gen_W.sum() + 1e-10)
            gen_O = gen_O / (gen_O.sum() + 1e-10)

            # Compute spatial loss (JSD)
            spatial_loss = self._compute_jsd_gpu(gen_H, target_H, gen_W, target_W, gen_O, target_O)

            # Compute gradients and update alpha using Adam optimizer
            grad_H = -(gen_H - target_H)
            grad_W = -(gen_W - target_W)
            grad_O = -(gen_O - target_O)

            # Keep learning rate high throughout - Adam handles adaptation
            alpha_lr = self.config.lr_alpha
            optimizer.step(grad_H, grad_W, grad_O, alpha_lr)

            # ============================================
            # INTERACTION (GPU-accelerated)
            # ============================================
            gen_interaction = None
            if all_Q_gpu:
                # Compute interaction using GPU for each batch
                all_interactions = []
                for user_data, Q in all_Q_gpu:
                    loss, interaction = self._compute_interaction_gpu(
                        Q, user_data, grid_size, constraints.interaction
                    )
                    all_interactions.append((loss, interaction))
                    del Q

                torch.cuda.empty_cache()

                # Average interaction loss across batches
                if all_interactions:
                    last_interaction_loss = np.mean([x[0] for x in all_interactions])
                    # Merge sparse matrices from all batches
                    gen_interaction = self._merge_interactions([x[1] for x in all_interactions], grid_size)

                # Early stopping
                if last_interaction_loss < best_interaction_loss - 0.001:
                    best_interaction_loss = last_interaction_loss
                    best_state = potentials.copy_state()
                    no_improve_count = 0
                    best_iter = iteration
                else:
                    no_improve_count += 1

                # Update beta (on CPU, then move to GPU)
                if use_beta:
                    self._update_beta_gpu(potentials, gen_interaction,
                                         constraints.interaction, self.config.lr_beta)

            # Logging
            iter_time = time.time() - iter_start
            phase_str = "P1" if in_phase1 else "P2"
            if iteration % self.config.log_freq == 0 or iteration < 5:
                # Show last known interaction loss (not 0 when not computed)
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

        # Final pass (also in batches)
        print(f"\n[SS-DMFO GPU] Computing final allocations...")
        print(f"  Best interaction: {best_interaction_loss:.4f} at iter {best_iter}")

        final_responses = {}
        for user_data in user_data_batches:
            Q_final = self._batch_forward_gpu(
                user_data, potentials, grid_size,
                self.config.temp_final, self.config.gumbel_final, use_beta=True
            )

            # Convert to output format
            Q_np = Q_final.cpu().numpy()
            for u_idx, user_id in enumerate(user_data.user_ids):
                n = user_data.n_locs[u_idx].item()
                final_responses[user_id] = Q_np[u_idx, :n, :]

            del Q_final
            torch.cuda.empty_cache()

        return final_responses

    def _batch_forward_gpu(self,
                           user_data: BatchedUserData,
                           potentials: GPUPotentials,
                           grid_size: int,
                           temperature: float,
                           gumbel_scale: float,
                           use_beta: bool) -> torch.Tensor:
        """
        Batch forward pass for ALL users on GPU

        Returns:
            Q: (n_users, max_locs, grid_size) tensor
        """
        n_users = user_data.n_users
        max_locs = user_data.max_locs
        inv_temp = 1.0 / temperature

        # Stack alpha for vectorized lookup
        alpha_stack = torch.stack([potentials.alpha_H, potentials.alpha_W, potentials.alpha_O])
        # alpha_stack: (3, grid_size)

        # Gather alpha for each location: (n_users, max_locs, grid_size)
        loc_types_clamped = user_data.loc_types.clamp(min=0)  # Avoid -1 indexing
        alpha_per_loc = alpha_stack[loc_types_clamped]  # (n_users, max_locs, grid_size)

        # Generate Gumbel noise for all users/locations at once
        gumbel = torch.distributions.Gumbel(0, gumbel_scale).sample(
            (n_users, max_locs, grid_size)
        ).to(self.device, self.dtype)

        # FIX: Compute log Q = -alpha/T + gumbel (gumbel NOT scaled by temperature!)
        # This matches the CPU version behavior
        log_Q = -alpha_per_loc * inv_temp + gumbel

        # Softmax along grid dimension
        Q = F.softmax(log_Q, dim=-1)  # (n_users, max_locs, grid_size)

        # Zero out padded locations
        Q = Q * user_data.valid_mask.unsqueeze(-1).float()

        # MFVI with beta coupling (simplified - full version would iterate)
        if use_beta and self.config.mfvi_iter > 0:
            Q = self._mfvi_gpu(Q, user_data, potentials, alpha_per_loc,
                              temperature, gumbel_scale)

        return Q

    def _mfvi_gpu(self,
                  Q: torch.Tensor,
                  user_data: BatchedUserData,
                  potentials: GPUPotentials,
                  alpha_per_loc: torch.Tensor,
                  temperature: float,
                  gumbel_scale: float) -> torch.Tensor:
        """GPU-accelerated MFVI iterations - memory optimized"""
        n_users, max_locs, grid_size = Q.shape
        damping = self.config.mfvi_damping
        inv_temp = 1.0 / temperature

        # Check if any beta has content
        has_HW = potentials.beta_HW is not None and potentials.beta_HW._nnz() > 0
        has_HO = potentials.beta_HO is not None and potentials.beta_HO._nnz() > 0
        has_WO = potentials.beta_WO is not None and potentials.beta_WO._nnz() > 0

        if not (has_HW or has_HO or has_WO):
            return Q  # No beta interactions, skip MFVI

        for mfvi_iter in range(self.config.mfvi_iter):
            noise_scale = gumbel_scale * (0.5 ** mfvi_iter)

            # Generate noise
            gumbel = torch.distributions.Gumbel(0, max(noise_scale, 1e-6)).sample(
                (n_users, max_locs, grid_size)
            ).to(self.device, self.dtype)

            # For each location, compute field = alpha + sum_j beta @ Q_j
            field = alpha_per_loc.clone()

            # Sum Q by type: H_sum, W_sum, O_sum for each user
            H_sum = (Q * user_data.H_mask.unsqueeze(-1).float()).sum(dim=1)  # (n_users, grid_size)
            W_sum = (Q * user_data.W_mask.unsqueeze(-1).float()).sum(dim=1)
            O_sum = (Q * user_data.O_mask.unsqueeze(-1).float()).sum(dim=1)

            # Apply beta interactions
            if has_HW:
                # H locations get contribution from W
                HW_contrib = torch.sparse.mm(potentials.beta_HW, W_sum.T).T
                field = field + user_data.H_mask.unsqueeze(-1).float() * HW_contrib.unsqueeze(1)
                # W locations get contribution from H
                WH_contrib = torch.sparse.mm(potentials.beta_HW.T, H_sum.T).T
                field = field + user_data.W_mask.unsqueeze(-1).float() * WH_contrib.unsqueeze(1)

            if has_HO:
                HO_contrib = torch.sparse.mm(potentials.beta_HO, O_sum.T).T
                field = field + user_data.H_mask.unsqueeze(-1).float() * HO_contrib.unsqueeze(1)
                OH_contrib = torch.sparse.mm(potentials.beta_HO.T, H_sum.T).T
                field = field + user_data.O_mask.unsqueeze(-1).float() * OH_contrib.unsqueeze(1)

            if has_WO:
                WO_contrib = torch.sparse.mm(potentials.beta_WO, O_sum.T).T
                field = field + user_data.W_mask.unsqueeze(-1).float() * WO_contrib.unsqueeze(1)
                OW_contrib = torch.sparse.mm(potentials.beta_WO.T, W_sum.T).T
                field = field + user_data.O_mask.unsqueeze(-1).float() * OW_contrib.unsqueeze(1)

            # FIX: Softmax with correct temperature scaling (gumbel NOT scaled)
            log_Q = -field * inv_temp + gumbel
            Q_new = F.softmax(log_Q, dim=-1)

            # Damped update (in-place to save memory)
            Q = damping * Q_new + (1 - damping) * Q

            # Zero out padded
            Q = Q * user_data.valid_mask.unsqueeze(-1).float()

            # Free intermediate tensors
            del gumbel, field, Q_new, H_sum, W_sum, O_sum
            if has_HW:
                del HW_contrib, WH_contrib
            if has_HO:
                del HO_contrib, OH_contrib
            if has_WO:
                del WO_contrib, OW_contrib
            torch.cuda.empty_cache()

        return Q

    def _aggregate_spatial_gpu(self,
                               Q: torch.Tensor,
                               user_data: BatchedUserData,
                               grid_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate spatial statistics on GPU"""
        # Q: (n_users, max_locs, grid_size)
        # Sum by type
        gen_H = (Q * user_data.H_mask.unsqueeze(-1).float()).sum(dim=(0, 1))
        gen_W = (Q * user_data.W_mask.unsqueeze(-1).float()).sum(dim=(0, 1))
        gen_O = (Q * user_data.O_mask.unsqueeze(-1).float()).sum(dim=(0, 1))

        return gen_H, gen_W, gen_O

    def _compute_jsd_gpu(self, gen_H, target_H, gen_W, target_W, gen_O, target_O) -> float:
        """Compute JSD on GPU"""
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

    def _compute_interaction_gpu(self,
                                 Q: torch.Tensor,
                                 user_data: BatchedUserData,
                                 grid_size: int,
                                 real_interaction: InteractionConstraints) -> Tuple[float, 'InteractionConstraints']:
        """Smart interaction computation: Important Cells + Sparse Outer Products

        Strategy:
        1. Identify important cells from real spatial constraints (foundation)
        2. For each user, compute sparse outer product (captures true correlation)
        3. Aggregate across users (true joint distribution, not independence)

        This avoids the 40885×40885 full matrix while capturing real correlations.
        """
        n_users = user_data.n_users
        n_important = self.config.top_k  # Reuse top_k as number of important cells

        # Step 1: Identify important cells from aggregate distributions
        # These are cells where users actually have significant probability
        H_agg = (Q * user_data.H_mask.unsqueeze(-1).float()).sum(dim=1)  # (n_users, grid_size)
        W_agg = (Q * user_data.W_mask.unsqueeze(-1).float()).sum(dim=1)
        O_agg = (Q * user_data.O_mask.unsqueeze(-1).float()).sum(dim=1)

        # Get important cell indices based on total probability mass
        H_total = H_agg.sum(dim=0)  # (grid_size,)
        W_total = W_agg.sum(dim=0)
        O_total = O_agg.sum(dim=0)

        _, H_important = torch.topk(H_total, min(n_important, grid_size))
        _, W_important = torch.topk(W_total, min(n_important, grid_size))
        _, O_important = torch.topk(O_total, min(n_important, grid_size))

        # Step 2: For each user, compute sparse outer product at important positions only
        # Extract values at important positions: (n_users, n_important)
        H_vals = H_agg[:, H_important]
        W_vals = W_agg[:, W_important]
        O_vals = O_agg[:, O_important]

        # Normalize per user (each user's contribution should sum to 1)
        H_vals = H_vals / (H_vals.sum(dim=1, keepdim=True) + 1e-10)
        W_vals = W_vals / (W_vals.sum(dim=1, keepdim=True) + 1e-10)
        O_vals = O_vals / (O_vals.sum(dim=1, keepdim=True) + 1e-10)

        # Compute outer products for ALL users at once: (n_users, n_important, n_important)
        # This captures the TRUE correlation structure per user
        HW_outer = torch.bmm(H_vals.unsqueeze(2), W_vals.unsqueeze(1))
        HO_outer = torch.bmm(H_vals.unsqueeze(2), O_vals.unsqueeze(1))
        WO_outer = torch.bmm(W_vals.unsqueeze(2), O_vals.unsqueeze(1))

        # Step 3: Aggregate across users (mean, not sum, to get probability)
        HW_agg = HW_outer.mean(dim=0)  # (n_important, n_important)
        HO_agg = HO_outer.mean(dim=0)
        WO_agg = WO_outer.mean(dim=0)

        # Convert to CPU numpy
        HW_np = HW_agg.cpu().numpy()
        HO_np = HO_agg.cpu().numpy()
        WO_np = WO_agg.cpu().numpy()

        H_idx = H_important.cpu().numpy()
        W_idx = W_important.cpu().numpy()
        O_idx = O_important.cpu().numpy()

        # Build sparse matrices with correct global indices
        def build_sparse_interaction(values, row_idx, col_idx):
            rows, cols = np.meshgrid(row_idx, col_idx, indexing='ij')
            mask = values > 1e-10
            if not np.any(mask):
                return sparse.csr_matrix((grid_size, grid_size))
            return sparse.csr_matrix(
                (values[mask], (rows[mask], cols[mask])),
                shape=(grid_size, grid_size)
            )

        HW = build_sparse_interaction(HW_np, H_idx, W_idx)
        HO = build_sparse_interaction(HO_np, H_idx, O_idx)
        WO = build_sparse_interaction(WO_np, W_idx, O_idx)

        gen_interaction = InteractionConstraints(HW=HW, HO=HO, WO=WO)
        gen_interaction.normalize()

        # Compute JSD with real interaction
        loss = self._interaction_jsd_cpu(gen_interaction, real_interaction)

        return loss, gen_interaction

    def _merge_interactions(self,
                            interactions: List[InteractionConstraints],
                            grid_size: int) -> InteractionConstraints:
        """Merge interaction constraints from multiple batches"""
        if len(interactions) == 1:
            return interactions[0]

        # Sum sparse matrices
        HW = sum(x.HW for x in interactions)
        HO = sum(x.HO for x in interactions)
        WO = sum(x.WO for x in interactions)

        merged = InteractionConstraints(HW=HW, HO=HO, WO=WO)
        merged.normalize()
        return merged

    def _compute_interaction_cpu_batched(self,
                                         all_Q_cpu: List[Tuple[BatchedUserData, np.ndarray]],
                                         user_patterns: Dict[int, UserPattern],
                                         grid_size: int,
                                         real_interaction: InteractionConstraints) -> Tuple[float, InteractionConstraints]:
        """Compute interaction on CPU from batched Q data"""
        top_k = self.config.top_k

        # Collect all user locations by type
        user_locs = {}

        for user_data, Q_np in all_Q_cpu:
            for u_idx, user_id in enumerate(user_data.user_ids):
                n = user_data.n_locs[u_idx].item()
                Q = Q_np[u_idx, :n, :]

                user_locs[user_id] = {'H': [], 'W': [], 'O': []}
                pattern = user_patterns[user_id]
                for loc_idx, loc in enumerate(pattern.locations):
                    user_locs[user_id][loc.type].append(Q[loc_idx])

        # Compute interactions
        hw_data, hw_rows, hw_cols = [], [], []
        ho_data, ho_rows, ho_cols = [], [], []
        wo_data, wo_rows, wo_cols = [], [], []

        for user_id in user_locs.keys():
            H_locs = user_locs[user_id]['H']
            W_locs = user_locs[user_id]['W']
            O_locs = user_locs[user_id]['O']

            if H_locs and W_locs:
                self._add_pairwise_fast(H_locs, W_locs, top_k, grid_size, hw_data, hw_rows, hw_cols)
            if H_locs and O_locs:
                self._add_pairwise_fast(H_locs, O_locs, top_k, grid_size, ho_data, ho_rows, ho_cols)
            if W_locs and O_locs:
                self._add_pairwise_fast(W_locs, O_locs, top_k, grid_size, wo_data, wo_rows, wo_cols)

        # Build sparse matrices
        HW = sparse.csr_matrix((hw_data, (hw_rows, hw_cols)), shape=(grid_size, grid_size)) if hw_data else sparse.csr_matrix((grid_size, grid_size))
        HO = sparse.csr_matrix((ho_data, (ho_rows, ho_cols)), shape=(grid_size, grid_size)) if ho_data else sparse.csr_matrix((grid_size, grid_size))
        WO = sparse.csr_matrix((wo_data, (wo_rows, wo_cols)), shape=(grid_size, grid_size)) if wo_data else sparse.csr_matrix((grid_size, grid_size))

        gen_interaction = InteractionConstraints(HW=HW, HO=HO, WO=WO)
        gen_interaction.normalize()

        # Compute JSD
        loss = self._interaction_jsd_cpu(gen_interaction, real_interaction)

        return loss, gen_interaction

    def _compute_interaction_cpu(self,
                                 Q_np: np.ndarray,
                                 user_data: BatchedUserData,
                                 user_patterns: Dict[int, UserPattern],
                                 grid_size: int,
                                 real_interaction: InteractionConstraints) -> Tuple[float, InteractionConstraints]:
        """Compute interaction on CPU (sparse operations) - single batch version"""
        top_k = self.config.top_k

        # Build responses dict
        responses = {}
        for u_idx, user_id in enumerate(user_data.user_ids):
            n = user_data.n_locs[u_idx].item()
            responses[user_id] = Q_np[u_idx, :n, :]

        # Collect by type
        user_locs = {uid: {'H': [], 'W': [], 'O': []} for uid in responses.keys()}
        for user_id, Q in responses.items():
            pattern = user_patterns[user_id]
            for loc_idx, loc in enumerate(pattern.locations):
                user_locs[user_id][loc.type].append(Q[loc_idx])

        # Compute interactions
        hw_data, hw_rows, hw_cols = [], [], []
        ho_data, ho_rows, ho_cols = [], [], []
        wo_data, wo_rows, wo_cols = [], [], []

        for user_id in responses.keys():
            H_locs = user_locs[user_id]['H']
            W_locs = user_locs[user_id]['W']
            O_locs = user_locs[user_id]['O']

            if H_locs and W_locs:
                self._add_pairwise_fast(H_locs, W_locs, top_k, grid_size, hw_data, hw_rows, hw_cols)
            if H_locs and O_locs:
                self._add_pairwise_fast(H_locs, O_locs, top_k, grid_size, ho_data, ho_rows, ho_cols)
            if W_locs and O_locs:
                self._add_pairwise_fast(W_locs, O_locs, top_k, grid_size, wo_data, wo_rows, wo_cols)

        # Build sparse matrices
        HW = sparse.csr_matrix((hw_data, (hw_rows, hw_cols)), shape=(grid_size, grid_size)) if hw_data else sparse.csr_matrix((grid_size, grid_size))
        HO = sparse.csr_matrix((ho_data, (ho_rows, ho_cols)), shape=(grid_size, grid_size)) if ho_data else sparse.csr_matrix((grid_size, grid_size))
        WO = sparse.csr_matrix((wo_data, (wo_rows, wo_cols)), shape=(grid_size, grid_size)) if wo_data else sparse.csr_matrix((grid_size, grid_size))

        gen_interaction = InteractionConstraints(HW=HW, HO=HO, WO=WO)
        gen_interaction.normalize()

        # Compute JSD
        loss = self._interaction_jsd_cpu(gen_interaction, real_interaction)

        return loss, gen_interaction

    def _add_pairwise_fast(self, locs1, locs2, top_k, grid_size, data, rows, cols):
        """Vectorized pairwise computation"""
        q1 = np.sum(locs1, axis=0)
        q2 = np.sum(locs2, axis=0)

        if top_k < grid_size:
            idx1 = np.argpartition(q1, -top_k)[-top_k:]
            idx2 = np.argpartition(q2, -top_k)[-top_k:]
        else:
            idx1 = np.where(q1 > 1e-10)[0]
            idx2 = np.where(q2 > 1e-10)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            return

        p1 = q1[idx1]
        p2 = q2[idx2]
        p1 = p1 / (p1.sum() + 1e-10)
        p2 = p2 / (p2.sum() + 1e-10)

        outer = np.outer(p1, p2)
        mask = outer > 1e-15
        i_mesh, j_mesh = np.meshgrid(idx1, idx2, indexing='ij')

        data.extend(outer[mask].tolist())
        rows.extend(i_mesh[mask].tolist())
        cols.extend(j_mesh[mask].tolist())

    def _interaction_jsd_cpu(self, gen, real) -> float:
        """Compute interaction JSD on CPU - optimized with sampling"""
        def sparse_jsd_fast(p, q, n_samples=5000):
            """Fast JSD using sampling from both distributions"""
            p_coo = p.tocoo()
            q_coo = q.tocoo()

            if p_coo.nnz == 0 and q_coo.nnz == 0:
                return 0.0

            # Sample from generated distribution (smaller, so sample all if small)
            if p_coo.nnz <= n_samples:
                p_rows, p_cols, p_data = p_coo.row, p_coo.col, p_coo.data
            else:
                idx = np.random.choice(p_coo.nnz, n_samples, replace=False)
                p_rows, p_cols, p_data = p_coo.row[idx], p_coo.col[idx], p_coo.data[idx]

            # Sample from real distribution
            if q_coo.nnz <= n_samples:
                q_rows, q_cols, q_data = q_coo.row, q_coo.col, q_coo.data
            else:
                idx = np.random.choice(q_coo.nnz, n_samples, replace=False)
                q_rows, q_cols, q_data = q_coo.row[idx], q_coo.col[idx], q_coo.data[idx]

            # Combine indices using numpy (faster than Python sets)
            p_keys = p_rows * q.shape[1] + p_cols  # Flatten to 1D keys
            q_keys = q_rows * q.shape[1] + q_cols

            all_keys = np.unique(np.concatenate([p_keys, q_keys]))

            if len(all_keys) == 0:
                return 0.0

            # Limit samples
            if len(all_keys) > n_samples:
                all_keys = np.random.choice(all_keys, n_samples, replace=False)

            # Convert back to (row, col)
            all_rows = all_keys // q.shape[1]
            all_cols = all_keys % q.shape[1]

            # Get values using sparse matrix indexing (vectorized)
            p_vals = np.asarray(p[all_rows, all_cols]).flatten() + 1e-10
            q_vals = np.asarray(q[all_rows, all_cols]).flatten() + 1e-10

            # Normalize
            p_vals = p_vals / p_vals.sum()
            q_vals = q_vals / q_vals.sum()
            m = 0.5 * (p_vals + q_vals)

            return 0.5 * (np.sum(p_vals * np.log(p_vals / m)) + np.sum(q_vals * np.log(q_vals / m)))

        losses = []
        if gen.HW.nnz > 0 or real.HW.nnz > 0:
            losses.append(sparse_jsd_fast(gen.HW, real.HW))
        if gen.HO.nnz > 0 or real.HO.nnz > 0:
            losses.append(sparse_jsd_fast(gen.HO, real.HO))
        if gen.WO.nnz > 0 or real.WO.nnz > 0:
            losses.append(sparse_jsd_fast(gen.WO, real.WO))

        return np.mean(losses) if losses else 0.0

    def _update_beta_gpu(self, potentials: GPUPotentials,
                         gen: InteractionConstraints,
                         real: InteractionConstraints,
                         lr: float):
        """Update beta potentials on GPU"""
        for pair, gen_mat, real_mat in [('HW', gen.HW, real.HW), ('HO', gen.HO, real.HO), ('WO', gen.WO, real.WO)]:
            if gen_mat.nnz > 0 or real_mat.nnz > 0:
                # Compute gradient on CPU
                grad = gen_mat - real_mat

                # Convert to torch sparse (FIX: use np.array to avoid warning)
                grad_coo = grad.tocoo()
                indices = torch.from_numpy(
                    np.vstack([grad_coo.row, grad_coo.col]).astype(np.int64)
                ).to(self.device)
                values = torch.from_numpy(
                    grad_coo.data.astype(np.float32 if self.dtype == torch.float32 else np.float64)
                ).to(self.device)
                grad_torch = torch.sparse_coo_tensor(
                    indices, values, grad.shape, device=self.device, dtype=self.dtype
                ).coalesce()

                # Update
                current = potentials.get_beta(pair[0], pair[1])
                if current is None:
                    potentials.set_beta(pair[0], pair[1], -lr * grad_torch)
                else:
                    # Sparse addition
                    updated = current + (-lr * grad_torch)
                    potentials.set_beta(pair[0], pair[1], updated.coalesce())


def create_ssdmfo_gpu(preset: str = 'default') -> SSDMFOv3GPU:
    """Create GPU-accelerated SS-DMFO optimizer"""
    if preset == 'fast':
        config = GPUConfig(
            max_iter=50,
            batch_size=2000,
            mfvi_iter=3,
            interaction_freq=5,
            top_k=50
        )
    elif preset == 'accurate':
        config = GPUConfig(
            max_iter=150,
            batch_size=1000,
            mfvi_iter=5,
            interaction_freq=2,
            top_k=100,
            temp_init=2.0,
            temp_final=1.0
        )
    else:
        config = GPUConfig()

    return SSDMFOv3GPU(config)
