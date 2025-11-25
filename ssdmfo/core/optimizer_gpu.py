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
    lr_alpha: float = 0.1
    lr_beta: float = 0.01
    tolerance: float = 1e-4

    # MFVI
    mfvi_iter: int = 5
    mfvi_damping: float = 0.5

    # Temperature annealing
    temp_init: float = 2.0
    temp_final: float = 1.0

    # Gumbel noise
    gumbel_scale: float = 0.3
    gumbel_decay: float = 0.995
    gumbel_final: float = 0.05

    # Batch size (GPU can handle all users at once)
    batch_size: int = 1000  # Much larger than CPU version

    # Logging
    log_freq: int = 5

    # Interaction
    interaction_freq: int = 3
    top_k: int = 50

    # Early stopping
    early_stop_patience: int = 10
    phase_separation: bool = True

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
        # Initialize as empty sparse tensors
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

        # Preprocess user data for batching
        print(f"[SS-DMFO GPU] Preprocessing user data...")
        user_data = BatchedUserData(user_patterns, self.device)
        print(f"  Max locations per user: {user_data.max_locs}")

        # Early stopping
        best_interaction_loss = float('inf')
        best_state = None
        no_improve_count = 0
        best_iter = 0

        # Phase separation
        phase1_iters = self.config.max_iter // 3 if self.config.phase_separation else 0

        print(f"\n[SS-DMFO GPU] Starting optimization (max_iter={self.config.max_iter})...")

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
            # GPU BATCH FORWARD PASS
            # ============================================
            Q = self._batch_forward_gpu(
                user_data, potentials, grid_size,
                temperature, gumbel_scale, use_beta
            )

            # ============================================
            # AGGREGATE SPATIAL STATISTICS
            # ============================================
            gen_H, gen_W, gen_O = self._aggregate_spatial_gpu(Q, user_data, grid_size)

            # Normalize
            gen_H = gen_H / (gen_H.sum() + 1e-10)
            gen_W = gen_W / (gen_W.sum() + 1e-10)
            gen_O = gen_O / (gen_O.sum() + 1e-10)

            # Compute spatial loss (JSD)
            spatial_loss = self._compute_jsd_gpu(gen_H, target_H, gen_W, target_W, gen_O, target_O)

            # Compute gradients and update alpha
            grad_H = -(gen_H - target_H)
            grad_W = -(gen_W - target_W)
            grad_O = -(gen_O - target_O)

            alpha_lr = self.config.lr_alpha if in_phase1 else self.config.lr_alpha * 0.5
            potentials.alpha_H -= alpha_lr * grad_H
            potentials.alpha_W -= alpha_lr * grad_W
            potentials.alpha_O -= alpha_lr * grad_O

            # ============================================
            # INTERACTION (less frequent)
            # ============================================
            interaction_loss = 0.0
            if constraints.interaction is not None and not in_phase1 and iteration % self.config.interaction_freq == 0:
                # Move Q to CPU for interaction computation (sparse ops)
                Q_cpu = Q.cpu().numpy()
                interaction_loss, gen_interaction = self._compute_interaction_cpu(
                    Q_cpu, user_data, user_patterns, grid_size, constraints.interaction
                )

                # Early stopping
                if interaction_loss < best_interaction_loss - 0.001:
                    best_interaction_loss = interaction_loss
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
                print(f"  Iter {iteration:3d} [{phase_str}]: Spatial={spatial_loss:.4f}, "
                      f"Interact={interaction_loss:.4f}, T={temperature:.2f}, "
                      f"Gumbel={gumbel_scale:.3f} ({iter_time:.1f}s)")

            # Early stopping
            if not in_phase1 and no_improve_count >= self.config.early_stop_patience:
                print(f"  Early stopping at iter {iteration}")
                if best_state is not None:
                    potentials.restore_state(best_state)
                break

        # Final pass
        print(f"\n[SS-DMFO GPU] Computing final allocations...")
        print(f"  Best interaction: {best_interaction_loss:.4f} at iter {best_iter}")

        Q_final = self._batch_forward_gpu(
            user_data, potentials, grid_size,
            self.config.temp_final, self.config.gumbel_final, use_beta=True
        )

        # Convert to output format
        Q_np = Q_final.cpu().numpy()
        final_responses = {}
        for u_idx, user_id in enumerate(user_data.user_ids):
            n = user_data.n_locs[u_idx].item()
            final_responses[user_id] = Q_np[u_idx, :n, :]

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

        # Compute log Q = (-alpha + gumbel) / T
        inv_temp = 1.0 / temperature
        log_Q = (-alpha_per_loc + gumbel) * inv_temp

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
        """GPU-accelerated MFVI iterations"""
        n_users, max_locs, grid_size = Q.shape
        damping = self.config.mfvi_damping
        inv_temp = 1.0 / temperature

        for mfvi_iter in range(self.config.mfvi_iter):
            Q_old = Q.clone()
            noise_scale = gumbel_scale * (0.5 ** mfvi_iter)

            # Generate noise
            gumbel = torch.distributions.Gumbel(0, max(noise_scale, 1e-6)).sample(
                (n_users, max_locs, grid_size)
            ).to(self.device, self.dtype)

            # For each location, compute field = alpha + sum_j beta @ Q_j
            field = alpha_per_loc.clone()

            # Add beta interactions (simplified: aggregate by type)
            # Sum Q by type: H_sum, W_sum, O_sum for each user
            H_sum = (Q_old * user_data.H_mask.unsqueeze(-1).float()).sum(dim=1)  # (n_users, grid_size)
            W_sum = (Q_old * user_data.W_mask.unsqueeze(-1).float()).sum(dim=1)
            O_sum = (Q_old * user_data.O_mask.unsqueeze(-1).float()).sum(dim=1)

            # Apply beta interactions
            if potentials.beta_HW is not None and potentials.beta_HW._nnz() > 0:
                # H locations get contribution from W
                HW_contrib = torch.sparse.mm(potentials.beta_HW, W_sum.T).T  # (n_users, grid_size)
                field = field + user_data.H_mask.unsqueeze(-1).float() * HW_contrib.unsqueeze(1)
                # W locations get contribution from H
                WH_contrib = torch.sparse.mm(potentials.beta_HW.T, H_sum.T).T
                field = field + user_data.W_mask.unsqueeze(-1).float() * WH_contrib.unsqueeze(1)

            if potentials.beta_HO is not None and potentials.beta_HO._nnz() > 0:
                HO_contrib = torch.sparse.mm(potentials.beta_HO, O_sum.T).T
                field = field + user_data.H_mask.unsqueeze(-1).float() * HO_contrib.unsqueeze(1)
                OH_contrib = torch.sparse.mm(potentials.beta_HO.T, H_sum.T).T
                field = field + user_data.O_mask.unsqueeze(-1).float() * OH_contrib.unsqueeze(1)

            if potentials.beta_WO is not None and potentials.beta_WO._nnz() > 0:
                WO_contrib = torch.sparse.mm(potentials.beta_WO, O_sum.T).T
                field = field + user_data.W_mask.unsqueeze(-1).float() * WO_contrib.unsqueeze(1)
                OW_contrib = torch.sparse.mm(potentials.beta_WO.T, W_sum.T).T
                field = field + user_data.O_mask.unsqueeze(-1).float() * OW_contrib.unsqueeze(1)

            # Softmax
            log_Q = (-field + gumbel) * inv_temp
            Q_new = F.softmax(log_Q, dim=-1)

            # Damped update
            Q = damping * Q_new + (1 - damping) * Q_old

            # Zero out padded
            Q = Q * user_data.valid_mask.unsqueeze(-1).float()

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

    def _compute_interaction_cpu(self,
                                 Q_np: np.ndarray,
                                 user_data: BatchedUserData,
                                 user_patterns: Dict[int, UserPattern],
                                 grid_size: int,
                                 real_interaction: InteractionConstraints) -> Tuple[float, InteractionConstraints]:
        """Compute interaction on CPU (sparse operations)"""
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
        """Compute interaction JSD on CPU"""
        def sparse_jsd(p, q):
            p_coo = p.tocoo()
            q_coo = q.tocoo()
            if p_coo.nnz == 0 and q_coo.nnz == 0:
                return 0.0

            p_idx = set(zip(p_coo.row.tolist(), p_coo.col.tolist()))
            q_idx = set(zip(q_coo.row.tolist(), q_coo.col.tolist()))
            all_idx = list(p_idx | q_idx)

            if len(all_idx) > 10000:
                all_idx = [all_idx[i] for i in np.random.choice(len(all_idx), 10000, replace=False)]

            if not all_idx:
                return 0.0

            p_vals = np.array([p[r, c] for r, c in all_idx]) + 1e-10
            q_vals = np.array([q[r, c] for r, c in all_idx]) + 1e-10
            p_vals = p_vals / p_vals.sum()
            q_vals = q_vals / q_vals.sum()
            m = 0.5 * (p_vals + q_vals)

            return 0.5 * (np.sum(p_vals * np.log(p_vals / m)) + np.sum(q_vals * np.log(q_vals / m)))

        losses = []
        if gen.HW.nnz > 0 or real.HW.nnz > 0:
            losses.append(sparse_jsd(gen.HW, real.HW))
        if gen.HO.nnz > 0 or real.HO.nnz > 0:
            losses.append(sparse_jsd(gen.HO, real.HO))
        if gen.WO.nnz > 0 or real.WO.nnz > 0:
            losses.append(sparse_jsd(gen.WO, real.WO))

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

                # Convert to torch sparse
                grad_coo = grad.tocoo()
                indices = torch.tensor([grad_coo.row, grad_coo.col], dtype=torch.long)
                values = torch.tensor(grad_coo.data, dtype=self.dtype)
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
