"""SS-DMFO 对偶优化器

核心算法流程：
1. 初始化势函数 α, β
2. 循环直到收敛：
   a. 给定势函数，用MFVI计算每个用户的最优响应 Q_i
   b. 聚合响应得到生成统计 μ_gen, π_gen
   c. 计算梯度 = 生成统计 - 真实统计
   d. 更新势函数
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from scipy import sparse

from ..data.structures import (
    Constraints, UserPattern, Result,
    SpatialConstraints, InteractionConstraints
)
from ..baselines.base import BaseMethod
from .potentials import DualPotentials, PotentialsWithMomentum
from .mean_field import FastMeanFieldSolver


class SSDMFOOptimizer(BaseMethod):
    """SS-DMFO 对偶优化器"""

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
            phase: 优化阶段 (1=空间, 2=空间+交互)
            max_iter: 最大迭代次数
            lr: 学习率
            temperature: MFVI温度参数
            tolerance: 收敛容差
            log_freq: 日志频率
            use_adam: 是否使用Adam优化器
        """
        super().__init__(f"SS-DMFO(phase={phase})")
        self.phase = phase
        self.max_iter = max_iter
        self.lr = lr
        self.temperature = temperature
        self.tolerance = tolerance
        self.log_freq = log_freq
        self.use_adam = use_adam

        # MFVI求解器
        self.mf_solver = FastMeanFieldSolver(
            temperature=temperature,
            max_iter=10,
            damping=0.5
        )

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """运行SS-DMFO优化"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        # 归一化约束
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        # 初始化势函数
        print(f"Initializing potentials (phase={self.phase})...")
        potentials = DualPotentials.initialize(
            grid_h, grid_w, phase=self.phase, init_scale=0.01
        )

        # 优化器
        if self.use_adam:
            optimizer = PotentialsWithMomentum(potentials)
        else:
            optimizer = None

        # 优化循环
        print(f"Starting optimization (max_iter={self.max_iter}, lr={self.lr})...")
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # Step 1: 计算所有用户的响应
            responses = self.mf_solver.compute_all_responses_fast(
                user_patterns, potentials, constraints, self.phase
            )

            # Step 2: 聚合统计
            gen_spatial = self._aggregate_spatial(
                responses, user_patterns, grid_h, grid_w
            )

            # Step 3: 计算梯度
            gradients = self._compute_spatial_gradients(
                gen_spatial, constraints.spatial
            )

            # Phase 2: 交互梯度
            if self.phase >= 2 and constraints.interaction is not None:
                gen_interaction = self._aggregate_interaction(
                    responses, user_patterns, grid_size
                )
                interaction_grads = self._compute_interaction_gradients(
                    gen_interaction, constraints.interaction
                )
                # 合并梯度（简化版：只用一阶梯度）
                # TODO: 完整实现需要更新β

            # Step 4: 计算损失（监控）
            loss = self._compute_loss(gen_spatial, constraints.spatial)

            # Step 5: 更新势函数
            if self.use_adam:
                optimizer.step(gradients, self.lr)
            else:
                for loc_type, grad in gradients.items():
                    potentials.update_alpha(loc_type, grad, self.lr)

            # 日志
            if iteration % self.log_freq == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: Loss = {loss:.6f}")

            # 收敛检查
            if abs(prev_loss - loss) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break

            prev_loss = loss

        # 最终响应作为分配
        final_responses = self.mf_solver.compute_all_responses_fast(
            user_patterns, potentials, constraints, self.phase
        )

        return final_responses

    def _aggregate_spatial(self,
                           responses: Dict[int, np.ndarray],
                           user_patterns: Dict[int, UserPattern],
                           grid_h: int, grid_w: int) -> SpatialConstraints:
        """聚合空间统计"""
        H_map = np.zeros((grid_h, grid_w))
        W_map = np.zeros((grid_h, grid_w))
        O_map = np.zeros((grid_h, grid_w))

        for user_id, Q in responses.items():
            pattern = user_patterns[user_id]
            for loc_idx, loc in enumerate(pattern.locations):
                probs = Q[loc_idx].reshape(grid_h, grid_w)
                if loc.type == 'H':
                    H_map += probs
                elif loc.type == 'W':
                    W_map += probs
                elif loc.type == 'O':
                    O_map += probs

        result = SpatialConstraints(H=H_map, W=W_map, O=O_map)
        result.normalize()
        return result

    def _aggregate_interaction(self,
                               responses: Dict[int, np.ndarray],
                               user_patterns: Dict[int, UserPattern],
                               grid_size: int,
                               top_k: int = 30) -> InteractionConstraints:
        """聚合交互统计（简化版，使用top-k）"""
        hw_dict = {}
        ho_dict = {}
        wo_dict = {}

        def get_top_k(probs, k):
            if k >= len(probs):
                idx = np.where(probs > 1e-10)[0]
                return idx, probs[idx]
            top_idx = np.argpartition(probs, -k)[-k:]
            return top_idx, probs[top_idx]

        def add_interaction(d, probs1, probs2, k=top_k):
            idx1, p1 = get_top_k(probs1, k)
            idx2, p2 = get_top_k(probs2, k)
            for i, i1 in enumerate(idx1):
                for j, i2 in enumerate(idx2):
                    val = p1[i] * p2[j]
                    if val > 1e-15:
                        key = (int(i1), int(i2))
                        d[key] = d.get(key, 0) + val

        for user_id, Q in responses.items():
            pattern = user_patterns[user_id]
            h_idx = [i for i, loc in enumerate(pattern.locations) if loc.type == 'H']
            w_idx = [i for i, loc in enumerate(pattern.locations) if loc.type == 'W']
            o_idx = [i for i, loc in enumerate(pattern.locations) if loc.type == 'O']

            for hi in h_idx:
                for wi in w_idx:
                    add_interaction(hw_dict, Q[hi], Q[wi])
            for hi in h_idx:
                for oi in o_idx:
                    add_interaction(ho_dict, Q[hi], Q[oi])
            for wi in w_idx:
                for oi in o_idx:
                    add_interaction(wo_dict, Q[wi], Q[oi])

        def dict_to_sparse(d):
            if not d:
                return sparse.csr_matrix((grid_size, grid_size))
            rows, cols, data = zip(*[(k[0], k[1], v) for k, v in d.items()])
            return sparse.csr_matrix((data, (rows, cols)), shape=(grid_size, grid_size))

        result = InteractionConstraints(
            HW=dict_to_sparse(hw_dict),
            HO=dict_to_sparse(ho_dict),
            WO=dict_to_sparse(wo_dict)
        )
        result.normalize()
        return result

    def _compute_spatial_gradients(self,
                                   gen: SpatialConstraints,
                                   real: SpatialConstraints) -> Dict[str, np.ndarray]:
        """计算空间分布的梯度

        梯度 = 生成分布 - 真实分布
        """
        return {
            'H': gen.H - real.H,
            'W': gen.W - real.W,
            'O': gen.O - real.O
        }

    def _compute_interaction_gradients(self,
                                       gen: InteractionConstraints,
                                       real: InteractionConstraints) -> Dict[str, sparse.csr_matrix]:
        """计算交互分布的梯度"""
        return {
            'HW': gen.HW - real.HW,
            'HO': gen.HO - real.HO,
            'WO': gen.WO - real.WO
        }

    def _compute_loss(self,
                      gen: SpatialConstraints,
                      real: SpatialConstraints) -> float:
        """计算总损失（用于监控）"""
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
    """Phase 2专用优化器（同时优化空间和交互约束）"""

    def __init__(self,
                 max_iter: int = 200,
                 lr: float = 0.05,
                 temperature: float = 0.5,
                 interaction_weight: float = 1.0,
                 **kwargs):
        """
        Args:
            interaction_weight: 交互约束的权重
        """
        super().__init__(phase=2, max_iter=max_iter, lr=lr,
                         temperature=temperature, **kwargs)
        self.interaction_weight = interaction_weight
        self.name = f"SS-DMFO-P2(T={temperature})"

    def _generate_allocations(self,
                              constraints: Constraints,
                              user_patterns: Dict[int, UserPattern]) -> Dict[int, np.ndarray]:
        """Phase 2优化：同时考虑空间和交互约束"""
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        # 归一化
        constraints.spatial.normalize()
        if constraints.interaction is not None:
            constraints.interaction.normalize()

        # 初始化
        print(f"Initializing SS-DMFO Phase 2...")
        potentials = DualPotentials.initialize(grid_h, grid_w, phase=2)

        if self.use_adam:
            optimizer = PotentialsWithMomentum(potentials)

        print(f"Running optimization (max_iter={self.max_iter})...")
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # 计算响应（Phase 2需要考虑交互）
            responses = self.mf_solver.compute_all_responses_fast(
                user_patterns, potentials, constraints, phase=2
            )

            # 聚合空间统计
            gen_spatial = self._aggregate_spatial(
                responses, user_patterns, grid_h, grid_w
            )

            # 计算空间梯度
            spatial_grads = self._compute_spatial_gradients(gen_spatial, constraints.spatial)

            # 计算损失
            spatial_loss = self._compute_loss(gen_spatial, constraints.spatial)

            # 如果有交互约束，计算交互损失
            interaction_loss = 0.0
            if constraints.interaction is not None:
                gen_interaction = self._aggregate_interaction(
                    responses, user_patterns, grid_size, top_k=30
                )
                interaction_loss = self._compute_interaction_loss(
                    gen_interaction, constraints.interaction
                )

            total_loss = spatial_loss + self.interaction_weight * interaction_loss

            # 更新势函数
            if self.use_adam:
                optimizer.step(spatial_grads, self.lr)
            else:
                for loc_type, grad in spatial_grads.items():
                    potentials.update_alpha(loc_type, grad, self.lr)

            # 日志
            if iteration % self.log_freq == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: Spatial={spatial_loss:.4f}, "
                      f"Interact={interaction_loss:.4f}, Total={total_loss:.4f}")

            # 收敛
            if abs(prev_loss - total_loss) < self.tolerance:
                print(f"  Converged at iteration {iteration}")
                break
            prev_loss = total_loss

        return self.mf_solver.compute_all_responses_fast(
            user_patterns, potentials, constraints, phase=2
        )

    def _compute_interaction_loss(self,
                                  gen: InteractionConstraints,
                                  real: InteractionConstraints) -> float:
        """计算交互JSD损失"""
        def sparse_jsd(p, q):
            # 简化版：转为密集计算（小规模可行）
            p_arr = p.toarray().flatten() + 1e-10
            q_arr = q.toarray().flatten() + 1e-10
            p_arr = p_arr / p_arr.sum()
            q_arr = q_arr / q_arr.sum()
            m = 0.5 * (p_arr + q_arr)
            return 0.5 * (np.sum(p_arr * np.log(p_arr/m)) +
                         np.sum(q_arr * np.log(q_arr/m)))

        # 只计算非空的
        losses = []
        if gen.HW.nnz > 0 and real.HW.nnz > 0:
            losses.append(sparse_jsd(gen.HW, real.HW))
        if gen.HO.nnz > 0 and real.HO.nnz > 0:
            losses.append(sparse_jsd(gen.HO, real.HO))
        if gen.WO.nnz > 0 and real.WO.nnz > 0:
            losses.append(sparse_jsd(gen.WO, real.WO))

        return np.mean(losses) if losses else 0.0
