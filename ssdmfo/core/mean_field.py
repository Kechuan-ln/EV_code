"""Mean Field Variational Inference (MFVI) - Vectorized Version

Given potentials, compute optimal response distribution Q_i for each user.
Core idea: each semantic location's distribution is updated independently,
considering the influence of potentials.

Q_i(l -> g) proportional to exp(-alpha_c(g) - sum_{l'} beta_{cc'}(g, g') * Q_i(l' -> g'))
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import sparse

from ..data.structures import UserPattern, Constraints
from .potentials import DualPotentials


class MeanFieldSolver:
    """Mean Field Variational Inference Solver"""

    def __init__(self,
                 temperature: float = 1.0,
                 max_iter: int = 10,
                 tolerance: float = 1e-4,
                 damping: float = 0.5):
        """
        Args:
            temperature: temperature parameter, smaller = sharper distribution
            max_iter: maximum MFVI iterations
            tolerance: convergence tolerance
            damping: damping coefficient to prevent oscillation
        """
        self.temperature = temperature
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.damping = damping

    def compute_user_response(self,
                              user: UserPattern,
                              potentials: DualPotentials,
                              grid_h: int, grid_w: int,
                              phase: int = 1) -> np.ndarray:
        """Compute optimal response distribution for a single user

        Args:
            user: user pattern
            potentials: current potentials
            grid_h, grid_w: grid dimensions
            phase: optimization phase

        Returns:
            Q: shape (n_locations, grid_size), spatial distribution for each semantic location
        """
        n_locs = len(user.locations)
        grid_size = grid_h * grid_w

        # Initialize: uniform distribution
        Q = np.ones((n_locs, grid_size)) / grid_size

        # Pre-compute type and corresponding first-order potential for each location
        loc_types = [loc.type for loc in user.locations]
        alpha_flat = {
            'H': potentials.alpha_H.flatten(),
            'W': potentials.alpha_W.flatten(),
            'O': potentials.alpha_O.flatten()
        }

        for iteration in range(self.max_iter):
            Q_old = Q.copy()

            # Update each location
            for loc_idx in range(n_locs):
                loc_type = loc_types[loc_idx]

                # First-order field: from alpha
                field = alpha_flat[loc_type].copy()

                # Second-order field: from beta and other locations' Q (Phase 2 only)
                if phase >= 2:
                    field += self._compute_interaction_field(
                        loc_idx, loc_types, Q, potentials, grid_size
                    )

                # Boltzmann distribution
                log_q = -field / self.temperature
                log_q -= log_q.max()  # numerical stability
                q_new = np.exp(log_q)
                q_new /= q_new.sum() + 1e-10

                # Damped update
                Q[loc_idx] = self.damping * q_new + (1 - self.damping) * Q_old[loc_idx]

            # Check convergence
            diff = np.abs(Q - Q_old).max()
            if diff < self.tolerance:
                break

        return Q

    def _compute_interaction_field(self,
                                   loc_idx: int,
                                   loc_types: List[str],
                                   Q: np.ndarray,
                                   potentials: DualPotentials,
                                   grid_size: int) -> np.ndarray:
        """Compute second-order interaction field

        field_l(g) = sum_{l' != l} sum_{g'} beta_{cc'}(g,g') * Q(l' -> g')
        """
        field = np.zeros(grid_size)
        loc_type = loc_types[loc_idx]

        for other_idx, other_type in enumerate(loc_types):
            if other_idx == loc_idx:
                continue

            # Get corresponding beta matrix
            beta = potentials.get_beta(loc_type, other_type)
            if beta is None or beta.nnz == 0:
                continue

            # Sparse matrix-vector multiplication: beta @ Q[other]
            q_other = Q[other_idx]
            field += beta.dot(q_other)

        return field

    def compute_all_responses(self,
                              user_patterns: Dict[int, UserPattern],
                              potentials: DualPotentials,
                              constraints: Constraints,
                              phase: int = 1) -> Dict[int, np.ndarray]:
        """Compute response distributions for all users

        Returns:
            responses: {user_id: Q} dictionary
        """
        responses = {}
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w

        for user_id, user in user_patterns.items():
            Q = self.compute_user_response(
                user, potentials, grid_h, grid_w, phase
            )
            responses[user_id] = Q

        return responses


class FastMeanFieldSolver(MeanFieldSolver):
    """Optimized Mean Field Solver (batch processing + vectorization)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_all_responses_fast(self,
                                   user_patterns: Dict[int, UserPattern],
                                   potentials: DualPotentials,
                                   constraints: Constraints,
                                   phase: int = 1) -> Dict[int, np.ndarray]:
        """Fast computation of all user responses (vectorized)

        For Phase 1: all same-type locations have same distribution (template).
        For Phase 2: use vectorized batch computation.
        """
        grid_h = constraints.grid_h
        grid_w = constraints.grid_w
        grid_size = grid_h * grid_w

        if phase == 1:
            return self._compute_phase1_fast(user_patterns, potentials, grid_size)
        else:
            return self._compute_phase2_fast(user_patterns, potentials, grid_size)

    def _compute_phase1_fast(self,
                             user_patterns: Dict[int, UserPattern],
                             potentials: DualPotentials,
                             grid_size: int) -> Dict[int, np.ndarray]:
        """Phase 1: all same-type locations have same distribution"""
        # Pre-compute templates for each type
        Q_templates = {}
        for loc_type in ['H', 'W', 'O']:
            alpha = potentials.get_alpha(loc_type).flatten()
            log_q = -alpha / self.temperature
            log_q -= log_q.max()
            q = np.exp(log_q)
            q /= q.sum() + 1e-10
            Q_templates[loc_type] = q

        # Assign template to each user (vectorized)
        responses = {}
        for user_id, user in user_patterns.items():
            n_locs = len(user.locations)
            Q = np.zeros((n_locs, grid_size))
            for loc_idx, loc in enumerate(user.locations):
                Q[loc_idx] = Q_templates[loc.type]
            responses[user_id] = Q

        return responses

    def _compute_phase2_fast(self,
                             user_patterns: Dict[int, UserPattern],
                             potentials: DualPotentials,
                             grid_size: int) -> Dict[int, np.ndarray]:
        """Phase 2: vectorized MFVI with simplified approximation

        Key optimization: Instead of full MFVI iteration per user,
        use a single-pass approximation based on alpha potentials
        with a small perturbation from beta.
        """
        # Pre-compute base distributions from alpha
        alpha_flat = {
            'H': potentials.alpha_H.flatten(),
            'W': potentials.alpha_W.flatten(),
            'O': potentials.alpha_O.flatten()
        }

        # Compute base Q for each type
        Q_base = {}
        for loc_type in ['H', 'W', 'O']:
            log_q = -alpha_flat[loc_type] / self.temperature
            log_q -= log_q.max()
            q = np.exp(log_q)
            q /= q.sum() + 1e-10
            Q_base[loc_type] = q

        # For Phase 2, we do a simplified 1-iteration update
        # This is much faster than full MFVI while still incorporating beta
        responses = {}

        for user_id, user in user_patterns.items():
            n_locs = len(user.locations)
            loc_types = [loc.type for loc in user.locations]

            # Start with base distribution
            Q = np.zeros((n_locs, grid_size))
            for i, lt in enumerate(loc_types):
                Q[i] = Q_base[lt].copy()

            # Single refinement pass incorporating beta
            if self.max_iter > 0:
                Q = self._refine_with_beta_vectorized(
                    Q, loc_types, potentials, alpha_flat, grid_size
                )

            responses[user_id] = Q

        return responses

    def _refine_with_beta_vectorized(self,
                                     Q: np.ndarray,
                                     loc_types: List[str],
                                     potentials: DualPotentials,
                                     alpha_flat: Dict[str, np.ndarray],
                                     grid_size: int) -> np.ndarray:
        """Single refinement pass incorporating beta potentials (vectorized)"""
        n_locs = len(loc_types)

        # Build interaction field for all locations at once
        fields = np.zeros((n_locs, grid_size))

        # Add alpha contribution
        for i, lt in enumerate(loc_types):
            fields[i] = alpha_flat[lt]

        # Add beta contribution (interaction with other locations)
        for i in range(n_locs):
            for j in range(n_locs):
                if i == j:
                    continue
                beta = potentials.get_beta(loc_types[i], loc_types[j])
                if beta is not None and beta.nnz > 0:
                    # beta @ Q[j] gives interaction field
                    fields[i] += beta.dot(Q[j])

        # Convert fields to distributions (vectorized softmax)
        log_q = -fields / self.temperature
        log_q -= log_q.max(axis=1, keepdims=True)  # stability
        Q_new = np.exp(log_q)
        Q_new /= Q_new.sum(axis=1, keepdims=True) + 1e-10

        # Damped update
        return self.damping * Q_new + (1 - self.damping) * Q
