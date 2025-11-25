"""Dual Potentials (Lagrange multipliers)

SS-DMFO uses dual optimization framework:
- alpha_c(g): first-order potential for spatial distribution constraints
- beta_cc'(g,g'): second-order potential for interaction constraints
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class DualPotentials:
    """Container for dual variables (potentials)

    First-order potentials alpha: control spatial distribution of each location type
    Second-order potentials beta: control joint distribution (interaction)
    """
    # First-order potentials (grid_h, grid_w)
    alpha_H: np.ndarray
    alpha_W: np.ndarray
    alpha_O: np.ndarray

    # Second-order potentials (grid_size, grid_size) - sparse storage
    beta_HW: Optional[sparse.csr_matrix] = None
    beta_HO: Optional[sparse.csr_matrix] = None
    beta_WO: Optional[sparse.csr_matrix] = None

    @classmethod
    def initialize(cls, grid_h: int, grid_w: int,
                   phase: int = 1,
                   init_scale: float = 0.01) -> 'DualPotentials':
        """Initialize potentials

        Args:
            grid_h: grid height
            grid_w: grid width
            phase: 1=first-order only, 2=first+second order
            init_scale: initialization scale (small random values)
        """
        grid_size = grid_h * grid_w

        # First-order: small random initialization
        alpha_H = np.random.randn(grid_h, grid_w) * init_scale
        alpha_W = np.random.randn(grid_h, grid_w) * init_scale
        alpha_O = np.random.randn(grid_h, grid_w) * init_scale

        potentials = cls(
            alpha_H=alpha_H,
            alpha_W=alpha_W,
            alpha_O=alpha_O
        )

        if phase >= 2:
            # Second-order: initialize as zero sparse matrices
            potentials.beta_HW = sparse.csr_matrix((grid_size, grid_size))
            potentials.beta_HO = sparse.csr_matrix((grid_size, grid_size))
            potentials.beta_WO = sparse.csr_matrix((grid_size, grid_size))

        return potentials

    def get_alpha(self, loc_type: str) -> np.ndarray:
        """Get first-order potential for specified type"""
        if loc_type == 'H':
            return self.alpha_H
        elif loc_type == 'W':
            return self.alpha_W
        elif loc_type == 'O':
            return self.alpha_O
        else:
            raise ValueError(f"Unknown location type: {loc_type}")

    def get_beta(self, type1: str, type2: str) -> Optional[sparse.csr_matrix]:
        """Get second-order potential for specified type pair"""
        key = ''.join(sorted([type1, type2]))
        if key == 'HW':
            return self.beta_HW
        elif key == 'HO':
            return self.beta_HO
        elif key == 'OW':
            return self.beta_WO
        return None

    def update_alpha(self, loc_type: str, gradient: np.ndarray, lr: float):
        """Update first-order potential"""
        if loc_type == 'H':
            self.alpha_H -= lr * gradient
        elif loc_type == 'W':
            self.alpha_W -= lr * gradient
        elif loc_type == 'O':
            self.alpha_O -= lr * gradient

    def update_beta(self, type1: str, type2: str,
                    gradient: sparse.csr_matrix, lr: float):
        """Update second-order potential"""
        key = ''.join(sorted([type1, type2]))
        if key == 'HW' and self.beta_HW is not None:
            self.beta_HW = self.beta_HW - lr * gradient
        elif key == 'HO' and self.beta_HO is not None:
            self.beta_HO = self.beta_HO - lr * gradient
        elif key == 'OW' and self.beta_WO is not None:
            self.beta_WO = self.beta_WO - lr * gradient


class PotentialsWithMomentum:
    """Optimizer with momentum (Adam-like)"""

    def __init__(self, potentials: DualPotentials,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        self.potentials = potentials
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # First moment (momentum)
        self.m_alpha = {
            'H': np.zeros_like(potentials.alpha_H),
            'W': np.zeros_like(potentials.alpha_W),
            'O': np.zeros_like(potentials.alpha_O),
        }

        # Second moment (RMSprop)
        self.v_alpha = {
            'H': np.zeros_like(potentials.alpha_H),
            'W': np.zeros_like(potentials.alpha_W),
            'O': np.zeros_like(potentials.alpha_O),
        }

    def step(self, gradients: Dict[str, np.ndarray], lr: float):
        """Execute one Adam update step"""
        self.t += 1

        for loc_type in ['H', 'W', 'O']:
            if loc_type not in gradients:
                continue

            g = gradients[loc_type]

            # Update first moment
            self.m_alpha[loc_type] = (self.beta1 * self.m_alpha[loc_type] +
                                      (1 - self.beta1) * g)

            # Update second moment
            self.v_alpha[loc_type] = (self.beta2 * self.v_alpha[loc_type] +
                                      (1 - self.beta2) * g**2)

            # Bias correction
            m_hat = self.m_alpha[loc_type] / (1 - self.beta1**self.t)
            v_hat = self.v_alpha[loc_type] / (1 - self.beta2**self.t)

            # Update parameters
            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.potentials.update_alpha(loc_type, update, lr=1.0)
