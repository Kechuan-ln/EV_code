"""对偶变量（势函数）定义

SS-DMFO使用对偶优化框架，势函数是拉格朗日乘子的连续版本。
- α_c(g): 一阶势函数，对应空间分布约束
- β_cc'(g,g'): 二阶势函数，对应交互约束
"""

import numpy as np
from scipy import sparse
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class DualPotentials:
    """对偶变量容器

    一阶势函数 α: 控制每种地点类型的空间分布
    二阶势函数 β: 控制不同类型地点的联合分布（交互）
    """
    # 一阶势函数 (grid_h, grid_w)
    alpha_H: np.ndarray
    alpha_W: np.ndarray
    alpha_O: np.ndarray

    # 二阶势函数 (grid_size, grid_size) - 稀疏存储
    beta_HW: Optional[sparse.csr_matrix] = None
    beta_HO: Optional[sparse.csr_matrix] = None
    beta_WO: Optional[sparse.csr_matrix] = None

    @classmethod
    def initialize(cls, grid_h: int, grid_w: int,
                   phase: int = 1,
                   init_scale: float = 0.01) -> 'DualPotentials':
        """初始化势函数

        Args:
            grid_h: 栅格高度
            grid_w: 栅格宽度
            phase: 1=仅一阶, 2=一阶+二阶
            init_scale: 初始化尺度（小随机值）
        """
        grid_size = grid_h * grid_w

        # 一阶势函数：小随机初始化
        alpha_H = np.random.randn(grid_h, grid_w) * init_scale
        alpha_W = np.random.randn(grid_h, grid_w) * init_scale
        alpha_O = np.random.randn(grid_h, grid_w) * init_scale

        potentials = cls(
            alpha_H=alpha_H,
            alpha_W=alpha_W,
            alpha_O=alpha_O
        )

        if phase >= 2:
            # 二阶势函数：初始化为零稀疏矩阵
            # 实际非零位置会在优化过程中根据约束动态确定
            potentials.beta_HW = sparse.csr_matrix((grid_size, grid_size))
            potentials.beta_HO = sparse.csr_matrix((grid_size, grid_size))
            potentials.beta_WO = sparse.csr_matrix((grid_size, grid_size))

        return potentials

    def get_alpha(self, loc_type: str) -> np.ndarray:
        """获取指定类型的一阶势函数"""
        if loc_type == 'H':
            return self.alpha_H
        elif loc_type == 'W':
            return self.alpha_W
        elif loc_type == 'O':
            return self.alpha_O
        else:
            raise ValueError(f"Unknown location type: {loc_type}")

    def get_beta(self, type1: str, type2: str) -> Optional[sparse.csr_matrix]:
        """获取指定类型对的二阶势函数"""
        key = ''.join(sorted([type1, type2]))
        if key == 'HW':
            return self.beta_HW
        elif key == 'HO':
            return self.beta_HO
        elif key == 'OW':
            return self.beta_WO
        return None

    def update_alpha(self, loc_type: str, gradient: np.ndarray, lr: float):
        """更新一阶势函数"""
        if loc_type == 'H':
            self.alpha_H -= lr * gradient
        elif loc_type == 'W':
            self.alpha_W -= lr * gradient
        elif loc_type == 'O':
            self.alpha_O -= lr * gradient

    def update_beta(self, type1: str, type2: str,
                    gradient: sparse.csr_matrix, lr: float):
        """更新二阶势函数"""
        key = ''.join(sorted([type1, type2]))
        if key == 'HW' and self.beta_HW is not None:
            self.beta_HW = self.beta_HW - lr * gradient
        elif key == 'HO' and self.beta_HO is not None:
            self.beta_HO = self.beta_HO - lr * gradient
        elif key == 'OW' and self.beta_WO is not None:
            self.beta_WO = self.beta_WO - lr * gradient


class PotentialsWithMomentum:
    """带动量的势函数优化器（类似Adam）"""

    def __init__(self, potentials: DualPotentials,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        self.potentials = potentials
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # 一阶矩（动量）
        self.m_alpha = {
            'H': np.zeros_like(potentials.alpha_H),
            'W': np.zeros_like(potentials.alpha_W),
            'O': np.zeros_like(potentials.alpha_O),
        }

        # 二阶矩（RMSprop）
        self.v_alpha = {
            'H': np.zeros_like(potentials.alpha_H),
            'W': np.zeros_like(potentials.alpha_W),
            'O': np.zeros_like(potentials.alpha_O),
        }

    def step(self, gradients: Dict[str, np.ndarray], lr: float):
        """执行一步Adam更新"""
        self.t += 1

        for loc_type in ['H', 'W', 'O']:
            if loc_type not in gradients:
                continue

            g = gradients[loc_type]

            # 更新一阶矩
            self.m_alpha[loc_type] = (self.beta1 * self.m_alpha[loc_type] +
                                      (1 - self.beta1) * g)

            # 更新二阶矩
            self.v_alpha[loc_type] = (self.beta2 * self.v_alpha[loc_type] +
                                      (1 - self.beta2) * g**2)

            # 偏差校正
            m_hat = self.m_alpha[loc_type] / (1 - self.beta1**self.t)
            v_hat = self.v_alpha[loc_type] / (1 - self.beta2**self.t)

            # 更新参数
            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.potentials.update_alpha(loc_type, update, lr=1.0)
