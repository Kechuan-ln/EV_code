"""评估指标计算"""
import numpy as np
from typing import Dict, Optional
from scipy import sparse
from ..data.structures import SpatialConstraints, InteractionConstraints


class MetricsCalculator:
    """计算各种评估指标"""

    def __init__(self):
        self.epsilon = 1e-10  # 防止log(0)和除零

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度: KL(p||q) = Σ p log(p/q)"""
        p = p.flatten() + self.epsilon
        q = q.flatten() + self.epsilon

        # 归一化
        p = p / p.sum()
        q = q / q.sum()

        kl = np.sum(p * np.log(p / q))
        return kl

    def jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算JS散度: JSD(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), m=(p+q)/2"""
        p = p.flatten() + self.epsilon
        q = q.flatten() + self.epsilon

        # 归一化
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)
        jsd = 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)
        return jsd

    def total_variation_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算Total Variation距离: TV(p,q) = 0.5 * Σ|p-q|"""
        p = p.flatten()
        q = q.flatten()

        # 归一化
        p = p / (p.sum() + self.epsilon)
        q = q / (q.sum() + self.epsilon)

        tv = 0.5 * np.sum(np.abs(p - q))
        return tv

    def sparse_jensen_shannon_divergence(self, p: sparse.spmatrix, q: sparse.spmatrix) -> float:
        """计算稀疏矩阵之间的JS散度

        由于稀疏矩阵可能很大，只在非零位置计算
        """
        # 转换为COO格式便于处理
        p_coo = p.tocoo()
        q_coo = q.tocoo()

        # 收集所有非零位置
        p_dict = {(r, c): v for r, c, v in zip(p_coo.row, p_coo.col, p_coo.data)}
        q_dict = {(r, c): v for r, c, v in zip(q_coo.row, q_coo.col, q_coo.data)}

        all_keys = set(p_dict.keys()) | set(q_dict.keys())

        if not all_keys:
            return 0.0

        # 构建密集向量（只包含非零位置）
        p_vals = np.array([p_dict.get(k, 0) for k in all_keys])
        q_vals = np.array([q_dict.get(k, 0) for k in all_keys])

        # 归一化
        p_sum = p_vals.sum() + self.epsilon
        q_sum = q_vals.sum() + self.epsilon
        p_vals = p_vals / p_sum + self.epsilon
        q_vals = q_vals / q_sum + self.epsilon

        # 计算JSD
        m = 0.5 * (p_vals + q_vals)
        kl_pm = np.sum(p_vals * np.log(p_vals / m))
        kl_qm = np.sum(q_vals * np.log(q_vals / m))
        jsd = 0.5 * (kl_pm + kl_qm)

        return jsd

    def compute_spatial_metrics(self,
                                generated: SpatialConstraints,
                                real: SpatialConstraints) -> Dict[str, float]:
        """计算空间分布指标"""
        metrics = {}

        # JSD for each type
        metrics['jsd_H'] = self.jensen_shannon_divergence(generated.H, real.H)
        metrics['jsd_W'] = self.jensen_shannon_divergence(generated.W, real.W)
        metrics['jsd_O'] = self.jensen_shannon_divergence(generated.O, real.O)

        # Mean JSD
        metrics['jsd_mean'] = np.mean([metrics['jsd_H'],
                                      metrics['jsd_W'],
                                      metrics['jsd_O']])

        # Total Variation Distance
        metrics['tvd_H'] = self.total_variation_distance(generated.H, real.H)
        metrics['tvd_W'] = self.total_variation_distance(generated.W, real.W)
        metrics['tvd_O'] = self.total_variation_distance(generated.O, real.O)

        metrics['tvd_mean'] = np.mean([metrics['tvd_H'],
                                      metrics['tvd_W'],
                                      metrics['tvd_O']])

        return metrics

    def compute_interaction_metrics(self,
                                   generated: InteractionConstraints,
                                   real: InteractionConstraints) -> Dict[str, float]:
        """计算交互分布指标"""
        metrics = {}

        # JSD for each interaction type
        metrics['jsd_HW'] = self.sparse_jensen_shannon_divergence(generated.HW, real.HW)
        metrics['jsd_HO'] = self.sparse_jensen_shannon_divergence(generated.HO, real.HO)
        metrics['jsd_WO'] = self.sparse_jensen_shannon_divergence(generated.WO, real.WO)

        # Mean JSD
        metrics['jsd_interaction_mean'] = np.mean([
            metrics['jsd_HW'], metrics['jsd_HO'], metrics['jsd_WO']
        ])

        return metrics

    def compute_all_metrics(self,
                           generated_spatial: SpatialConstraints,
                           real_spatial: SpatialConstraints,
                           generated_interaction: Optional[InteractionConstraints] = None,
                           real_interaction: Optional[InteractionConstraints] = None) -> Dict[str, float]:
        """计算所有指标（空间 + 交互）"""
        metrics = self.compute_spatial_metrics(generated_spatial, real_spatial)

        if generated_interaction is not None and real_interaction is not None:
            interaction_metrics = self.compute_interaction_metrics(
                generated_interaction, real_interaction
            )
            metrics.update(interaction_metrics)

            # 综合指标
            metrics['jsd_total_mean'] = np.mean([
                metrics['jsd_mean'],  # 空间
                metrics['jsd_interaction_mean']  # 交互
            ])
        else:
            metrics['jsd_total_mean'] = metrics['jsd_mean']

        return metrics

    def print_metrics(self, metrics: Dict[str, float], phase: int = 1):
        """打印指标"""
        print("\nMetrics:")
        print("-" * 50)

        # JSD指标（空间）
        print("Spatial JSD (Jensen-Shannon Divergence):")
        print(f"  H: {metrics['jsd_H']:.6f}")
        print(f"  W: {metrics['jsd_W']:.6f}")
        print(f"  O: {metrics['jsd_O']:.6f}")
        print(f"  Mean: {metrics['jsd_mean']:.6f}")

        # 交互指标（Phase 2+）
        if 'jsd_HW' in metrics:
            print("\nInteraction JSD:")
            print(f"  HW: {metrics['jsd_HW']:.6f}")
            print(f"  HO: {metrics['jsd_HO']:.6f}")
            print(f"  WO: {metrics['jsd_WO']:.6f}")
            print(f"  Mean: {metrics['jsd_interaction_mean']:.6f}")

        # 综合指标
        if 'jsd_total_mean' in metrics:
            print(f"\nTotal JSD Mean: {metrics['jsd_total_mean']:.6f}")

        # TVD指标
        print("\nTotal Variation Distance:")
        print(f"  H: {metrics['tvd_H']:.6f}")
        print(f"  W: {metrics['tvd_W']:.6f}")
        print(f"  O: {metrics['tvd_O']:.6f}")
        print(f"  Mean: {metrics['tvd_mean']:.6f}")

        print("-" * 50)


def test_metrics():
    """测试指标计算"""
    calc = MetricsCalculator()

    # 创建测试数据
    p = np.array([[1, 2, 3], [4, 5, 6]])
    q = np.array([[1.1, 2.2, 2.9], [3.8, 5.1, 6.2]])

    # 测试JSD
    jsd = calc.jensen_shannon_divergence(p, q)
    print(f"JSD: {jsd:.6f}")

    # 测试TVD
    tvd = calc.total_variation_distance(p, q)
    print(f"TVD: {tvd:.6f}")

    # 相同分布应该接近0
    jsd_same = calc.jensen_shannon_divergence(p, p)
    tvd_same = calc.total_variation_distance(p, p)
    print(f"\nSame distribution:")
    print(f"  JSD: {jsd_same:.10f}")
    print(f"  TVD: {tvd_same:.10f}")


if __name__ == '__main__':
    test_metrics()
