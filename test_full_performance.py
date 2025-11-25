#!/usr/bin/env python3
"""Full Performance Test with Visualization

Test SS-DMFO GPU with realistic parameters and visualize results.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_gpu import SSDMFOv3GPU, GPUConfig, HAS_TORCH
from ssdmfo.evaluation.metrics import MetricsCalculator

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_hw_distances(result, user_patterns, grid_h, grid_w, cell_size_km=1.0):
    """Compute Home-Work distances from result allocations"""
    distances = []

    for user_id, alloc in result.allocations.items():
        pattern = user_patterns[user_id]

        # Find H and W locations for this user
        h_indices = []
        w_indices = []
        for loc_idx, loc in enumerate(pattern.locations):
            if loc.type == 'H':
                h_indices.append(loc_idx)
            elif loc.type == 'W':
                w_indices.append(loc_idx)

        if not h_indices or not w_indices:
            continue

        # Get most probable cell for each H and W
        for h_idx in h_indices:
            h_cell = np.argmax(alloc[h_idx])
            h_row, h_col = h_cell // grid_w, h_cell % grid_w

            for w_idx in w_indices:
                w_cell = np.argmax(alloc[w_idx])
                w_row, w_col = w_cell // grid_w, w_cell % grid_w

                # Euclidean distance in km
                dist = np.sqrt((h_row - w_row)**2 + (h_col - w_col)**2) * cell_size_km
                distances.append(dist)

    return np.array(distances)


def compute_real_hw_distances(constraints, cell_size_km=1.0):
    """Compute Home-Work distances from real interaction constraints"""
    hw_coo = constraints.interaction.HW.tocoo()

    if hw_coo.nnz == 0:
        return np.array([])

    # Sample if too many
    max_samples = 10000
    if hw_coo.nnz > max_samples:
        idx = np.random.choice(hw_coo.nnz, max_samples, replace=False,
                               p=hw_coo.data / hw_coo.data.sum())
        rows, cols, weights = hw_coo.row[idx], hw_coo.col[idx], hw_coo.data[idx]
    else:
        rows, cols, weights = hw_coo.row, hw_coo.col, hw_coo.data

    grid_w = constraints.grid_w

    distances = []
    for r, c, w in zip(rows, cols, weights):
        h_row, h_col = r // grid_w, r % grid_w
        w_row, w_col = c // grid_w, c % grid_w
        dist = np.sqrt((h_row - w_row)**2 + (h_col - w_col)**2) * cell_size_km
        # Weight by interaction strength
        distances.extend([dist] * max(1, int(w * 1000)))

    return np.array(distances)


def visualize_results(constraints, gen_spatial, gen_interaction,
                      hw_distances_gen, hw_distances_real,
                      save_path='results_visualization.png'):
    """Create comprehensive visualization"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Home spatial distribution comparison
    ax = axes[0, 0]
    real_H = constraints.spatial.H.flatten()
    gen_H = gen_spatial.H.flatten()

    # Show as 2D heatmaps side by side
    ax.set_title('Home Distribution (Real vs Gen)')
    real_2d = real_H.reshape(constraints.grid_h, constraints.grid_w)
    gen_2d = gen_H.reshape(constraints.grid_h, constraints.grid_w)

    # Plot difference
    diff = gen_2d - real_2d
    im = ax.imshow(diff, cmap='RdBu', vmin=-0.001, vmax=0.001)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Gen - Real')

    # 2. Work spatial distribution comparison
    ax = axes[0, 1]
    real_W = constraints.spatial.W.flatten()
    gen_W = gen_spatial.W.flatten()

    real_2d = real_W.reshape(constraints.grid_h, constraints.grid_w)
    gen_2d = gen_W.reshape(constraints.grid_h, constraints.grid_w)
    diff = gen_2d - real_2d

    ax.set_title('Work Distribution (Gen - Real)')
    im = ax.imshow(diff, cmap='RdBu', vmin=-0.001, vmax=0.001)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Gen - Real')

    # 3. Spatial marginal comparison (1D)
    ax = axes[0, 2]
    # Compare marginal distributions along one axis
    real_H_marginal = real_H.reshape(constraints.grid_h, constraints.grid_w).sum(axis=1)
    gen_H_marginal = gen_H.reshape(constraints.grid_h, constraints.grid_w).sum(axis=1)

    ax.plot(real_H_marginal, 'r--', label='Real H', alpha=0.8)
    ax.plot(gen_H_marginal, 'b-', label='Gen H', alpha=0.8)
    ax.set_title('Home Row Marginal')
    ax.set_xlabel('Row')
    ax.set_ylabel('Probability')
    ax.legend()

    # 4. Home-Work Distance Distribution
    ax = axes[1, 0]
    if len(hw_distances_gen) > 0 and len(hw_distances_real) > 0:
        bins = np.linspace(0, 80, 40)
        ax.hist(hw_distances_real, bins=bins, density=True, alpha=0.5,
                label=f'Real (N={len(hw_distances_real)})', color='red')
        ax.hist(hw_distances_gen, bins=bins, density=True, alpha=0.5,
                label=f'Gen (N={len(hw_distances_gen)})', color='blue')
        ax.set_title('Home-Work Distance Distribution')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Density')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No distance data', ha='center', va='center')

    # 5. Interaction HW heatmap (sampled)
    ax = axes[1, 1]
    real_hw = constraints.interaction.HW.toarray()
    gen_hw = gen_interaction.HW.toarray()

    # Show top region only (where most mass is)
    real_sum_row = real_hw.sum(axis=1)
    real_sum_col = real_hw.sum(axis=0)
    top_rows = np.argsort(real_sum_row)[-50:]
    top_cols = np.argsort(real_sum_col)[-50:]

    real_sub = real_hw[np.ix_(top_rows, top_cols)]
    gen_sub = gen_hw[np.ix_(top_rows, top_cols)]

    diff_sub = gen_sub - real_sub
    im = ax.imshow(diff_sub, cmap='RdBu', vmin=-0.0001, vmax=0.0001)
    ax.set_title('HW Interaction (Gen-Real, top 50×50)')
    ax.set_xlabel('Work cells')
    ax.set_ylabel('Home cells')
    plt.colorbar(im, ax=ax)

    # 6. Summary metrics
    ax = axes[1, 2]
    ax.axis('off')

    # Compute JSD
    def jsd(p, q):
        p = p.flatten() + 1e-10
        q = q.flatten() + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log(p/m)) + np.sum(q * np.log(q/m)))

    jsd_h = jsd(gen_spatial.H, constraints.spatial.H)
    jsd_w = jsd(gen_spatial.W, constraints.spatial.W)
    jsd_o = jsd(gen_spatial.O, constraints.spatial.O)

    summary_text = f"""
    Performance Summary
    ==================

    Spatial JSD:
      H: {jsd_h:.6f}
      W: {jsd_w:.6f}
      O: {jsd_o:.6f}
      Mean: {(jsd_h+jsd_w+jsd_o)/3:.6f}

    Interaction:
      HW nnz: {gen_interaction.HW.nnz:,} vs {constraints.interaction.HW.nnz:,}
      HO nnz: {gen_interaction.HO.nnz:,} vs {constraints.interaction.HO.nnz:,}
      WO nnz: {gen_interaction.WO.nnz:,} vs {constraints.interaction.WO.nnz:,}

    H-W Distance:
      Gen mean: {np.mean(hw_distances_gen):.1f} km
      Real mean: {np.mean(hw_distances_real):.1f} km
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def main():
    """Main test function"""
    print("=" * 70)
    print("SS-DMFO GPU - FULL PERFORMANCE TEST")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed.")
        return

    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_start = time.time()

    # Load data
    print("\n[1] Loading data...")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    print(f"Grid: {constraints.grid_h} x {constraints.grid_w}")
    print(f"HW interactions: {constraints.interaction.HW.nnz:,}")

    # Load users - use more for realistic test
    n_users = 1000  # Realistic test size
    print(f"\n[2] Loading {n_users} user patterns...")
    user_patterns = loader.load_user_patterns(n_users=n_users)
    print(f"Loaded {len(user_patterns)} users")

    calculator = MetricsCalculator()

    # =====================================
    # IPF Baseline
    # =====================================
    print("\n" + "=" * 70)
    print("BASELINE: IPF")
    print("=" * 70)

    ipf = IterativeProportionalFitting(max_iter=20)
    ipf_start = time.time()
    ipf_result = ipf.run(constraints, user_patterns)

    ipf_spatial = ipf_result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    ipf_spatial.normalize()

    ipf_interaction = ipf_result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w, top_k=200
    )
    ipf_interaction.normalize()

    ipf_metrics = calculator.compute_all_metrics(
        ipf_spatial, constraints.spatial,
        ipf_interaction, constraints.interaction
    )
    calculator.print_metrics(ipf_metrics, phase=2)
    print(f"IPF time: {time.time() - ipf_start:.1f}s")

    # Compute H-W distances for IPF
    hw_dist_ipf = compute_hw_distances(ipf_result, user_patterns,
                                       constraints.grid_h, constraints.grid_w)

    # =====================================
    # SS-DMFO GPU
    # =====================================
    print("\n" + "=" * 70)
    print("SS-DMFO GPU (Full Test)")
    print("=" * 70)

    config = GPUConfig(
        max_iter=150,                      # More iterations
        gpu_batch_size=500,
        lr_alpha=0.15,
        lr_beta=0.08,
        mfvi_iter=3,
        temp_init=2.0,
        temp_final=0.3,
        gumbel_scale=0.3,
        gumbel_decay=0.99,
        gumbel_final=0.01,
        interaction_freq=2,
        top_k=500,                         # 500 important cells
        log_freq=10,
        early_stop_patience=25,
        phase_separation=True,
        phase1_ratio=0.15,                 # 15% for spatial only
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    gpu_method = SSDMFOv3GPU(config)
    gpu_start = time.time()
    gpu_result = gpu_method.run(constraints, user_patterns)

    print("\nComputing final statistics...")
    gpu_spatial = gpu_result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    gpu_spatial.normalize()

    gpu_interaction = gpu_result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w, top_k=500
    )
    gpu_interaction.normalize()

    gpu_metrics = calculator.compute_all_metrics(
        gpu_spatial, constraints.spatial,
        gpu_interaction, constraints.interaction
    )
    calculator.print_metrics(gpu_metrics, phase=2)
    print(f"GPU time: {time.time() - gpu_start:.1f}s")

    # Compute H-W distances for GPU
    hw_dist_gpu = compute_hw_distances(gpu_result, user_patterns,
                                       constraints.grid_h, constraints.grid_w)

    # Real H-W distances
    hw_dist_real = compute_real_hw_distances(constraints)

    # =====================================
    # Summary
    # =====================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<15} {'Spatial JSD':<12} {'Interact JSD':<12} {'Time(s)':<10}")
    print("-" * 50)
    print(f"{'IPF':<15} {ipf_metrics['jsd_mean']:<12.6f} {ipf_metrics.get('jsd_interaction_mean', 0):<12.4f} {time.time()-ipf_start:<10.1f}")
    print(f"{'SS-DMFO GPU':<15} {gpu_metrics['jsd_mean']:<12.6f} {gpu_metrics.get('jsd_interaction_mean', 0):<12.4f} {time.time()-gpu_start:<10.1f}")

    # =====================================
    # Visualization
    # =====================================
    print("\n[3] Creating visualization...")
    visualize_results(
        constraints, gpu_spatial, gpu_interaction,
        hw_dist_gpu, hw_dist_real,
        save_path='ssdmfo_results.png'
    )

    # Also create IPF visualization for comparison
    visualize_results(
        constraints, ipf_spatial, ipf_interaction,
        hw_dist_ipf, hw_dist_real,
        save_path='ipf_results.png'
    )

    print(f"\nTotal time: {time.time() - total_start:.1f}s")
    print("\nDone! Check ssdmfo_results.png and ipf_results.png")


if __name__ == '__main__':
    main()
