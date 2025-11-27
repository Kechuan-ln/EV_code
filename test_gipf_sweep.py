#!/usr/bin/env python3
"""Parameter sweep for SS-DMFO 4.0 G-IPF Optimizer

Tests multiple configurations and saves results to JSON for analysis.
"""

import sys
import os
import time
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting
from ssdmfo.core.optimizer_gipf import SSDMFOGIPF, GIPFConfig


def compute_jsd(p, q):
    """Compute Jensen-Shannon Divergence"""
    p = p.flatten() + 1e-10
    q = q.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def evaluate_result(result, constraints, user_patterns):
    """Evaluate a result and return metrics"""
    # Spatial stats
    generated_spatial = result.compute_spatial_stats(
        user_patterns, constraints.grid_h, constraints.grid_w
    )
    generated_spatial.normalize()

    jsd_H = compute_jsd(generated_spatial.H, constraints.spatial.H)
    jsd_W = compute_jsd(generated_spatial.W, constraints.spatial.W)
    jsd_O = compute_jsd(generated_spatial.O, constraints.spatial.O)
    spatial_mean = (jsd_H + jsd_W + jsd_O) / 3

    # Interaction stats (on support set)
    generated_interaction = result.compute_interaction_stats(
        user_patterns, constraints.grid_h, constraints.grid_w, top_k=50
    )
    generated_interaction.normalize()

    def support_jsd(gen, real):
        real_coo = real.tocoo()
        if real_coo.nnz == 0:
            return 0.0
        real_vals = real_coo.data + 1e-10
        gen_vals = np.array(gen[real_coo.row, real_coo.col]).flatten() + 1e-10
        real_vals = real_vals / real_vals.sum()
        gen_vals = gen_vals / gen_vals.sum()
        m = 0.5 * (real_vals + gen_vals)
        return float(0.5 * (np.sum(real_vals * np.log(real_vals / m)) +
                           np.sum(gen_vals * np.log(gen_vals / m))))

    hw_jsd = support_jsd(generated_interaction.HW, constraints.interaction.HW)
    ho_jsd = support_jsd(generated_interaction.HO, constraints.interaction.HO)
    wo_jsd = support_jsd(generated_interaction.WO, constraints.interaction.WO)
    interact_mean = (hw_jsd + ho_jsd + wo_jsd) / 3

    return {
        'spatial_H': jsd_H,
        'spatial_W': jsd_W,
        'spatial_O': jsd_O,
        'spatial_mean': spatial_mean,
        'interact_HW': hw_jsd,
        'interact_HO': ho_jsd,
        'interact_WO': wo_jsd,
        'interact_mean': interact_mean,
    }


def run_config(name, config, constraints, user_patterns):
    """Run a single configuration"""
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  alpha_damp={config.alpha_damping}, beta_damp={config.beta_damping}")
    print(f"  freeze_alpha={config.freeze_alpha_in_phase2}, use_beta_final={config.use_beta_in_final}")
    print(f"  temp={config.temp_init}, mfvi_iter={config.mfvi_iter}")
    print('='*60)

    start = time.time()
    method = SSDMFOGIPF(config)
    result = method.run(constraints, user_patterns)
    elapsed = time.time() - start

    metrics = evaluate_result(result, constraints, user_patterns)
    metrics['time'] = elapsed
    metrics['config'] = {
        'alpha_damping': config.alpha_damping,
        'beta_damping': config.beta_damping,
        'freeze_alpha_in_phase2': config.freeze_alpha_in_phase2,
        'use_beta_in_final': config.use_beta_in_final,
        'temp_init': config.temp_init,
        'mfvi_iter': config.mfvi_iter,
    }

    print(f"  Result: Spatial={metrics['spatial_mean']:.4f}, Interact={metrics['interact_mean']:.4f}, Time={elapsed:.1f}s")

    return metrics


def main():
    # Load data
    print("\n[Loading data...]")
    loader = ConstraintDataLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
    )
    constraints = loader.load_all_constraints(phase=2)

    n_users = 100
    print(f"\n[Loading {n_users} users...]")
    user_patterns = loader.load_user_patterns(n_users=n_users)
    print(f"Loaded {len(user_patterns)} users")

    # Run IPF baseline first
    print("\n" + "="*60)
    print("BASELINE: IPF")
    print("="*60)
    start = time.time()
    ipf = IterativeProportionalFitting(max_iter=20)
    ipf_result = ipf.run(constraints, user_patterns)
    ipf_time = time.time() - start
    ipf_metrics = evaluate_result(ipf_result, constraints, user_patterns)
    ipf_metrics['time'] = ipf_time
    ipf_metrics['config'] = {'method': 'IPF'}
    print(f"  IPF: Spatial={ipf_metrics['spatial_mean']:.4f}, Interact={ipf_metrics['interact_mean']:.4f}, Time={ipf_time:.1f}s")

    # Define configurations to test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_config = {
        'max_iter': 100,
        'gpu_batch_size': 100,
        'sddmm_batch_size': 100,
        'gumbel_scale': 0.05,
        'gumbel_decay': 0.99,
        'gumbel_final': 0.01,
        'spatial_first_iters': 40,
        'interaction_freq': 2,
        'gauss_seidel': False,
        'log_freq': 20,
        'early_stop_patience': 30,
        'device': device,
    }

    configs = {
        # Vary damping
        'A1_low_damp': {**base_config, 'alpha_damping': 0.05, 'beta_damping': 0.01,
                        'freeze_alpha_in_phase2': True, 'use_beta_in_final': True,
                        'temp_init': 1.0, 'temp_final': 0.3, 'mfvi_iter': 3},

        'A2_mid_damp': {**base_config, 'alpha_damping': 0.1, 'beta_damping': 0.02,
                        'freeze_alpha_in_phase2': True, 'use_beta_in_final': True,
                        'temp_init': 1.0, 'temp_final': 0.3, 'mfvi_iter': 3},

        'A3_high_damp': {**base_config, 'alpha_damping': 0.2, 'beta_damping': 0.05,
                         'freeze_alpha_in_phase2': True, 'use_beta_in_final': True,
                         'temp_init': 1.0, 'temp_final': 0.3, 'mfvi_iter': 3},

        # Vary freeze_alpha
        'B1_no_freeze': {**base_config, 'alpha_damping': 0.05, 'beta_damping': 0.02,
                         'freeze_alpha_in_phase2': False, 'use_beta_in_final': True,
                         'temp_init': 1.0, 'temp_final': 0.3, 'mfvi_iter': 3},

        # Vary use_beta_in_final
        'C1_no_beta_final': {**base_config, 'alpha_damping': 0.1, 'beta_damping': 0.02,
                              'freeze_alpha_in_phase2': True, 'use_beta_in_final': False,
                              'temp_init': 1.0, 'temp_final': 0.3, 'mfvi_iter': 3},

        # Vary temperature
        'D1_high_temp': {**base_config, 'alpha_damping': 0.1, 'beta_damping': 0.02,
                         'freeze_alpha_in_phase2': True, 'use_beta_in_final': True,
                         'temp_init': 2.0, 'temp_final': 0.5, 'mfvi_iter': 3},

        # Vary MFVI iterations
        'E1_more_mfvi': {**base_config, 'alpha_damping': 0.1, 'beta_damping': 0.02,
                         'freeze_alpha_in_phase2': True, 'use_beta_in_final': True,
                         'temp_init': 1.0, 'temp_final': 0.3, 'mfvi_iter': 5},

        # Combined best guess
        'F1_balanced': {**base_config, 'alpha_damping': 0.08, 'beta_damping': 0.03,
                        'freeze_alpha_in_phase2': False, 'use_beta_in_final': True,
                        'temp_init': 1.5, 'temp_final': 0.4, 'mfvi_iter': 3},
    }

    # Run all configurations
    results = {'IPF': ipf_metrics}

    for name, cfg in configs.items():
        config = GIPFConfig(**cfg)
        metrics = run_config(name, config, constraints, user_patterns)
        results[name] = metrics

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'gipf_sweep_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved to {output_file}]")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Spatial':<10} {'Interact':<10} {'Time':<8} {'Key Params'}")
    print("-"*80)

    # Sort by spatial + interact
    sorted_results = sorted(results.items(),
                           key=lambda x: x[1]['spatial_mean'] + x[1]['interact_mean'])

    for name, m in sorted_results:
        cfg = m.get('config', {})
        if 'method' in cfg:
            params = 'baseline'
        else:
            params = f"α={cfg.get('alpha_damping','?')}, β={cfg.get('beta_damping','?')}, freeze={cfg.get('freeze_alpha_in_phase2','?')}"
        print(f"{name:<20} {m['spatial_mean']:<10.4f} {m['interact_mean']:<10.4f} {m['time']:<8.1f} {params}")

    # Best config
    best = sorted_results[0]
    print(f"\nBest: {best[0]} (Spatial={best[1]['spatial_mean']:.4f}, Interact={best[1]['interact_mean']:.4f})")


if __name__ == '__main__':
    main()
