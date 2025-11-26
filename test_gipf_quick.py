#!/usr/bin/env python3
"""Quick test for SS-DMFO 4.0 G-IPF Optimizer"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.core.optimizer_gipf import SSDMFOGIPF, GIPFConfig
from ssdmfo.evaluation.metrics import MetricsCalculator

# Load data
print("\n[Loading data...]")
loader = ConstraintDataLoader(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting')
)
constraints = loader.load_all_constraints(phase=2)

print(f"\nGrid: {constraints.grid_h} x {constraints.grid_w}")
print(f"Support Set:")
print(f"  HW: {constraints.interaction.HW.nnz:,}")
print(f"  HO: {constraints.interaction.HO.nnz:,}")
print(f"  WO: {constraints.interaction.WO.nnz:,}")

# Load users
print("\n[Loading 100 users...]")
user_patterns = loader.load_user_patterns(n_users=100)
print(f"Loaded {len(user_patterns)} users")

# Test G-IPF optimizer
print("\n[Testing G-IPF SS-DMFO 4.0...]")
config = GIPFConfig(
    max_iter=150,              # More iterations for G-IPF
    gpu_batch_size=100,
    sddmm_batch_size=100,
    # G-IPF parameters - IMPORTANT: use small damping for stability
    alpha_damping=0.1,         # Conservative! Large updates can destabilize
    beta_damping=0.05,         # Very conservative for interaction
    # Temperature - lower is more stable
    temp_anneal=True,
    temp_init=1.0,             # Lower temp = smaller updates
    temp_final=0.3,
    # MFVI
    mfvi_iter=3,               # Fewer MFVI iters in pure spatial phase
    mfvi_damping=0.5,
    # Gumbel - lower noise for G-IPF
    gumbel_scale=0.05,
    gumbel_decay=0.99,
    gumbel_final=0.01,
    # Schedule
    spatial_first_iters=50,    # Longer pure spatial phase
    interaction_freq=2,        # Less frequent interaction updates
    gauss_seidel=False,        # Simpler for now
    # Logging
    log_freq=10,
    early_stop_patience=30,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

start = time.time()
gipf_method = SSDMFOGIPF(config)
result = gipf_method.run(constraints, user_patterns)
elapsed = time.time() - start

# Compute spatial stats
print("\n[Computing spatial statistics...]")
generated_spatial = result.compute_spatial_stats(
    user_patterns, constraints.grid_h, constraints.grid_w
)
generated_spatial.normalize()

# Compute spatial JSD
def compute_jsd(p, q):
    p = p.flatten() + 1e-10
    q = q.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

jsd_H = compute_jsd(generated_spatial.H, constraints.spatial.H)
jsd_W = compute_jsd(generated_spatial.W, constraints.spatial.W)
jsd_O = compute_jsd(generated_spatial.O, constraints.spatial.O)

print(f"\n[Results]")
print(f"  Elapsed time: {elapsed:.1f}s")
print(f"  Spatial JSD:")
print(f"    H: {jsd_H:.4f}")
print(f"    W: {jsd_W:.4f}")
print(f"    O: {jsd_O:.4f}")
print(f"    Mean: {(jsd_H + jsd_W + jsd_O) / 3:.4f}")

# Check allocation shapes
print(f"\n  Sample allocations:")
for uid, alloc in list(result.allocations.items())[:3]:
    print(f"    User {uid}: shape={alloc.shape}, sum={alloc.sum():.2f}")

mean_jsd = (jsd_H + jsd_W + jsd_O) / 3
if mean_jsd < 0.05:
    print("\n[SUCCESS] Excellent spatial JSD!")
elif mean_jsd < 0.1:
    print("\n[GOOD] Spatial JSD is acceptable")
else:
    print("\n[WARNING] Spatial JSD is still high")

# Memory usage
if torch.cuda.is_available():
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
