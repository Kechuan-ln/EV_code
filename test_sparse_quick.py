#!/usr/bin/env python3
"""Quick test for SS-DMFO 3.0 Sparse Optimizer"""

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
from ssdmfo.core.optimizer_sparse import SSDMFOSparse, SparseConfig

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

# Load only 50 users for quick test
print("\n[Loading 50 users...]")
user_patterns = loader.load_user_patterns(n_users=50)
print(f"Loaded {len(user_patterns)} users")

# Test sparse optimizer
print("\n[Testing Sparse SS-DMFO...]")
config = SparseConfig(
    max_iter=30,  # Short test
    gpu_batch_size=50,
    mfvi_iter=2,
    interaction_freq=3,
    log_freq=5,
    phase1_ratio=0.2,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

start = time.time()
sparse_method = SSDMFOSparse(config)
result = sparse_method.run(constraints, user_patterns)
elapsed = time.time() - start

print(f"\n[Results]")
print(f"  Allocations generated: {len(result.allocations)}")
print(f"  Elapsed time: {elapsed:.1f}s")

# Check allocation shapes
for uid, alloc in list(result.allocations.items())[:3]:
    print(f"  User {uid}: shape={alloc.shape}, sum={alloc.sum():.2f}")

print("\n[SUCCESS] Sparse optimizer works!")

# Memory usage
if torch.cuda.is_available():
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
