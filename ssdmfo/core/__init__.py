"""SS-DMFO Core Module"""

from .potentials import DualPotentials, PotentialsWithMomentum
from .mean_field import MeanFieldSolver, FastMeanFieldSolver
from .optimizer import SSDMFOOptimizer, SSDMFOPhase2
from .optimizer_v3 import SSDMFOv3, SSDMFO3Config, create_ssdmfo_v3

# GPU version (optional, requires PyTorch)
try:
    from .optimizer_gpu import SSDMFOv3GPU, GPUConfig, create_ssdmfo_gpu
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

__all__ = [
    'DualPotentials',
    'PotentialsWithMomentum',
    'MeanFieldSolver',
    'FastMeanFieldSolver',
    'SSDMFOOptimizer',
    'SSDMFOPhase2',
    # V3 - Fixed version
    'SSDMFOv3',
    'SSDMFO3Config',
    'create_ssdmfo_v3',
    # GPU version
    'SSDMFOv3GPU',
    'GPUConfig',
    'create_ssdmfo_gpu',
    'HAS_GPU',
]
