# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SS-DMFO (Semantic-Spatial Distribution Matching with Flow Optimization) is a research project for **user-based urban trajectory generation**. The goal is to generate realistic spatial coordinates for users' semantic locations (Home/Work/Other) such that aggregate statistics match real-world city mobility data.

**Core Problem**: Given ~93,000 users with known life patterns (where they spend time semantically), assign physical grid coordinates to each semantic location so that:
1. Spatial distributions of H/W/O match real data
2. Co-location patterns (HW, HO, WO interactions) match real data
3. Hourly OD flows match real data

## Commands

### Running Tests and Comparisons
```bash
# Run end-to-end test with random baseline
python test_phase1.py

# Compare all baseline methods (Random, Gravity, IPF)
python compare_methods.py

# Quick test
python quick_test.py

# Test individual components
python -m ssdmfo.data.loader
python -m ssdmfo.evaluation.metrics
python -m ssdmfo.baselines.gravity
python -m ssdmfo.baselines.ipf
```

### Dependencies
```bash
pip install numpy pandas matplotlib torch psutil
```

## Architecture

### Module Structure
```
ssdmfo/                    # Main implementation package
├── data/
│   ├── structures.py      # Core dataclasses: SpatialConstraints, UserPattern, Result
│   └── loader.py          # ConstraintDataLoader - loads constraints and user patterns
├── baselines/
│   ├── base.py            # BaseMethod ABC - all methods inherit from this
│   ├── random.py          # RandomBaseline
│   ├── gravity.py         # GravityModel(beta) - distance-weighted allocation
│   └── ipf.py             # IterativeProportionalFitting - Sinkhorn-type algorithm
└── evaluation/
    └── metrics.py         # MetricsCalculator - JSD, TVD, KL divergence

EV_Splatting/              # Legacy/reference implementation
├── src/validation.py      # Visualization and validation functions
└── data/                  # All raw data files
```

### Data Pipeline
```
EV_Splatting/data/
├── anchor_points/
│   └── HWO_distribute.csv     → SpatialConstraints (H/W/O probability maps)
├── life_pattern/
│   └── sh_2311_lifepattern_activity.csv.zip  → UserPattern objects
└── od_data/
    └── OD_Hour_{0-23}.csv     → Hourly OD flow matrices (Phase 2+)
```

### Key Interfaces

**BaseMethod** (`ssdmfo/baselines/base.py`):
```python
class BaseMethod(ABC):
    def run(self, constraints: Constraints, user_patterns: Dict[int, UserPattern]) -> Result
    def _generate_allocations(self, ...) -> Dict[int, np.ndarray]  # user_id → (n_locs, grid_size)
```

**Result.compute_spatial_stats()**: Aggregates per-user allocations into city-wide H/W/O distributions

**MetricsCalculator**: Computes JSD (Jensen-Shannon Divergence) and TVD (Total Variation Distance) between generated and real distributions

### Implementation Phases

| Phase | Constraints | Status |
|-------|------------|--------|
| Phase 1 | Spatial distributions (H/W/O) only | Implemented |
| Phase 2 | + Interaction constraints (HW/HO/WO) | Not started |
| Phase 3 | + Hourly OD flows | Not started |

## Data Formats

- **Grid**: 221 rows × 185 cols (~1.25km × 0.83km per cell)
- **User locations**: Up to 18 semantic locations per user (3H + 5W + 10O max)
- **Allocation output**: `Dict[user_id, np.ndarray]` where array is `(n_locations, grid_h * grid_w)` probability distribution

## Evaluation Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| JSD (spatial) | < 0.1 | < 0.2 |
| Runtime | < 10 min | < 30 min |
| Memory | < 4 GB | < 8 GB |

## Key Concepts

- **Life Pattern**: Per-user probability P(location | hour) derived from mobility data
- **Semantic Location**: H_0, W_1, O_3 etc. - abstract locations that need physical coordinates
- **Allocation**: Probability distribution over grid cells for each semantic location
- **IPF (Iterative Proportional Fitting)**: Alternately scales allocations to match marginal constraints

## Notes

- Shanghai mobility data (November 2023, prefix `sh_2311`)
- Cross-day handling: if `shour > ehour`, split activity across midnight
- All distributions should be normalized before computing metrics
- The `EV_Splatting/` directory contains legacy code; new development is in `ssdmfo/`
