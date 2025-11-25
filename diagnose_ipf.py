#!/usr/bin/env python3
"""诊断IPF的分配是否有意义"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ssdmfo.data.loader import ConstraintDataLoader
from ssdmfo.baselines.ipf import IterativeProportionalFitting


def main():
    print("Loading data...")
    loader = ConstraintDataLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EV_Splatting'))
    constraints = loader.load_all_constraints(phase=1)
    users = loader.load_user_patterns(n_users=10)  # 只看10个用户

    print("\nRunning IPF...")
    ipf = IterativeProportionalFitting(max_iter=20)
    result = ipf.run(constraints, users)

    print("\n" + "=" * 70)
    print("DIAGNOSING IPF ALLOCATIONS")
    print("=" * 70)

    # 检查单个用户的分配
    for user_id, pattern in list(users.items())[:3]:
        alloc = result.allocations[user_id]
        print(f"\n--- User {user_id} ---")
        print(f"Locations: {pattern.locations}")
        print(f"Allocation shape: {alloc.shape}")

        # 检查每个地点的分配
        for loc_idx, loc in enumerate(pattern.locations):
            probs = alloc[loc_idx]

            # 计算分配的统计特性
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(probs))  # 均匀分布的熵
            normalized_entropy = entropy / max_entropy

            # 找到概率最大的位置
            top_5_idx = np.argsort(probs)[-5:][::-1]
            top_5_probs = probs[top_5_idx]

            print(f"\n  {loc}:")
            print(f"    Entropy: {entropy:.2f} / {max_entropy:.2f} = {normalized_entropy:.4f}")
            print(f"    Max prob: {probs.max():.6f}")
            print(f"    Top 5 probs: {top_5_probs}")
            print(f"    Sum of top 5: {top_5_probs.sum():.4f}")

    # 检查不同用户的同类型地点分配是否相同
    print("\n\n" + "=" * 70)
    print("CHECKING IF ALL H ALLOCATIONS ARE IDENTICAL")
    print("=" * 70)

    h_allocations = []
    for user_id, pattern in users.items():
        alloc = result.allocations[user_id]
        for loc_idx, loc in enumerate(pattern.locations):
            if loc.type == 'H':
                h_allocations.append(alloc[loc_idx])

    if len(h_allocations) >= 2:
        # 比较前两个H分配
        diff = np.abs(h_allocations[0] - h_allocations[1]).max()
        print(f"Max difference between first two H allocations: {diff:.10f}")

        if diff < 1e-6:
            print("WARNING: All H allocations are essentially IDENTICAL!")
            print("This means IPF is NOT assigning different locations to different semantic points.")
            print("It's just copying the target distribution to every allocation.")

    # 对比target分布
    print("\n\n" + "=" * 70)
    print("COMPARING WITH TARGET DISTRIBUTION")
    print("=" * 70)

    target_H = constraints.spatial.H.flatten()
    target_H = target_H / target_H.sum()

    if len(h_allocations) > 0:
        diff_to_target = np.abs(h_allocations[0] - target_H).max()
        corr_to_target = np.corrcoef(h_allocations[0], target_H)[0, 1]
        print(f"First H allocation vs target_H:")
        print(f"  Max diff: {diff_to_target:.10f}")
        print(f"  Correlation: {corr_to_target:.6f}")

        if corr_to_target > 0.999:
            print("\nCONCLUSION: IPF is 'cheating' - each allocation is just a copy of the target!")


if __name__ == '__main__':
    main()
