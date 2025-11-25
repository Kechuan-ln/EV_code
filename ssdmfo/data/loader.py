"""数据加载器"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import zipfile
import io

from .structures import (
    SpatialConstraints, UserPattern, SemanticLocation, Constraints,
    InteractionConstraints
)
from scipy import sparse


class ConstraintDataLoader:
    """加载和管理城市约束数据"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache = {}

    def load_spatial_constraints(self) -> SpatialConstraints:
        """加载一阶空间分布约束 μ_c^real"""
        print("Loading spatial constraints from HWO_distribute.csv...")

        df = pd.read_csv(self.data_dir / 'data/anchor_points/HWO_distribute.csv')

        # 获取栅格范围
        lon_min, lon_max = df['loncol'].min(), df['loncol'].max()
        lat_min, lat_max = df['latcol'].min(), df['latcol'].max()

        print(f"Grid range: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

        # 创建索引映射（从栅格坐标到数组索引）
        lon_to_idx = {lon: i for i, lon in enumerate(range(lon_min, lon_max + 1))}
        lat_to_idx = {lat: i for i, lat in enumerate(range(lat_min, lat_max + 1))}

        grid_h = len(lat_to_idx)
        grid_w = len(lon_to_idx)

        print(f"Grid size: {grid_h} x {grid_w}")

        # 初始化
        H_map = np.zeros((grid_h, grid_w))
        W_map = np.zeros((grid_h, grid_w))
        O_map = np.zeros((grid_h, grid_w))

        # 填充数据
        for _, row in df.iterrows():
            i = lat_to_idx[row['latcol']]
            j = lon_to_idx[row['loncol']]
            count = row['count']

            if row['type'] == 'H':
                H_map[i, j] += count
            elif row['type'] == 'W':
                W_map[i, j] += count
            elif row['type'] == 'O':
                O_map[i, j] += count

        print(f"H total: {H_map.sum()}, W total: {W_map.sum()}, O total: {O_map.sum()}")

        constraints = SpatialConstraints(H=H_map, W=W_map, O=O_map)
        constraints.normalize()

        # 缓存映射
        self._cache['lon_to_idx'] = lon_to_idx
        self._cache['lat_to_idx'] = lat_to_idx

        return constraints

    def load_user_patterns(self, n_users: Optional[int] = None,
                          sample_ratio: float = 0.1) -> Dict[int, UserPattern]:
        """加载用户生活模式"""
        print(f"Loading user patterns (sample_ratio={sample_ratio})...")

        activity_file = self.data_dir / 'data/life_pattern/sh_2311_lifepattern_activity.csv.zip'

        # 读取压缩文件
        with zipfile.ZipFile(activity_file) as z:
            with z.open('sh_2311_lifepattern_activity.csv') as f:
                df = pd.read_csv(f)

        # 获取唯一用户
        all_users = df['reindex'].unique()
        print(f"Total users in dataset: {len(all_users)}")

        # 采样用户
        if n_users is not None:
            n_sample = min(n_users, len(all_users))
        else:
            n_sample = int(len(all_users) * sample_ratio)

        sampled_users = np.random.choice(all_users, n_sample, replace=False)
        print(f"Sampled {n_sample} users")

        # 构建用户模式
        user_patterns = {}
        for user_id in sampled_users:
            user_df = df[df['reindex'] == user_id]

            # 提取语义地点
            locations = self._extract_locations(user_df)

            if len(locations) == 0:
                continue

            user_patterns[user_id] = UserPattern(
                user_id=user_id,
                locations=locations
            )

        print(f"Loaded {len(user_patterns)} valid user patterns")
        return user_patterns

    def _extract_locations(self, user_df: pd.DataFrame) -> List[SemanticLocation]:
        """从用户数据中提取语义地点"""
        unique_types = user_df['type'].unique()
        locations = []

        for type_str in unique_types:
            if pd.isna(type_str) or type_str == '':
                continue

            # 解析 "H_0", "W_1" 等
            parts = type_str.split('_')
            if len(parts) != 2:
                continue

            loc_type, loc_index = parts[0], int(parts[1])
            if loc_type in ['H', 'W', 'O']:
                locations.append(SemanticLocation(type=loc_type, index=loc_index))

        return sorted(locations, key=lambda x: (x.type, x.index))

    def load_interaction_constraints(self) -> InteractionConstraints:
        """加载二阶交互约束 π_cc'^real"""
        print("Loading interaction constraints...")

        # 获取栅格映射（需要先加载空间约束）
        if 'lon_to_idx' not in self._cache:
            self.load_spatial_constraints()

        lon_to_idx = self._cache['lon_to_idx']
        lat_to_idx = self._cache['lat_to_idx']
        grid_h = len(lat_to_idx)
        grid_w = len(lon_to_idx)
        grid_size = grid_h * grid_w

        def load_interaction_matrix(filename: str, col1_prefix: str, col2_prefix: str):
            """加载单个交互矩阵"""
            filepath = self.data_dir / f'data/anchor_points/{filename}'
            df = pd.read_csv(filepath)

            rows, cols, data = [], [], []

            for _, row in df.iterrows():
                # 第一个地点的坐标
                lon1 = row[f'{col1_prefix}loncol']
                lat1 = row[f'{col1_prefix}latcol']
                # 第二个地点的坐标
                lon2 = row[f'{col2_prefix}loncol']
                lat2 = row[f'{col2_prefix}latcol']
                count = row['count']

                # 检查是否在栅格范围内
                if (lon1 in lon_to_idx and lat1 in lat_to_idx and
                    lon2 in lon_to_idx and lat2 in lat_to_idx):

                    # 转换为线性索引
                    idx1 = lat_to_idx[lat1] * grid_w + lon_to_idx[lon1]
                    idx2 = lat_to_idx[lat2] * grid_w + lon_to_idx[lon2]

                    rows.append(idx1)
                    cols.append(idx2)
                    data.append(count)

            matrix = sparse.csr_matrix(
                (data, (rows, cols)), shape=(grid_size, grid_size)
            )
            return matrix

        # 加载三个交互矩阵
        HW = load_interaction_matrix('HW_interact.csv', 'h', 'w')
        print(f"  HW: {HW.nnz} non-zero entries, total={HW.sum():.0f}")

        HO = load_interaction_matrix('HO_interact.csv', 'h', 'o')
        print(f"  HO: {HO.nnz} non-zero entries, total={HO.sum():.0f}")

        WO = load_interaction_matrix('WO_interact.csv', 'w', 'o')
        print(f"  WO: {WO.nnz} non-zero entries, total={WO.sum():.0f}")

        return InteractionConstraints(HW=HW, HO=HO, WO=WO)

    def load_all_constraints(self, phase: int = 1) -> Constraints:
        """加载所有约束（根据phase）"""
        spatial = self.load_spatial_constraints()

        if phase == 1:
            # Phase 1: 仅空间约束
            return Constraints(spatial=spatial)
        elif phase == 2:
            # Phase 2: 空间约束 + 交互约束
            interaction = self.load_interaction_constraints()
            return Constraints(spatial=spatial, interaction=interaction)
        else:
            raise NotImplementedError(f"Phase {phase} not implemented yet")


def test_loader():
    """测试数据加载器"""
    loader = ConstraintDataLoader('/Volumes/FastACIS/Project/EVproject/EV_Splatting')

    # 测试空间约束加载
    print("\n=== Testing Spatial Constraints ===")
    constraints = loader.load_spatial_constraints()
    print(f"Grid shape: {constraints.shape}")
    print(f"H sum: {constraints.H.sum():.6f}")
    print(f"W sum: {constraints.W.sum():.6f}")
    print(f"O sum: {constraints.O.sum():.6f}")

    # 测试用户模式加载（100个用户）
    print("\n=== Testing User Patterns ===")
    users = loader.load_user_patterns(n_users=100)
    print(f"Loaded {len(users)} users")

    # 打印前3个用户
    for i, (user_id, pattern) in enumerate(list(users.items())[:3]):
        print(f"\nUser {user_id}:")
        print(f"  Locations: {pattern.locations}")
        print(f"  N_locations: {len(pattern.locations)}")


if __name__ == '__main__':
    test_loader()
