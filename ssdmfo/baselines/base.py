"""基线方法基类"""
from abc import ABC, abstractmethod
from typing import Dict
import time
import psutil
import os

from ..data.structures import Constraints, UserPattern, Result


class BaseMethod(ABC):
    """所有方法的基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _generate_allocations(self,
                             constraints: Constraints,
                             user_patterns: Dict[int, UserPattern]) -> Dict[int, any]:
        """生成分配（子类实现）"""
        pass

    def run(self, constraints: Constraints,
           user_patterns: Dict[int, UserPattern]) -> Result:
        """运行方法并返回结果"""
        print(f"\n=== Running {self.name} ===")
        print(f"Users: {len(user_patterns)}")
        print(f"Grid size: {constraints.grid_h} x {constraints.grid_w}")

        # 记录开始时间和内存
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # 生成分配
        allocations = self._generate_allocations(constraints, user_patterns)

        # 记录结束时间和内存
        runtime = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_peak = mem_after - mem_before

        print(f"Runtime: {runtime:.2f}s")
        print(f"Memory: {memory_peak:.2f}MB")

        return Result(
            method_name=self.name,
            allocations=allocations,
            runtime=runtime,
            memory_peak=memory_peak
        )
