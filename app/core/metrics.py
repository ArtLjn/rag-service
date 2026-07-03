"""Prometheus 风格计数器与直方图（进程内内存版本）。

毕设范围内不做 /metrics 暴露，仅用于日志与未来扩展。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class Counter:
    name: str
    description: str = ""
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class Histogram:
    name: str
    description: str = ""
    observations: list[float] = field(default_factory=list)

    def observe(self, seconds: float) -> None:
        self.observations.append(seconds)

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def sum(self) -> float:
        return sum(self.observations)

    def p95(self) -> float:
        if not self.observations:
            return 0.0
        sorted_obs = sorted(self.observations)
        idx = max(0, int(len(sorted_obs) * 0.95) - 1)
        return sorted_obs[idx]


class MetricsRegistry:
    def __init__(self) -> None:
        self.counters: dict[str, Counter] = {}
        self.histograms: dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter(name=name, description=description)
        return self.counters[name]

    def histogram(self, name: str, description: str = "") -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram(name=name, description=description)
        return self.histograms[name]

    @contextmanager
    def time(self, name: str, description: str = "") -> Iterator[None]:
        hist = self.histogram(name, description)
        start = perf_counter()
        try:
            yield
        finally:
            hist.observe(perf_counter() - start)

    def time_func(self, name: str, description: str = "") -> Callable:
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with self.time(name, description):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def snapshot(self) -> dict:
        return {
            "counters": {n: c.value for n, c in self.counters.items()},
            "histograms": {
                n: {"count": h.count, "sum": h.sum, "p95": h.p95()}
                for n, h in self.histograms.items()
            },
        }


metrics = MetricsRegistry()
