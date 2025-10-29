from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Tuple


@dataclass
class RollingStats:
    window: int
    values: Deque[float]
    sum: float
    sumsq: float

    @classmethod
    def from_seed(cls, values: Iterable[float], window: int) -> "RollingStats":
        dq = deque(maxlen=window)
        s = 0.0
        s2 = 0.0
        count = 0
        for v in values:
            dq.append(float(v))
            s += float(v)
            s2 += float(v) * float(v)
            count += 1
        return cls(window=window, values=dq, sum=s, sumsq=s2)

    def update(self, value: float) -> Tuple[float, float, float]:
        v = float(value)
        if len(self.values) == self.window:
            oldest = self.values[0]
            self.sum -= oldest
            self.sumsq -= oldest * oldest
        self.values.append(v)
        self.sum += v
        self.sumsq += v * v
        mu = self.mean
        sigma = self.std
        z = 0.0 if sigma == 0.0 else (v - mu) / sigma
        return mu, sigma, z

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    @property
    def std(self) -> float:
        n = len(self.values)
        if n < 2:
            return 0.0
        mean_sq = (self.sum * self.sum) / n
        variance = max((self.sumsq - mean_sq) / (n - 1), 0.0)
        return variance ** 0.5


def compute_z_series(closes: Iterable[float], window: int) -> List[float]:
    stats = RollingStats.from_seed([], window)
    out: List[float] = []
    for price in closes:
        mu, sigma, z = stats.update(price)
        out.append(z)
    return out


__all__ = ["RollingStats", "compute_z_series"]

