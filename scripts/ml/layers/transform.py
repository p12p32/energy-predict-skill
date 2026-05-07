"""transform.py — 按数据分布自动选择最优变换."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


@dataclass
class TransformConfig:
    name: str = "identity"
    C: Optional[float] = None
    lmbda: Optional[float] = None

    def __repr__(self):
        if self.name == "log_plus_C":
            return f"log(x+{self.C:.2f})"
        if self.name == "yeo_johnson":
            return f"YeoJohnson(λ={self.lmbda:.2f})"
        return self.name


class TransformSelector:
    """按数据分布自动选择变换: identity | log | log+C | asinh | yeo_johnson."""

    CANDIDATES = ["identity", "log", "asinh", "yeo_johnson"]
    C_RANGE = [0.01, 0.1, 1.0, 10.0]

    def select(self, values: np.ndarray, type_name: str = "") -> TransformConfig:
        v = values[np.isfinite(values)]
        if len(v) < 50:
            return TransformConfig("identity")

        has_zero = (v <= 0).any()
        has_negative = (v < 0).any()

        # 缩减候选
        candidates = list(self.CANDIDATES)
        if has_negative:
            candidates = ["identity", "asinh", "yeo_johnson"]
        if has_zero:
            candidates = [c for c in candidates if c != "log"]

        best_config = TransformConfig("identity")
        best_score = -1.0

        for name in candidates:
            configs = [TransformConfig(name)]
            if name == "log_plus_C":
                q01 = max(np.quantile(v[v > 0], 0.01) if (v > 0).any() else 0.1, 0.01)
                configs = [TransformConfig("log_plus_C", C=c) for c in [*self.C_RANGE, q01]]

            for cfg in configs:
                try:
                    t = self.apply(v, cfg)
                    score = self._score(t)
                    if score > best_score:
                        best_score = score
                        best_config = cfg
                except Exception:
                    continue

        # 风电/光伏偏向 asinh (多零值)
        if any(kw in type_name for kw in ("风电", "光伏")):
            if best_config.name == "identity" and best_score < 0.5:
                return TransformConfig("asinh")

        return best_config

    def _score(self, t: np.ndarray) -> float:
        t = t[np.isfinite(t)]
        if len(t) < 50:
            return 0.0
        skew_before = abs(stats.skew(t))
        skew_after = min(skew_before, 0.5)
        skew_score = 1.0 - (skew_after / max(skew_before, 0.01))
        skew_score = max(0.0, min(1.0, skew_score))

        _, p = stats.normaltest(t) if len(t) <= 1000 else (0, 0)
        norm_score = min(1.0, p / 0.05) if p > 0 else 0.0

        return 0.6 * skew_score + 0.4 * norm_score

    def apply(self, values: np.ndarray, config: TransformConfig) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if config.name == "identity":
            return v
        if config.name == "log":
            return np.log(np.maximum(v, 1e-8))
        if config.name == "log_plus_C":
            return np.log(np.maximum(v + (config.C or 0.01), 1e-8))
        if config.name == "asinh":
            return np.arcsinh(v)
        if config.name == "yeo_johnson":
            lmbda = config.lmbda if config.lmbda is not None else self._fit_yeo_johnson(v)
            config.lmbda = lmbda
            return self._yeo_johnson(v, lmbda)
        return v

    def inverse(self, values: np.ndarray, config: TransformConfig) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        if config.name == "identity":
            return v
        if config.name == "log":
            return np.exp(v)
        if config.name == "log_plus_C":
            return np.exp(v) - (config.C or 0.01)
        if config.name == "asinh":
            return np.sinh(v)
        if config.name == "yeo_johnson":
            lmbda = config.lmbda or 0.0
            return self._inverse_yeo_johnson(v, lmbda)
        return v

    @staticmethod
    def _fit_yeo_johnson(v: np.ndarray) -> float:
        _, lmbda = stats.yeojohnson(v)
        return float(lmbda)

    @staticmethod
    def _yeo_johnson(v: np.ndarray, lmbda: float) -> np.ndarray:
        pos = v >= 0
        t = np.zeros_like(v)
        if abs(lmbda) < 1e-8:
            t[pos] = np.log1p(v[pos])
            t[~pos] = -np.log1p(-v[~pos])
        else:
            t[pos] = ((v[pos] + 1) ** lmbda - 1) / lmbda
            t[~pos] = -((1 - v[~pos]) ** (2 - lmbda) - 1) / (2 - lmbda)
        return t

    @staticmethod
    def _inverse_yeo_johnson(v: np.ndarray, lmbda: float) -> np.ndarray:
        pos = v >= 0
        r = np.zeros_like(v)
        if abs(lmbda) < 1e-8:
            r[pos] = np.expm1(v[pos])
            r[~pos] = -np.expm1(-v[~pos])
        else:
            r[pos] = (v[pos] * lmbda + 1) ** (1 / lmbda) - 1
            r[~pos] = 1 - (1 - v[~pos] * (2 - lmbda)) ** (1 / (2 - lmbda))
        return r
