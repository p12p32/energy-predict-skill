"""net_load.py — 净负荷计算: 电网需要平衡的核心变量.

Net Load = Total Load − Solar Output − Wind Output

这是现货电价波动的核心驱动:
  - 高净负荷 → 需要火电出力 → 电价上行
  - 低/负净负荷 → 可再生过剩 → 电价受压或负电价
"""
import numpy as np
from typing import Dict, Optional


class NetLoadComputer:
    """一等公民: 净负荷 = 负荷 − 风光出力.

    不建模, 只计算. 显式追踪 气象→风光→负荷→净负荷→电价 因果链.
    """

    def compute(self, load: np.ndarray, solar: np.ndarray,
                wind: np.ndarray) -> np.ndarray:
        """计算净负荷.

        Args:
            load: 负荷预测 (N,)
            solar: 光伏出力预测 (N,)
            wind: 风电出力预测 (N,)

        Returns:
            net_load (N,): 电网需要火电等可控电源提供的功率
        """
        return load - solar - wind

    def decompose(self, net_load: np.ndarray,
                  load: Optional[np.ndarray] = None,
                  solar: Optional[np.ndarray] = None,
                  wind: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """分解净负荷为: 水平、爬坡、波动性分量.

        Returns dict with:
          - base: 慢变分量 (24点移动平均)
          - ramp: 爬坡分量 (1阶差分)
          - volatility: 波动分量 (残差)
          - renewable_share: 可再生渗透率
        """
        n = len(net_load)
        window = min(24, max(4, n // 4))
        kernel = np.ones(window) / window
        base = np.convolve(net_load, kernel, mode="same")
        ramp = np.zeros(n)
        ramp[1:] = np.diff(net_load)
        volatility = net_load - base

        result = {"base": base, "ramp": ramp, "volatility": volatility}

        if load is not None:
            result["load"] = load
            if solar is not None and wind is not None:
                renewable = solar + wind
                eps = 1e-6
                result["renewable_share"] = renewable / (load + eps)
                result["renewable_share"] = np.clip(result["renewable_share"], 0, 1)

        return result

    def detect_regime(self, net_load: np.ndarray,
                      thresholds: Optional[Dict] = None) -> np.ndarray:
        """净负荷区间分类.

        默认阈值 (相对 max):
          0: surplus (< 30%)
          1: normal  (30-70%)
          2: tight   (70-90%)
          3: extreme (> 90%)

        Returns:
            regime (N,): 0-3 整数标签
        """
        if thresholds is None:
            max_val = np.max(net_load) + 1e-6
            thresholds = {
                "surplus": 0.30 * max_val,
                "tight": 0.70 * max_val,
                "extreme": 0.90 * max_val,
            }

        regime = np.full(len(net_load), 1, dtype=int)  # default: normal
        regime[net_load < thresholds["surplus"]] = 0
        regime[net_load >= thresholds["tight"]] = 2
        regime[net_load >= thresholds["extreme"]] = 3
        return regime
