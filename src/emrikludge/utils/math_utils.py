# src/emrikludge/utils/math_utils.py
"""
常用数学辅助函数。

与其他模块的关系：
- orbits.evolution_pn_fluxes: 可以使用 safe_division 等函数；
- waveforms.aak_waveform: 计算傅里叶变换或数值导数时可使用这里的工具。
"""

from __future__ import annotations

import numpy as np


def safe_divide(a, b, eps=1e-30):
    """安全除法，避免除零引发的 NaN。"""
    return a / (b + eps)


def sinc(x):
    """归一化 sinc 函数 sin(x)/x；x=0 时返回 1。"""
    x = np.asarray(x)
    out = np.ones_like(x, dtype=float)
    mask = (x != 0.0)
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def finite_difference_first_derivative(fvals, dt):
    """
    简单的一阶有限差分导数：
        f'(t_i) ≈ (f_{i+1} - f_{i-1}) / (2 dt)
    边界使用一阶差分。
    """
    fvals = np.asarray(fvals)
    df = np.zeros_like(fvals)
    df[1:-1] = (fvals[2:] - fvals[:-2]) / (2.0 * dt)
    df[0] = (fvals[1] - fvals[0]) / dt
    df[-1] = (fvals[-1] - fvals[-2]) / dt
    return df
