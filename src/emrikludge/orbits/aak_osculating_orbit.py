# src/emrikludge/orbits/aak_osculating_orbit.py
"""
AAK 轨道（osculating AK）演化模块（Python 版）。

功能：
- 给定 EMRIParameters + WaveformConfig（主要用 M, mu, a, p0, e0, iota0），
  使用 PN 平均 flux (目前为 Peters-Mathews) 积分得到随时间变化的
  (p(t), e(t), iota(t)) 以及轨道/进动相位 (Phi, gamma, alpha)；
- 输出轨道时间序列，用于后续波形计算。

与其他模块的关系：
- evolution_pn_fluxes.peters_mathews_fluxes: 提供 (dp/dt, de/dt)；
- waveforms.aak_waveform: 读取本模块给出的轨道，计算瞬时频率和振幅；
- core.aak_cpu: 顶层调用此模块，获得轨道后再进入波形模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import EMRIParameters, WaveformConfig
from .evolution_pn_fluxes import peters_mathews_fluxes
from ..constants import G_SI, C_SI, M_SUN_SI

import numpy as np
from scipy.integrate import solve_ivp
@dataclass
class AAKOrbitTrajectory:
    """
    AAK 轨道时间序列。

    所有量都是在“AK/AAK 坐标系”下的轨道参数，而不是 Kerr 本征常数。

    Attributes
    ----------
    t : ndarray
        物理时间 [s]
    p : ndarray
        半通径 p/M（无量纲）
    e : ndarray
        偏心率
    iota : ndarray
        轨道倾角（与 BH 自旋轴的夹角）[rad]
    Phi : ndarray
        轨道相位（类似于离心近点角）[rad]
    gamma : ndarray
        近日点进动相位 [rad]
    alpha : ndarray
        轨道平面进动相位（Lense–Thirring）[rad]
    """
    t: np.ndarray
    p: np.ndarray
    e: np.ndarray
    iota: np.ndarray
    Phi: np.ndarray
    gamma: np.ndarray
    alpha: np.ndarray


def _orbital_frequency_ak(M_solar: float,
                          p_dimless: float,
                          e: float) -> float:
    r"""
    AK 模型中的平均轨道频率 ν (Hz)，主阶近似。

    Barack & Cutler 的做法是从半通径 p 和偏心率 e 出发：
        a = p / (1 - e^2)
        ν_geom = (M / a^3)^{1/2} / (2π)      （几何单位）
        ν_SI   = ν_geom / (GM/c^3)

    这里先保留牛顿主阶，把 PN 修正留给后续 TODO。
    """
    # 几何时间单位：1 M 对应 GM/c^3 秒
    M_geom_time = G_SI * (M_solar * M_SUN_SI) / C_SI**3  # [s]
    a_dimless = p_dimless / (1.0 - e**2)

    # 几何单位的频率（以 1/M 为单位）
    nu_geom = np.sqrt(1.0 / (a_dimless**3)) / (2.0 * np.pi)
    # 转回 SI：除以 M_geom_time
    nu_SI = nu_geom / M_geom_time
    return nu_SI

def mu_small(M_solar: float) -> float:
    """
    在 EMRI 极端质量比极限下，mu << M。
    这里只是占位函数，实际演化中我们直接用 EMRIParameters.mu。
    """
    # 这里只是为了 _orbital_frequency_kepler 在没拿到 mu 时不崩溃：
    return 10.0  # M_sun，占位；真正调用时你应该直接传 mu_solar


def _phase_derivatives(M_solar: float,
                       mu_solar: float,
                       a_spin: float,
                       p_dimless: float,
                       e: float,
                       iota: float) -> Tuple[float, float, float]:
    r"""
    给出 (Phi, gamma, alpha) 的时间导数。

    结构上仿照 Barack & Cutler 2004 Sec.III B–D：
        dPhi/dt   = 2π ν
        dgamma/dt = \dot{\tilde\gamma}        （近日点进动）
        dalpha/dt = \dot{\alpha}              （轨道平面进动）

    当前实现：
    - ν 使用 _orbital_frequency_ak 的牛顿主阶；
    - \dot{\tilde\gamma} 使用 Schwarzschild 1PN 主阶：~ 3 M/p * dPhi/dt；
    - \dot{\alpha} 使用 Lense–Thirring 1.5PN 主阶：~ 2 a (M/p)^3 cosι * 2πν。

    TODO（未来可以继续完成的部分）：
    - 把 ν, \dot{\tilde\gamma}, \dot{\alpha} 替换成 Barack & Cutler
      Eqs.(43–54) 给出的完整 PN 展开（最多到 3.5PN）；
    - AAK 情况下，用 Chua et al. 2017 的 Kerr geodesic 频率
      Ω_r, Ω_θ, Ω_φ 做频率映射。
    """
    # 1. 轨道平均频率 ν
    nu = _orbital_frequency_ak(M_solar, p_dimless, e)  # [Hz]
    dPhi_dt = 2.0 * np.pi * nu                         # [rad/s]

    # 2. 近日点进动：Schwarzschild 1PN 主阶 ~ 3 (M/p) dPhi/dt
    #    其中 p_dimless = p/M  =>  M/p = 1/p_dimless
    pericenter_factor = 3.0 / p_dimless
    dgamma_dt = pericenter_factor * dPhi_dt

    # 3. Lense–Thirring 进动：1.5PN 主阶 ~ 2 a (M/p)^3 cosι * 2πν
    dalpha_dt = 2.0 * a_spin * (1.0 / p_dimless)**3 * np.cos(iota) * 2.0 * np.pi * nu

    return dPhi_dt, dgamma_dt, dalpha_dt

def evolve_aak_orbit(params: EMRIParameters,
                     config: WaveformConfig) -> AAKOrbitTrajectory:
    """
    使用 PN 平均 flux + 简化相位演化，积分 AAK 轨道。

    输入：
    - params: EMRIParameters（M, mu, a, p0, e0, iota0, 相位等）；
    - config: WaveformConfig（总时长 T，步长 dt 等）。

    输出：
    - AAKOrbitTrajectory: 轨道时间序列。
    """
    M_solar = params.M
    mu_solar = params.mu
    a_spin = params.a

    # 初始状态向量 y = [p, e, iota, Phi, gamma, alpha]
    y0 = np.array([
        params.p0,
        params.e0,
        params.iota0,
        params.Phi0,
        params.gamma0,
        params.alpha0,
    ], dtype=float)

    t0 = 0.0
    t1 = config.T

    # 我们会在外面按固定 dt 做插值，所以这里用自适应步长更方便
    def rhs(t, y):
        p, e, iota, Phi, gamma, alpha = y

        # Orbital parameter evolution:
        dp_dt, de_dt = peters_mathews_fluxes(
            p_dimless=p,
            e=e,
            M_solar=M_solar,
            mu_solar=mu_solar,
        )

        # 目前不考虑 iota 的演化（AAK 中可由 flux 得到 diota/dt），
        # 这里先设为 0，并在注释提示你后续替换：
        diota_dt = 0.0

        # Phase evolution:
        dPhi_dt, dgamma_dt, dalpha_dt = _phase_derivatives(
            M_solar=M_solar,
            mu_solar=mu_solar,
            a_spin=a_spin,
            p_dimless=p,
            e=e,
            iota=iota,
        )

        return np.array([dp_dt, de_dt, diota_dt,
                         dPhi_dt, dgamma_dt, dalpha_dt], dtype=float)

    kwargs = dict(
        method="RK45",
        rtol=config.rtol,
        atol=config.atol,
        dense_output=True,
    )
    # 只有在用户真的指定了 max_step 时才传给 solve_ivp
    if config.max_step is not None:
        kwargs["max_step"] = config.max_step

    sol = solve_ivp(
        rhs,
        t_span=(t0, t1),
        y0=y0,
        **kwargs,
    )
    # 在等间隔时间栅格上采样，以匹配波形采样率
    t_grid = np.arange(0.0, config.T, config.dt)
    y_grid = sol.sol(t_grid)
    p, e, iota, Phi, gamma, alpha = y_grid

    return AAKOrbitTrajectory(
        t=t_grid,
        p=p,
        e=e,
        iota=iota,
        Phi=Phi,
        gamma=gamma,
        alpha=alpha,
    )
