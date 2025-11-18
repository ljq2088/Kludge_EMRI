# # src/emrikludge/orbits/aak_osculating_orbit.py
# """
# AAK 轨道（osculating AK）演化模块（Python 版）。

# 功能：
# - 给定 EMRIParameters + WaveformConfig（主要用 M, mu, a, p0, e0, iota0），
#   使用 PN 平均 flux (目前为 Peters-Mathews) 积分得到随时间变化的
#   (p(t), e(t), iota(t)) 以及轨道/进动相位 (Phi, gamma, alpha)；
# - 输出轨道时间序列，用于后续波形计算。

# 与其他模块的关系：
# - evolution_pn_fluxes.peters_mathews_fluxes: 提供 (dp/dt, de/dt)；
# - waveforms.aak_waveform: 读取本模块给出的轨道，计算瞬时频率和振幅；
# - core.aak_cpu: 顶层调用此模块，获得轨道后再进入波形模块。
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Tuple

# import numpy as np
# from scipy.integrate import solve_ivp

# from ..parameters import EMRIParameters, WaveformConfig
# from .evolution_pn_fluxes import peters_mathews_fluxes
# from ..constants import G_SI, C_SI, M_SUN_SI

# import numpy as np
# from scipy.integrate import solve_ivp
# @dataclass
# class AAKOrbitTrajectory:
#     """
#     AAK 轨道时间序列。

#     所有量都是在“AK/AAK 坐标系”下的轨道参数，而不是 Kerr 本征常数。

#     Attributes
#     ----------
#     t : ndarray
#         物理时间 [s]
#     p : ndarray
#         半通径 p/M（无量纲）
#     e : ndarray
#         偏心率
#     iota : ndarray
#         轨道倾角（与 BH 自旋轴的夹角）[rad]
#     Phi : ndarray
#         轨道相位（类似于离心近点角）[rad]
#     gamma : ndarray
#         近日点进动相位 [rad]
#     alpha : ndarray
#         轨道平面进动相位（Lense–Thirring）[rad]
#     """
#     t: np.ndarray
#     p: np.ndarray
#     e: np.ndarray
#     iota: np.ndarray
#     Phi: np.ndarray
#     gamma: np.ndarray
#     alpha: np.ndarray


# def _orbital_frequency_ak(M_solar: float,
#                           p_dimless: float,
#                           e: float) -> float:
#     r"""
#     AK 模型中的平均轨道频率 ν (Hz)，主阶近似。

#     Barack & Cutler 的做法是从半通径 p 和偏心率 e 出发：
#         a = p / (1 - e^2)
#         ν_geom = (M / a^3)^{1/2} / (2π)      （几何单位）
#         ν_SI   = ν_geom / (GM/c^3)

#     这里先保留牛顿主阶，把 PN 修正留给后续 TODO。
#     """
#     # 几何时间单位：1 M 对应 GM/c^3 秒
#     M_geom_time = G_SI * (M_solar * M_SUN_SI) / C_SI**3  # [s]
#     a_dimless = p_dimless / (1.0 - e**2)

#     # 几何单位的频率（以 1/M 为单位）
#     nu_geom = np.sqrt(1.0 / (a_dimless**3)) / (2.0 * np.pi)
#     # 转回 SI：除以 M_geom_time
#     nu_SI = nu_geom / M_geom_time
#     return nu_SI

# def mu_small(M_solar: float) -> float:
#     """
#     在 EMRI 极端质量比极限下，mu << M。
#     这里只是占位函数，实际演化中我们直接用 EMRIParameters.mu。
#     """
#     # 这里只是为了 _orbital_frequency_kepler 在没拿到 mu 时不崩溃：
#     return 10.0  # M_sun，占位；真正调用时你应该直接传 mu_solar


# def _phase_derivatives(M_solar: float,
#                        mu_solar: float,
#                        a_spin: float,
#                        p_dimless: float,
#                        e: float,
#                        iota: float) -> Tuple[float, float, float]:
#     r"""
#     给出 (Phi, gamma, alpha) 的时间导数。

#     结构上仿照 Barack & Cutler 2004 Sec.III B–D：
#         dPhi/dt   = 2π ν
#         dgamma/dt = \dot{\tilde\gamma}        （近日点进动）
#         dalpha/dt = \dot{\alpha}              （轨道平面进动）

#     当前实现：
#     - ν 使用 _orbital_frequency_ak 的牛顿主阶；
#     - \dot{\tilde\gamma} 使用 Schwarzschild 1PN 主阶：~ 3 M/p * dPhi/dt；
#     - \dot{\alpha} 使用 Lense–Thirring 1.5PN 主阶：~ 2 a (M/p)^3 cosι * 2πν。

#     TODO（未来可以继续完成的部分）：
#     - 把 ν, \dot{\tilde\gamma}, \dot{\alpha} 替换成 Barack & Cutler
#       Eqs.(43–54) 给出的完整 PN 展开（最多到 3.5PN）；
#     - AAK 情况下，用 Chua et al. 2017 的 Kerr geodesic 频率
#       Ω_r, Ω_θ, Ω_φ 做频率映射。
#     """
#     # 1. 轨道平均频率 ν
#     nu = _orbital_frequency_ak(M_solar, p_dimless, e)  # [Hz]
#     dPhi_dt = 2.0 * np.pi * nu                         # [rad/s]

#     # 2. 近日点进动：Schwarzschild 1PN 主阶 ~ 3 (M/p) dPhi/dt
#     #    其中 p_dimless = p/M  =>  M/p = 1/p_dimless
#     pericenter_factor = 3.0 / p_dimless
#     dgamma_dt = pericenter_factor * dPhi_dt

#     # 3. Lense–Thirring 进动：1.5PN 主阶 ~ 2 a (M/p)^3 cosι * 2πν
#     dalpha_dt = 2.0 * a_spin * (1.0 / p_dimless)**3 * np.cos(iota) * 2.0 * np.pi * nu

#     return dPhi_dt, dgamma_dt, dalpha_dt

# def evolve_aak_orbit(params: EMRIParameters,
#                      config: WaveformConfig) -> AAKOrbitTrajectory:
#     """
#     使用 PN 平均 flux + 简化相位演化，积分 AAK 轨道。

#     输入：
#     - params: EMRIParameters（M, mu, a, p0, e0, iota0, 相位等）；
#     - config: WaveformConfig（总时长 T，步长 dt 等）。

#     输出：
#     - AAKOrbitTrajectory: 轨道时间序列。
#     """
#     M_solar = params.M
#     mu_solar = params.mu
#     a_spin = params.a

#     # 初始状态向量 y = [p, e, iota, Phi, gamma, alpha]
#     y0 = np.array([
#         params.p0,
#         params.e0,
#         params.iota0,
#         params.Phi0,
#         params.gamma0,
#         params.alpha0,
#     ], dtype=float)

#     t0 = 0.0
#     t1 = config.T

#     # 我们会在外面按固定 dt 做插值，所以这里用自适应步长更方便
#     def rhs(t, y):
#         p, e, iota, Phi, gamma, alpha = y

#         # Orbital parameter evolution:
#         dp_dt, de_dt = peters_mathews_fluxes(
#             p_dimless=p,
#             e=e,
#             M_solar=M_solar,
#             mu_solar=mu_solar,
#         )

#         # 目前不考虑 iota 的演化（AAK 中可由 flux 得到 diota/dt），
#         # 这里先设为 0，并在注释提示你后续替换：
#         diota_dt = 0.0

#         # Phase evolution:
#         dPhi_dt, dgamma_dt, dalpha_dt = _phase_derivatives(
#             M_solar=M_solar,
#             mu_solar=mu_solar,
#             a_spin=a_spin,
#             p_dimless=p,
#             e=e,
#             iota=iota,
#         )

#         return np.array([dp_dt, de_dt, diota_dt,
#                          dPhi_dt, dgamma_dt, dalpha_dt], dtype=float)

#     kwargs = dict(
#         method="RK45",
#         rtol=config.rtol,
#         atol=config.atol,
#         dense_output=True,
#     )
#     # 只有在用户真的指定了 max_step 时才传给 solve_ivp
#     if config.max_step is not None:
#         kwargs["max_step"] = config.max_step

#     sol = solve_ivp(
#         rhs,
#         t_span=(t0, t1),
#         y0=y0,
#         **kwargs,
#     )
#     # 在等间隔时间栅格上采样，以匹配波形采样率
#     t_grid = np.arange(0.0, config.T, config.dt)
#     y_grid = sol.sol(t_grid)
#     p, e, iota, Phi, gamma, alpha = y_grid

#     return AAKOrbitTrajectory(
#         t=t_grid,
#         p=p,
#         e=e,
#         iota=iota,
#         Phi=Phi,
#         gamma=gamma,
#         alpha=alpha,
#     )
# src/emrikludge/orbits/aak_osculating_orbit.py
"""
AAK 轨道（osculating AK）演化模块（Python 版）。

本模块实现的是 Barack & Cutler 2004 (BC04) 的轨道平均演化：
- 以轨道平均频率 ν 和偏心率 e 为基本变量；
- 使用 Sec. II F 中的 Eqs. (27)–(31)：
    dΦ/dt, dν/dt, d\tildeγ/dt, de/dt, dα/dt；

并保持与旧界面兼容：
- 对外仍然返回一个 AAKOrbitTrajectory，其中包含：
    t, p(t), e(t), iota(t), Phi(t), gamma(t), alpha(t)
- 其中 p(t) 通过 Newtonian 关系从 (ν, e) 反推出；
- iota(t) 目前假定为常数 = iota0，对应 BC04 中固定的 λ。

与其他模块的关系：
- orbits.evolution_pn_fluxes.bc04_orbital_derivatives 提供 BC04 PN 通量；
- waveforms.aak_waveform 读取本模块给出的 (p,e,iota,Φ,γ,α) 生成多谐波波形；
- core.aak_cpu 是顶层入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import EMRIParameters, WaveformConfig
from ..constants import G_SI, C_SI, M_SUN_SI
from .evolution_pn_fluxes import bc04_orbital_derivatives


@dataclass
class AAKOrbitTrajectory:
    """
    AAK 轨道时间序列。

    Attributes
    ----------
    t : ndarray
        物理时间 [s]
    p : ndarray
        半通径 p/M（无量纲，利用 Newtonian 关系从 ν 反推）
    e : ndarray
        偏心率
    iota : ndarray
        轨道倾角（与 BH 自旋轴的夹角）[rad]，当前实现中为常数
    Phi : ndarray
        轨道相位 Φ [rad]
    gamma : ndarray
        近日点进动相位 γ，这里等同于 BC04 中的 \tildeγ [rad]
    alpha : ndarray
        轨道平面进动相位 α（Lense–Thirring）[rad]
    nu : ndarray | None
        轨道平均频率 ν(t) [Hz]；为方便调试，这里一并输出。
    """
    t: np.ndarray
    p: np.ndarray
    e: np.ndarray
    iota: np.ndarray
    Phi: np.ndarray
    gamma: np.ndarray
    alpha: np.ndarray
    nu: Optional[np.ndarray] = None


# ===========================
# Newtonian 映射： (p, e) <-> ν
# ===========================

def _kepler_nu_from_p(M_solar: float, p_dimless: float, e: float) -> float:
    """
    由 (p, e) 得到 Newtonian 轨道平均频率 ν [Hz]。

    采用：
        a = p M / (1 - e^2)
        n = sqrt(G M / a^3) / (2π)
    这里忽略质量比修正，取 M_total ≈ M。
    """
    M_kg = M_solar * M_SUN_SI
    M_geom_m = G_SI * M_kg / C_SI**2

    a_m = p_dimless * M_geom_m / (1.0 - e**2)
    n_hz = np.sqrt(G_SI * M_kg / a_m**3) / (2.0 * np.pi)
    return n_hz


def _kepler_p_from_nu(M_solar: float, nu: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    由 (ν, e) 反推 Newtonian 半通径 p/M（无量纲）。

    反解开普勒第三定律：
        n = sqrt(G M / a^3) / (2π)
      => a^3 = G M / (2π n)^2
      => a = [G M / (2π n)^2]^{1/3}
      => p = a (1 - e^2) / M
    """
    M_kg = M_solar * M_SUN_SI
    M_geom_m = G_SI * M_kg / C_SI**2

    # 防止除零
    nu = np.maximum(nu, 1e-20)

    a_m = (G_SI * M_kg / (2.0 * np.pi * nu)**2)**(1.0 / 3.0)
    p_dimless = a_m * (1.0 - e**2) / M_geom_m
    return p_dimless


def evolve_aak_orbit(params: EMRIParameters,
                     config: WaveformConfig) -> AAKOrbitTrajectory:
    """
    使用 BC04 高阶 PN 通量（TODO1）积分 AAK 轨道。

    输入
    ----
    params : EMRIParameters
        包含 M, μ, a, p0, e0, iota0 以及初始相位 Φ0, γ0, α0。
        - M, μ 以 M_sun 为单位；
        - a 为自旋参数 χ = S/M^2；
        - p0 为初始半通径 p/M（无量纲）。

    config : WaveformConfig
        波形配置，特别使用：
        - T, dt, atol, rtol, max_step
        - flux_model : 目前只实现 "BC04" 分支。

    输出
    ----
    AAKOrbitTrajectory
        在等间隔时间栅格上的轨道轨迹。
    """
    M_solar = params.M
    mu_solar = params.mu
    chi = params.a

    # 注意：BC04 中 λ 是 L 与 S 夹角，我们这里用 iota0 近似
    iota0 = params.iota0
    cos_lambda = float(np.cos(iota0))

    # 初始 ν：用 Newtonian 关系从 (p0, e0) 得到
    nu0 = _kepler_nu_from_p(M_solar, params.p0, params.e0)

    # 状态变量选择 (ν, e, Φ, γ, α)
    y0 = np.array([
        nu0,
        params.e0,
        params.Phi0,
        params.gamma0,
        params.alpha0,
    ], dtype=float)

    t0 = 0.0
    t1 = config.T

    def rhs_bc04(t, y):
        """
        右端函数：调用 bc04_orbital_derivatives 得到各变量导数。
        """
        nu, e, Phi, gamma, alpha = y

        dnu_dt, de_dt, dPhi_dt, dgamma_dt, dalpha_dt = bc04_orbital_derivatives(
            nu=nu,
            e=e,
            M_solar=M_solar,
            mu_solar=mu_solar,
            chi=chi,
            cos_lambda=cos_lambda,
        )

        return np.array([dnu_dt, de_dt, dPhi_dt, dgamma_dt, dalpha_dt], dtype=float)

    # 积分参数
    kwargs = dict(
        method="RK45",
        rtol=config.rtol,
        atol=config.atol,
        dense_output=True,
    )
    if config.max_step is not None:
        kwargs["max_step"] = config.max_step

    # 目前只实现 BC04 分支，如果以后需要 PM，可以在这里加 if-else
    if config.flux_model.upper() != "BC04":
        raise ValueError(
            f"WaveformConfig.flux_model={config.flux_model!r} 暂未实现，"
            "当前仅支持 'BC04'。"
        )

    sol = solve_ivp(
        rhs_bc04,
        t_span=(t0, t1),
        y0=y0,
        **kwargs,
    )

    # 在等间隔时间栅格上采样
    t_grid = np.arange(0.0, config.T, config.dt)
    y_grid = sol.sol(t_grid)
    nu_grid, e_grid, Phi_grid, gamma_grid, alpha_grid = y_grid

    # 轨道倾角目前取常数
    iota_grid = np.full_like(t_grid, iota0, dtype=float)

    # 用 Newtonian 关系从 (ν, e) 得到 p/M（无量纲）
    p_grid = _kepler_p_from_nu(M_solar, nu_grid, e_grid)

    return AAKOrbitTrajectory(
        t=t_grid,
        p=p_grid,
        e=e_grid,
        iota=iota_grid,
        Phi=Phi_grid,
        gamma=gamma_grid,
        alpha=alpha_grid,
        nu=nu_grid,
    )
