# src/emrikludge/orbits/nk_geodesic_orbit.py
"""
NK 轨道演化模块（当前版本：后开普勒 + 通量驱动 + 多频相位）。

这一层的定位：
----------------
- 作为 Babak 等人“numerical kludge”波形的“轨道生成器”；:contentReference[oaicite:3]{index=3}
- 状态变量包含慢变的 (p, e, iota) 以及快变相位 (psi_r, psi_theta, psi_phi)；
- 轨道演化由 nk_fluxes.compute_nk_fluxes 提供的 (dp/dt, de/dt, diota/dt) 驱动；
- 角向演化由一个 Kerr 频率近似 _approximate_kerr_frequencies 提供
  (Omega_r, Omega_theta, Omega_phi)。

后续升级：
-----------
- 把 _approximate_kerr_frequencies 替换成“真 Kerr 测地线频率”，
  例如通过 Fujita–Hikida 或 AAK 映射；:contentReference[oaicite:4]{index=4}
- 或者直接在这里解 Kerr geodesic ODE，并从 geodesic 中提取频率；
- 把 nk_fluxes 里的 0PN Peters–Mathews baseline 换成 Ryan / GG06 的 2PN hybrid flux。:contentReference[oaicite:5]{index=5}
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from ..parameters import EMRIParameters, WaveformConfig
from ..constants import G_SI, C_SI, M_SUN_SI
from .nk_fluxes import compute_nk_fluxes, NKFluxes


@dataclass
class NKOrbitTrajectory:
    """
    NK 轨道时间序列（当前版本：后开普勒近似）。

    Attributes
    ----------
    t : ndarray
        坐标时间 [s]
    p : ndarray
        半通径 p/M（无量纲）
    e : ndarray
        偏心率
    iota : ndarray
        倾角 [rad]

    psi_r, psi_theta, psi_phi : ndarray
        径向、极向、方位的“相位变量”，满足 dpsi/dt = Omega_{r,theta,phi}。

    r_over_M : ndarray
        r/M 的无量纲半径（由 p,e,psi_r 通过 Kepler 近似得到）；
    theta : ndarray
        近似的 Boyer–Lindquist 极角 [rad]；
    phi : ndarray
        近似的 Boyer–Lindquist 方位角 [rad]。
    """
    t: np.ndarray
    p: np.ndarray
    e: np.ndarray
    iota: np.ndarray
    psi_r: np.ndarray
    psi_theta: np.ndarray
    psi_phi: np.ndarray
    r_over_M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
def _ELQ_from_peiota(
    M: float, a: float, p: float, e: float, iota: float
) -> Tuple[float, float, float]:
    """
    从 (p, e, iota) 近似构造 Kerr 测地线不变量 (E, Lz, Q)。

    参考：
    - Gair & Glampedakis 2006, Phys. Rev. D 73, 064037（特别是 Sec. II
      中给出的 (E, Lz, Q) 拟合公式）；
    - 对于极端情况可以 fallback 到 Schmidt 2002 的解析表达式。
    """
    # TODO: 根据 GG06 的拟合公式，把 E(p,e,ι), Lz(p,e,ι), Q(p,e,ι) 写进来
    raise NotImplementedError


def _kerr_geodesic_frequencies(
    M: float, a: float, p: float, e: float, iota: float
) -> Tuple[float, float, float]:
    """
    使用 Kerr 测地线的基本频率 (Ω_r, Ω_θ, Ω_φ) 代替后开普勒频率。

    典型做法：
    1. (p,e,ι) -> (E, Lz, Q)
    2. (E,Lz,Q) -> (Ω_r, Ω_θ, Ω_φ) [Schmidt 2002; Fujita & Hikida 2009]

    目前先搭结构，具体公式可在下一步填入。
    """
    E, Lz, Q = _ELQ_from_peiota(M, a, p, e, iota)
    # TODO: 根据 Schmidt/Fujita-Hikida 的公式构造 Ω_r, Ω_θ, Ω_φ
    raise NotImplementedError


# ====== Kerr 频率的简化近似（占位：后续可替换成真 Kerr 频率）======

def _approximate_kerr_frequencies(
    M_solar: float,
    a_spin: float,
    p_dimless: float,
    e: float,
    iota: float,
) -> Tuple[float, float, float]:
    """
    返回 (Omega_r, Omega_theta, Omega_phi) [rad/s] 的简化近似。

    当前实现：从开普勒频率 Omega_K = sqrt(G M_tot / r^3)
    出发，加入一些依赖 (a, e, iota) 的经验修正因子。

    这不是严格的 Kerr 测地线频率，只是 NK 结构的“占位版本”，
    方便后续用更精确的表达式替换：
    - 可以用 Fujita–Hikida 或 Sago 等给出的 Kerr 解析频率；
    - 或者用 Chua 2017 AAK 中的三频映射公式。:contentReference[oaicite:6]{index=6}
    """
    # 质量：太阳质量 -> kg
    M_kg = M_solar * M_SUN_SI
    # 这里先忽略小天体质量对频率的修正（严格说 M_tot = M + mu）
    M_tot = M_kg

    # 取近似的“平均半径” r ~ p M（Schwarzschild-like）
    M_geom = G_SI * M_kg / C_SI**2
    r_geom = p_dimless * M_geom

    # 开普勒频率（坐标时间）：Omega_K = sqrt(G M_tot / r^3)
    Omega_K = np.sqrt(G_SI * M_tot / r_geom**3)  # [rad/s]

    # 非严格的修正：只是让频率对 (a, e, iota) 有一些依赖结构
    # 这些形式在后续可以被真正的 Kerr 频率替换掉。
    spin_factor = 1.0 + 0.5 * a_spin * np.cos(iota)
    ecc_factor = (1.0 - e**2)**(-3.0 / 2.0)

    Omega_phi = Omega_K * spin_factor
    Omega_r = Omega_K * (1.0 - 6.0 / p_dimless) * ecc_factor
    # 确保 Omega_r 不为负（极端情况下的安全措施）
    if Omega_r <= 0.0:
        Omega_r = 1e-12

    Omega_theta = Omega_K * (1.0 - 4.0 * a_spin / p_dimless * np.cos(iota))

    return Omega_r, Omega_theta, Omega_phi


# ====== 绝热演化的 RHS：通量 + 频率 ======

def _nk_adiabatic_rhs(
    t: float,
    y: np.ndarray,
    params: EMRIParameters,
    config: WaveformConfig,
) -> np.ndarray:
    """
    NK 绝热演化的一阶方程：
        y = (p, e, iota, psi_r, psi_theta, psi_phi)

    慢变量 (p, e, iota) 由 nk_fluxes 给出的通量控制，
    快变量 (psi_r, psi_theta, psi_phi) 由近似的 Kerr 频率控制。
    """
    M_solar = params.M
    mu_solar = params.mu
    a_spin = params.a

    p, e, iota, psi_r, psi_theta, psi_phi = y

    # --- 1. 慢变量：通量驱动的 (dp/dt, de/dt, diota/dt) ---
    # 当前默认方案：Peters–Mathews 0PN + constant–inclination（GHK 风格）。:contentReference[oaicite:7]{index=7}
    flux: NKFluxes = compute_nk_fluxes(
        p_dimless=p,
        e=e,
        iota=iota,
        a_spin=a_spin,
        M_solar=M_solar,
        mu_solar=mu_solar,
        scheme="peters_ghk",
    )

    # 目前 nk_fluxes_peters_ghk 是用 PM 通量 + Newtonian 链式法则得到 (dE, dL)，
    # 并没有直接返回 dp/dt, de/dt, diota/dt，所以这里暂时：
    # - 直接复用 Peters–Mathews 的 (dp/dt, de/dt)；
    # - diota/dt = 0（constant–inclination）。
    # 将来在 nk_fluxes_ryan_leading / nk_fluxes_gg06_2pn 里可以真正返回 diota/dt。

    from .evolution_pn_fluxes import peters_mathews_fluxes

    dp_dt, de_dt, diota_dt = compute_nk_fluxes(p_dimless=p, e=e, iota=iota,a_spin=a_spin,M_solar=M_solar,mu_solar=mu_solar,scheme=scheme,)
    diota_dt = 0.0  # GHK 原始假设：倾角基本不变；Ryan/GG06 会给出修正。:contentReference[oaicite:8]{index=8}

    # --- 2. 快变量：多频结构 ---
    Omega_r, Omega_theta, Omega_phi = _kerr_geodesic_frequencies(
        M=M_solar,
        a=a_spin,
        p=p,
        e=e,
        iota=iota,
    )

    dpsi_r_dt = Omega_r
    dpsi_theta_dt = Omega_theta
    dpsi_phi_dt = Omega_phi

    return np.array(
        [dp_dt_pm, de_dt_pm, diota_dt,
         dpsi_r_dt, dpsi_theta_dt, dpsi_phi_dt],
        dtype=float,
    )


# ====== 主接口：evolve_nk_orbit ======
from ..parameters import NKParameters, WaveformConfig
def evolve_nk_orbit(
    params: NKParameters,
    config: WaveformConfig,
) -> NKOrbitTrajectory:
    """
    绝热 NK 轨道演化（当前版本：PM 通量 + 后开普勒多频相位）。

    输入：
    ------
    params : EMRIParameters
        EMRI 物理参数（M, mu, a, p0, e0, iota0, 初始相位等）。
    config : WaveformConfig
        波形配置（T, dt, 积分精度等）。

    输出：
    ------
    NKOrbitTrajectory
        给出 t 序列上的 (p,e,iota, psi_r,psi_theta,psi_phi)
        以及通过简单 Kepler 映射得到的 (r/M, theta, phi)。

    说明：
    ------
    - 这个版本的 r(theta) 还不是严格 Kerr 测地线，只是
      r/M = p / (1 + e cos psi_r),
      theta = pi/2 - iota cos psi_theta,
      phi = psi_phi。
      这样做的目的是先把“多频 + 辐射反作用”整体结构搭起来；
    - 后续若要升级成真 NK，只需：
      1) 在这里把 (p,e,iota) 映射到 (E,L_z,Q)，
      2) 用 geodesic ODE 或 Kerr 解析解替换 (r,theta,phi) 部分，
      3) 通量部分用 GG06/Ryan 的 2PN hybrid flux 替代 PM 通量。:contentReference[oaicite:9]{index=9}
    """
    # 初始条件
    p0 = params.p0
    e0 = params.e0
    iota0 = params.iota0

    psi_r0 = 0.0
    psi_theta0 = 0.0

    # 如果 params 没有 phi0，就默认为 0.0
    psi_phi0 = getattr(params, "phi0", 0.0)

    
    y0 = np.array([p0, e0, iota0, psi_r0, psi_theta0, psi_phi0], dtype=float)

    t0 = 0.0
    t1 = config.T

    # 调用 solve_ivp，自适应步长，输出 dense_output 便于后插值
    ivp_kwargs = dict(
        method="RK45",
        rtol=config.rtol,
        atol=config.atol,
        dense_output=True,
    )
    if config.max_step is not None:
        ivp_kwargs["max_step"] = config.max_step

    sol = solve_ivp(
        fun=lambda t, y: _nk_adiabatic_rhs(t, y, params, config),
        t_span=(t0, t1),
        y0=y0,
        **ivp_kwargs,
    )

    # 在均匀时间栅格上采样（和波形采样率一致）
    t_grid = np.arange(t0, t1, config.dt, dtype=float)
    y_grid = sol.sol(t_grid)
    p, e, iota, psi_r, psi_theta, psi_phi = y_grid

    # 把相位映射成“伪 BL 坐标”
    # 1) r/M：Kepler 椭圆近似
    r_over_M = p / (1.0 + e * np.cos(psi_r))

    # 2) theta：“倾斜圆环 + 极向振荡”近似
    #    如果不想加极向振荡，也可以先用 theta = pi/2 - iota。
    theta = np.pi / 2.0 - iota * np.cos(psi_theta)

    # 3) phi：直接取 psi_phi
    phi = psi_phi

    return NKOrbitTrajectory(
        t=t_grid,
        p=p,
        e=e,
        iota=iota,
        psi_r=psi_r,
        psi_theta=psi_theta,
        psi_phi=psi_phi,
        r_over_M=r_over_M,
        theta=theta,
        phi=phi,
    )
