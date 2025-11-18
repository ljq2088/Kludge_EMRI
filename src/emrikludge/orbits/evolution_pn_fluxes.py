# src/emrikludge/orbits/evolution_pn_fluxes.py
"""
PN 轨道演化核（目前为 Peters–Mathews 级别）。

主要功能：
- 给定 (p, e) 和 (M, mu)，计算基本轨道参数的时间导数：
    - dp/dt, de/dt
- 这是 AK/AAK 的“轨道平均”演化核；
  后续可替换为 Chua+2017 AAK 中的 3PN/5PN flux。

与其他模块的关系：
- orbits.aak_osculating_orbit.AAKOrbitEvolver: 在 ODE 求解器中调用
  `peters_mathews_dpdt_dedt` 作为右端项；
- 将来如果你实现 AAK 的高阶 flux，可以在本文件新增
  `aak_fluxes_3PN` 等函数，保持相同接口。
"""

from __future__ import annotations

import numpy as np

from ..constants import G_SI, C_SI, M_SUN_SI


def peters_mathews_da_dt(a_SI: float, e: float, M_solar: float, mu_solar: float) -> float:
    """
    Peters & Mathews 公式给出的 da/dt（单位：SI，a 以米计，t 以秒计）。

    参数：
    - a_SI: 轨道半长轴 [m]
    - e:    偏心率
    - M_solar, mu_solar: MBH 和 CO 的质量 [M_sun]

    返回：
    - da/dt [m/s]
    """
    m1 = M_solar * M_SUN_SI
    m2 = mu_solar * M_SUN_SI
    Mtot = m1 + m2

    # 书写上保留 G,c，方便你以后 sanity check 维度
    pref = -64.0 / 5.0 * G_SI**3 * m1 * m2 * Mtot / C_SI**5
    denom = a_SI**3 * (1.0 - e**2)**(7.0 / 2.0)
    poly = 1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4
    return pref * poly / denom
def bc04_orbital_derivatives(
    nu: float,
    e: float,
    M_solar: float,
    mu_solar: float,
    chi: float,
    cos_lambda: float,
):
    r"""
    Barack & Cutler 2004 (Phys. Rev. D 69, 082005) Sec. II F 的轨道平均演化方程，
    对应 Eqs. (27)–(31)：

      - dΦ/dt        : Eq. (27)
      - dν/dt        : Eq. (28)
      - d\tildeγ/dt  : Eq. (29)   （我们用 gamma 表示 \tildeγ）
      - de/dt        : Eq. (30)
      - dα/dt        : Eq. (31)

    这里使用 SI 单位的时间 t [s] 与频率 ν [Hz]，但质量 M, μ 先转成
    几何单位（时间） M_geo = G M / c^3，再直接代入公式中的 (2π M ν)。
    这样几何单位下的维度是一致的。

    参数
    ----
    nu : float
        轨道平均频率 ν [Hz]（BC04 的 ν）。
    e : float
        偏心率。
    M_solar : float
        中子黑洞质量（总质量）M，以 M_sun 为单位。
    mu_solar : float
        小天体质量 μ，以 M_sun 为单位。
    chi : float
        维数无关自旋参数 χ = S / M^2，对应 EMRIParameters.a。
    cos_lambda : float
        轨道角动量 L 与自旋 S 之间夹角 λ 的余弦：cos λ = \hat{L} · \hat{S}。
        在当前实现中我们用 cos λ ≈ cos(iota0)。

    返回
    ----
    dnu_dt, de_dt, dPhi_dt, dgamma_dt, dalpha_dt : float
        五个量对物理时间的导数。
    """
    # --- 质量转成几何单位 (时间) M_geo = G M / c^3 [s] ---
    M_kg = M_solar * M_SUN_SI
    mu_kg = mu_solar * M_SUN_SI

    M_geo = G_SI * M_kg / C_SI**3   # 秒
    mu_geo = G_SI * mu_kg / C_SI**3 # 秒

    # 方便记号：u = 2π M ν，x = u^{2/3}
    u = 2.0 * np.pi * M_geo * nu
    x = u**(2.0 / 3.0)

    one_minus_e2 = 1.0 - e**2
    sqrt_one_minus_e2 = np.sqrt(one_minus_e2)

    # ============================================================
    # (27) dΦ/dt = 2πν
    # ============================================================
    dPhi_dt = 2.0 * np.pi * nu

    # ============================================================
    # (28) dν/dt
    # dν/dt = 96/(10π) * μ/M^3 * u^{11/3} (1-e^2)^(-9/2) {...}
    # ============================================================
    pref_nu = (96.0 / (10.0 * np.pi)) * (mu_geo / M_geo**3) \
        * u**(11.0 / 3.0) * one_minus_e2**(-4.5)

    term_n0 = (1.0 + (73.0 / 24.0) * e**2 + (37.0 / 96.0) * e**4) * one_minus_e2

    term_n1 = x * (
        1273.0 / 336.0
        - (2561.0 / 224.0) * e**2
        - (3885.0 / 128.0) * e**4
        - (13147.0 / 5376.0) * e**6
    )

    term_spin_n = - u * chi * cos_lambda * one_minus_e2**(-0.5) * (
        73.0 / 12.0
        + (1211.0 / 24.0) * e**2
        + (3143.0 / 96.0) * e**4
        + (65.0 / 64.0) * e**6
    )

    dnu_dt = pref_nu * (term_n0 + term_n1 + term_spin_n)

    # ============================================================
    # (29) d\tildeγ/dt  —— 这里记作 dgamma_dt
    # d\tildeγ/dt = 6πν x (1-e^2)^(-1)
    #               * [1 + (1/4)x(1-e^2)^(-1)(26 - 15 e^2)]
    #               - 12πν cosλ χ u (1-e^2)^(-3/2)
    # ============================================================
    base_pg = 6.0 * np.pi * nu * x * one_minus_e2**(-1.0)
    corr_pg = 1.0 + 0.25 * x * one_minus_e2**(-1.0) * (26.0 - 15.0 * e**2)
    spin_pg = -12.0 * np.pi * nu * cos_lambda * chi * u * one_minus_e2**(-1.5)

    dgamma_dt = base_pg * corr_pg + spin_pg

    # ============================================================
    # (30) de/dt
    # de/dt = - e/15 * μ/M^2 * u^{8/3} (1-e^2)^(-7/2) * [...]
    #         + e * μ/M^2 * χ cosλ u^{11/3} (1-e^2)^(-4) * [...]
    # ============================================================
    pref_e = - e * (mu_geo / M_geo**2) * (1.0 / 15.0) \
        * u**(8.0 / 3.0) * one_minus_e2**(-3.5)

    # 方括号第一部分
    A0 = (304.0 + 121.0 * e**2) * one_minus_e2 * (1.0 + 12.0 * x)

    # 方括号第二部分
    A1 = -(1.0 / 56.0) * x * (
        8.0 * 16705.0
        + 12.0 * 9082.0 * e**2
        - 25211.0 * e**4
    )

    de_dt_quad = pref_e * (A0 + A1)

    # 自旋项
    spin_e = e * (mu_geo / M_geo**2) * chi * cos_lambda \
        * u**(11.0 / 3.0) * one_minus_e2**(-4.0) * (
            1364.0 / 5.0
            + (5032.0 / 15.0) * e**2
            + (263.0 / 10.0) * e**4
        )

    de_dt = de_dt_quad + spin_e

    # ============================================================
    # (31) dα/dt
    # dα/dt = 4πν χ u (1-e^2)^(-3/2)
    # ============================================================
    dalpha_dt = 4.0 * np.pi * nu * chi * u * one_minus_e2**(-1.5)

    return dnu_dt, de_dt, dPhi_dt, dgamma_dt, dalpha_dt


def peters_mathews_de_dt(a_SI: float, e: float, M_solar: float, mu_solar: float) -> float:
    """
    Peters & Mathews 公式给出的 de/dt（单位：1/s）。

    参数同上。
    """
    if e == 0.0:
        return 0.0

    m1 = M_solar * M_SUN_SI
    m2 = mu_solar * M_SUN_SI
    Mtot = m1 + m2

    pref = -304.0 / 15.0 * G_SI**3 * m1 * m2 * Mtot / C_SI**5
    denom = a_SI**4 * (1.0 - e**2)**(5.0 / 2.0)
    poly = 1.0 + (121.0 / 304.0) * e**2
    return pref * e * poly / denom


def peters_mathews_dp_dt(p_dimless: float,
                         e: float,
                         M_solar: float,
                         mu_solar: float) -> float:
    """
    使用 Peters & Mathews 公式，从 (a,e) 演化得到 p 的时间导数。

    约定：
    - p_dimless: p/M，维度为无量纲（和 AK/AAK 文献习惯一致）；
    - t: 以秒计。

    使用关系：
        p = a (1 - e^2)  （这里 a 为半长轴，单位：米）
        => dp/dt = (1 - e^2) da/dt - 2 a e de/dt
    """
    # 把 M_solar -> kg，进而得到几何长度 GM/c^2：
    M_geom_m = G_SI * (M_solar * M_SUN_SI) / C_SI**2  # M in meters
    a_SI = p_dimless * M_geom_m / (1.0 - e**2)

    da_dt = peters_mathews_da_dt(a_SI, e, M_solar, mu_solar)
    de_dt = peters_mathews_de_dt(a_SI, e, M_solar, mu_solar)

    dp_dt_SI = (1.0 - e**2) * da_dt - 2.0 * a_SI * e * de_dt
    # p_dimless = p / M_geom_m，因此 p = p_dimless * M_geom_m
    # dp_dimless/dt = dp/dt / M_geom_m
    return dp_dt_SI / M_geom_m


def peters_mathews_fluxes(p_dimless: float,
                          e: float,
                          M_solar: float,
                          mu_solar: float) -> tuple[float, float]:
    """
    返回 (dp/dt, de/dt)，目前为 Peters–Mathews 级别。

    这是 AK/AAK“轨道平均”演化的最低阶版本。
    之后你可以在本文件中增加：
        aak_fluxes_3PN(...)
    并在 aak_osculating_orbit.py 中切换调用。
    """
    # 先用上面的工具函数算 dp/dt:
    dp_dt = peters_mathews_dp_dt(p_dimless, e, M_solar, mu_solar)

    # a_SI 用于 de/dt
    M_geom_m = G_SI * (M_solar * M_SUN_SI) / C_SI**2
    a_SI = p_dimless * M_geom_m / (1.0 - e**2)
    de_dt = peters_mathews_de_dt(a_SI, e, M_solar, mu_solar)

    return dp_dt, de_dt
