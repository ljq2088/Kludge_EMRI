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
