# src/emrikludge/orbits/nk_fluxes.py
"""
NK 后牛顿能流模块（radiation–reaction / fluxes）。

设计思路（对应 GG06, Ryan, Babak 等文献）：
------------------------------------------------
1. 输入轨道元素 (p, e, iota) 以及 BH 参数 (M, mu, a)；
2. 计算能量、角动量、Carter 常数的通量 (dE/dt, dLz/dt, dQ/dt)：
   - 当前实现：Peters–Mathews 0PN 通量 + constant–inclination (GHK 风格)；
   - 预留接口：Ryan 0PN+1.5PN, GG06 2PN 修正, 基于 Teukolsky 的拟合；
3. 上层（nk_geodesic_orbit.evolve_nk_orbit）通过
   (dE/dt, dLz/dt, dQ/dt) 和 Jacobian ∂(E,Lz,Q)/∂(p,e,iota) 反推
   (dp/dt, de/dt, diota/dt)。

注意：
- 这里不做 geodesic 求解，只做“弱场近似的通量”。
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..constants import G_SI, C_SI, M_SUN_SI
from .evolution_pn_fluxes import peters_mathews_fluxes


@dataclass
class NKFluxes:
    """
    存放 NK 轨道在某个时刻的能流。

    所有通量都是 SI 单位：
    - dE_dt:   能量通量 [J / s]
    - dLz_dt:  z 方向角动量通量 [kg m^2 / s^2]
    - dQ_dt:   Carter 常数通量 [几何单位下的 Q / s]（当前版本可以先置 0）
    """
    dE_dt: float
    dLz_dt: float
    dQ_dt: float


# ======== 一些帮助函数：把 (p,e) 映射为牛顿椭圆轨道的 a, E, L ========

def _newtonian_orbital_elements(M_solar: float,
                                mu_solar: float,
                                p_dimless: float,
                                e: float):
    """
    基于牛顿两体问题，把 (p, e) 映射为半长轴 a、能量 E_N 和角动量 L_N。

    - 这里采用 Schwarzschild-like 的近似：
      a = p M / (1 - e^2)，其中 M 用几何单位的长度。
    - 用的是经典 Kepler 能量/角动量：
      E_N = - G M_tot mu / (2 a)
      L_N = mu * sqrt(G M_tot a (1 - e^2))

    只在 0PN 通量中用到，作为基线近似。
    """
    # 质量：太阳质量 -> kg
    M_kg = M_solar * M_SUN_SI
    mu_kg = mu_solar * M_SUN_SI
    M_tot = M_kg + mu_kg

    # 把 M 转成几何单位长度 M_geom = GM/c^2
    M_geom = G_SI * M_kg / C_SI**2
    # 半长轴（几何单位长度）
    a_geom = p_dimless * M_geom / (1.0 - e**2)

    # Newtonian 能量和角动量
    E_N = - G_SI * M_tot * mu_kg / (2.0 * a_geom)        # [J]
    L_N = mu_kg * np.sqrt(G_SI * M_tot * a_geom * (1.0 - e**2))  # [kg m^2 / s]

    return a_geom, E_N, L_N


# ======== 0PN：Peters–Mathews 型的 baseline flux（GHK 原味） ========

def nk_fluxes_peters_ghk(p_dimless: float,
                         e: float,
                         iota: float,
                         M_solar: float,
                         mu_solar: float) -> NKFluxes:
    r"""
    基于 Peters–Mathews 0PN 通量的 GHK 风格能流近似：

    1. 使用你 AAK 那边已经实现的 Peters–Mathews 通量：
       dp/dt, de/dt = peters_mathews_fluxes(p, e; M, mu)
       这是 Schwarzschild 双体系统的 0PN 结果；
    2. 用牛顿轨道能量 E_N(p,e) 和角动量 L_N(p,e) 的链式法则得到
       dE/dt, dL/dt（见下方实现）；
    3. 采用 GHK 的 constant–inclination 近似：
       - 倾角 iota 视为常数；
       - 这相当于把 dQ/dt 选成使得 iota(t) 恒定（精确表达式可参考 GG06 式 (7),(14)）。

    文献对应关系：
    - Peters & Mathews 1963：给出 dp/dt, de/dt 的 0PN 表达式；
    - GHK / GG06：说明在 hybrid scheme 中，用 “强场定义的 p,e”
      把这些弱场通量 eval 在真实 Kerr 轨道上会有更好表现。:contentReference[oaicite:7]{index=7}

    目前实现只做“baseline”，高阶 PN 和 GG06 修正留到 nk_fluxes_gg06_2pn。
    """
    # 1. 0PN：从 Peters–Mathews 得到 (dp/dt, de/dt)，你 AAK 那边已实现
    dp_dt, de_dt = peters_mathews_fluxes(
        p_dimless=p_dimless,
        e=e,
        M_solar=M_solar,
        mu_solar=mu_solar,
    )

    # 2. 把 (dp/dt, de/dt) 转成 (dE/dt, dL/dt)：
    a_geom, E_N, L_N = _newtonian_orbital_elements(
        M_solar=M_solar,
        mu_solar=mu_solar,
        p_dimless=p_dimless,
        e=e,
    )

    # 对 E_N(p,e) 做链式求导：
    #   E_N = - G M_tot mu / (2 a)，  a = p M / (1 - e^2)
    M_kg = M_solar * M_SUN_SI
    mu_kg = mu_solar * M_SUN_SI
    M_tot = M_kg + mu_kg
    M_geom = G_SI * M_kg / C_SI**2

    # ∂a/∂p, ∂a/∂e
    da_dp = M_geom / (1.0 - e**2)
    da_de = a_geom * (2.0 * e / (1.0 - e**2))

    # ∂E/∂a
    dE_da = G_SI * M_tot * mu_kg / (2.0 * a_geom**2)

    dE_dp = dE_da * da_dp
    dE_de = dE_da * da_de

    dE_dt = dE_dp * dp_dt + dE_de * de_dt  # [J/s]

    # 对 L_N(p,e) 做链式求导：
    #   L_N = mu * sqrt(G M_tot a (1 - e^2))
    # 写成 L_N^2 = mu^2 G M_tot a (1 - e^2)
    #   ⇒ dL/dx = (1/(2 L_N)) d(L^2)/dx
    L_sq = (mu_kg**2) * G_SI * M_tot * a_geom * (1.0 - e**2)
    # 检查 L_sq 是否一致（只是 sanity check）
    # assert np.isclose(L_sq, L_N**2, rtol=1e-8)

    dLsq_da = (mu_kg**2) * G_SI * M_tot * (1.0 - e**2)
    dLsq_de = (mu_kg**2) * G_SI * M_tot * a_geom * (-2.0 * e)

    dLsq_dp = dLsq_da * da_dp
    dLsq_de = dLsq_de + dLsq_da * da_de  # 注意 a(e) 的贡献

    dL_dp = 0.5 * dLsq_dp / L_N
    dL_de = 0.5 * dLsq_de / L_N

    dL_dt = dL_dp * dp_dt + dL_de * de_dt  # [kg m^2 / s^2]

    # 3. constant–inclination 近似：暂时不演化 iota ⇒ 近似 dQ/dt = 0
    dQ_dt = 0.0

    return NKFluxes(dE_dt=dE_dt, dLz_dt=dL_dt, dQ_dt=dQ_dt)


# ======== 预留：Ryan / GG06 高阶 PN 能流 ========

def nk_fluxes_ryan_leading(p_dimless: float,
                           e: float,
                           iota: float,
                           a_spin: float,
                           M_solar: float,
                           mu_solar: float) -> NKFluxes:
    r"""
    TODO: 用 Ryan (1996) 的弱场展开给出 (dE/dt, dLz/dt, dQ/dt)。

    文献：
    - Ryan, Phys. Rev. D 53, 3064 (1996)：
      计算了 Kerr 背景下一般 (a,e,iota) 轨道的
      (\dot a, \dot e, \dot\iota)，在自旋一阶和弱场极限下完全一致。:contentReference[oaicite:8]{index=8}

    典型步骤：
    1. 定义速度参数 v = (M/p)^{1/2}；
    2. 用 Ryan 式 (2.x–3.x) 中的展开式写出 \dot a, \dot e, \dot\iota；
    3. 用 Newtonian 关系 a = p M / (1 - e^2) 把它们改写成 \dot p, \dot e, \dot\iota；
    4. 再用 E_N(p,e) 和 L_N(p,e) 的链式法则得到 \dot E, \dot L_z；
    5. Q 方面可以用 GG06 中的定义 Q(L, iota) 反推 \dot Q。

    这里先留空，后续可以按文献逐项填入具体 PN 系数。
    """
    raise NotImplementedError("nk_fluxes_ryan_leading 尚未实现，请后续按 Ryan 1996 填入 PN 系数。")


def nk_fluxes_gg06_2pn(p_dimless: float,
                       e: float,
                       iota: float,
                       a_spin: float,
                       M_solar: float,
                       mu_solar: float) -> NKFluxes:
    r"""
    TODO: GG06 改进版 2PN hybrid flux（推荐最终使用的方案）。

    文献：
    - Gair & Glampedakis, "Improved approximate inspirals of test-bodies into
      Kerr black holes", gr-qc/0510129：:contentReference[oaicite:9]{index=9}
      * 式 (4)–(6)：原始 GHK 通量的 Ryan 形式；
      * Sec. V：2PN 修正（Tagoshi 2.5PN + Shibata 2PN + Ryan leading）；
      * 式 (8)–(15)：保证 e→0, iota→0 极限下 “circular stays circular / polar stays polar”
        的一致性修正；
      * Sec. VI：Teukolsky 拟合对 circular–inclined 情形的再修正。

    建议实现步骤：
    1. 先在 Python 里把 GG06 式 (4)–(6) 的表达式完整抄写为函数；
    2. 按 Sec. V 的拼接规则加入 2PN 修正；
    3. 再根据 (8)–(15) 加上 e→0 和 iota→0 的修正项；
    4. 最后加入 circular–inclined Teuk 拟合修正，把 (dE, dLz, dQ) 替换到拟合曲线附近。

    当前版本先留出函数接口，确保上层调用结构稳定。
    """
    raise NotImplementedError("nk_fluxes_gg06_2pn 尚未实现，请后续按 GG06 逐项填入。")


# ======== 一个统一的选择接口，供 evolve_nk_orbit 使用 ========

def compute_nk_fluxes(p_dimless: float,
                      e: float,
                      iota: float,
                      a_spin: float,
                      M_solar: float,
                      mu_solar: float,
                      scheme: Literal["peters_ghk",
                                      "ryan_leading",
                                      "gg06_2pn"] = "peters_ghk") -> NKFluxes:
    """
    对外统一接口：给定 (p,e,iota,a,M,mu) 和选择的方案，返回 NKFluxes。

    scheme 选项说明：
    - "peters_ghk"   : 当前默认。0PN Peters–Mathews 通量 + constant–inclination；
                       对应 GHK 原版的思路，只是能流更直接地从 (p,e) 推到 (E,L)。
    - "ryan_leading" : 预留接口，用 Ryan 1996 的弱场展开替代 0PN；
    - "gg06_2pn"     : 预留接口，用 Gair & Glampedakis 2006 的改进 2PN 通量。

    上层（nk_geodesic_orbit.evolve_nk_orbit）只需要调用这个函数，
    不需要关心具体是哪一篇文献的 PN 系数。
    """
    if scheme == "peters_ghk":
        return nk_fluxes_peters_ghk(
            p_dimless=p_dimless,
            e=e,
            iota=iota,
            M_solar=M_solar,
            mu_solar=mu_solar,
        )
    elif scheme == "ryan_leading":
        return nk_fluxes_ryan_leading(
            p_dimless=p_dimless,
            e=e,
            iota=iota,
            a_spin=a_spin,
            M_solar=M_solar,
            mu_solar=mu_solar,
        )
    elif scheme == "gg06_2pn":
        return nk_fluxes_gg06_2pn(
            p_dimless=p_dimless,
            e=e,
            iota=iota,
            a_spin=a_spin,
            M_solar=M_solar,
            mu_solar=mu_solar,
        )
    elif scheme == "PM":
        return nk_fluxes_peters_ghk(p_dimless=p_dimless,
                                         e=e, iota=iota,
                                         M_solar=M_solar, mu_solar=mu_solar)
    else:
        raise ValueError(f"Unknown NK flux scheme: {scheme}")
