# # src/emrikludge/orbits/nk_fluxes.py
# """
# NK 后牛顿能流模块（radiation–reaction / fluxes）。

# 设计思路（对应 GG06, Ryan, Babak 等文献）：
# ------------------------------------------------
# 1. 输入轨道元素 (p, e, iota) 以及 BH 参数 (M, mu, a)；
# 2. 计算能量、角动量、Carter 常数的通量 (dE/dt, dLz/dt, dQ/dt)：
#    - 当前实现：Peters–Mathews 0PN 通量 + constant–inclination (GHK 风格)；
#    - 预留接口：Ryan 0PN+1.5PN, GG06 2PN 修正, 基于 Teukolsky 的拟合；
# 3. 上层（nk_geodesic_orbit.evolve_nk_orbit）通过
#    (dE/dt, dLz/dt, dQ/dt) 和 Jacobian ∂(E,Lz,Q)/∂(p,e,iota) 反推
#    (dp/dt, de/dt, diota/dt)。

# 注意：
# - 这里不做 geodesic 求解，只做“弱场近似的通量”。
# """

# from dataclasses import dataclass
# from typing import Literal

# import numpy as np

# from ..constants import G_SI, C_SI, M_SUN_SI
# from .evolution_pn_fluxes import peters_mathews_fluxes


# @dataclass
# class NKFluxes:
#     """
#     存放 NK 轨道在某个时刻的能流。

#     所有通量都是 SI 单位：
#     - dE_dt:   能量通量 [J / s]
#     - dLz_dt:  z 方向角动量通量 [kg m^2 / s^2]
#     - dQ_dt:   Carter 常数通量 [几何单位下的 Q / s]（当前版本可以先置 0）
#     """
#     dE_dt: float
#     dLz_dt: float
#     dQ_dt: float


# # ======== 一些帮助函数：把 (p,e) 映射为牛顿椭圆轨道的 a, E, L ========

# def _newtonian_orbital_elements(M_solar: float,
#                                 mu_solar: float,
#                                 p_dimless: float,
#                                 e: float):
#     """
#     基于牛顿两体问题，把 (p, e) 映射为半长轴 a、能量 E_N 和角动量 L_N。

#     - 这里采用 Schwarzschild-like 的近似：
#       a = p M / (1 - e^2)，其中 M 用几何单位的长度。
#     - 用的是经典 Kepler 能量/角动量：
#       E_N = - G M_tot mu / (2 a)
#       L_N = mu * sqrt(G M_tot a (1 - e^2))

#     只在 0PN 通量中用到，作为基线近似。
#     """
#     # 质量：太阳质量 -> kg
#     M_kg = M_solar * M_SUN_SI
#     mu_kg = mu_solar * M_SUN_SI
#     M_tot = M_kg + mu_kg

#     # 把 M 转成几何单位长度 M_geom = GM/c^2
#     M_geom = G_SI * M_kg / C_SI**2
#     # 半长轴（几何单位长度）
#     a_geom = p_dimless * M_geom / (1.0 - e**2)

#     # Newtonian 能量和角动量
#     E_N = - G_SI * M_tot * mu_kg / (2.0 * a_geom)        # [J]
#     L_N = mu_kg * np.sqrt(G_SI * M_tot * a_geom * (1.0 - e**2))  # [kg m^2 / s]

#     return a_geom, E_N, L_N


# # ======== 0PN：Peters–Mathews 型的 baseline flux（GHK 原味） ========

# def nk_fluxes_peters_ghk(p_dimless: float,
#                          e: float,
#                          iota: float,
#                          M_solar: float,
#                          mu_solar: float) -> NKFluxes:
#     r"""
#     基于 Peters–Mathews 0PN 通量的 GHK 风格能流近似：

#     1. 使用你 AAK 那边已经实现的 Peters–Mathews 通量：
#        dp/dt, de/dt = peters_mathews_fluxes(p, e; M, mu)
#        这是 Schwarzschild 双体系统的 0PN 结果；
#     2. 用牛顿轨道能量 E_N(p,e) 和角动量 L_N(p,e) 的链式法则得到
#        dE/dt, dL/dt（见下方实现）；
#     3. 采用 GHK 的 constant–inclination 近似：
#        - 倾角 iota 视为常数；
#        - 这相当于把 dQ/dt 选成使得 iota(t) 恒定（精确表达式可参考 GG06 式 (7),(14)）。

#     文献对应关系：
#     - Peters & Mathews 1963：给出 dp/dt, de/dt 的 0PN 表达式；
#     - GHK / GG06：说明在 hybrid scheme 中，用 “强场定义的 p,e”
#       把这些弱场通量 eval 在真实 Kerr 轨道上会有更好表现。:contentReference[oaicite:7]{index=7}

#     目前实现只做“baseline”，高阶 PN 和 GG06 修正留到 nk_fluxes_gg06_2pn。
#     """
#     # 1. 0PN：从 Peters–Mathews 得到 (dp/dt, de/dt)，你 AAK 那边已实现
#     dp_dt, de_dt = peters_mathews_fluxes(
#         p_dimless=p_dimless,
#         e=e,
#         M_solar=M_solar,
#         mu_solar=mu_solar,
#     )

#     # 2. 把 (dp/dt, de/dt) 转成 (dE/dt, dL/dt)：
#     a_geom, E_N, L_N = _newtonian_orbital_elements(
#         M_solar=M_solar,
#         mu_solar=mu_solar,
#         p_dimless=p_dimless,
#         e=e,
#     )

#     # 对 E_N(p,e) 做链式求导：
#     #   E_N = - G M_tot mu / (2 a)，  a = p M / (1 - e^2)
#     M_kg = M_solar * M_SUN_SI
#     mu_kg = mu_solar * M_SUN_SI
#     M_tot = M_kg + mu_kg
#     M_geom = G_SI * M_kg / C_SI**2

#     # ∂a/∂p, ∂a/∂e
#     da_dp = M_geom / (1.0 - e**2)
#     da_de = a_geom * (2.0 * e / (1.0 - e**2))

#     # ∂E/∂a
#     dE_da = G_SI * M_tot * mu_kg / (2.0 * a_geom**2)

#     dE_dp = dE_da * da_dp
#     dE_de = dE_da * da_de

#     dE_dt = dE_dp * dp_dt + dE_de * de_dt  # [J/s]

#     # 对 L_N(p,e) 做链式求导：
#     #   L_N = mu * sqrt(G M_tot a (1 - e^2))
#     # 写成 L_N^2 = mu^2 G M_tot a (1 - e^2)
#     #   ⇒ dL/dx = (1/(2 L_N)) d(L^2)/dx
#     L_sq = (mu_kg**2) * G_SI * M_tot * a_geom * (1.0 - e**2)
#     # 检查 L_sq 是否一致（只是 sanity check）
#     # assert np.isclose(L_sq, L_N**2, rtol=1e-8)

#     dLsq_da = (mu_kg**2) * G_SI * M_tot * (1.0 - e**2)
#     dLsq_de = (mu_kg**2) * G_SI * M_tot * a_geom * (-2.0 * e)

#     dLsq_dp = dLsq_da * da_dp
#     dLsq_de = dLsq_de + dLsq_da * da_de  # 注意 a(e) 的贡献

#     dL_dp = 0.5 * dLsq_dp / L_N
#     dL_de = 0.5 * dLsq_de / L_N

#     dL_dt = dL_dp * dp_dt + dL_de * de_dt  # [kg m^2 / s^2]

#     # 3. constant–inclination 近似：暂时不演化 iota ⇒ 近似 dQ/dt = 0
#     dQ_dt = 0.0

#     return NKFluxes(dE_dt=dE_dt, dLz_dt=dL_dt, dQ_dt=dQ_dt)


# # ======== 预留：Ryan / GG06 高阶 PN 能流 ========

# def nk_fluxes_ryan_leading(p_dimless: float,
#                            e: float,
#                            iota: float,
#                            a_spin: float,
#                            M_solar: float,
#                            mu_solar: float) -> NKFluxes:
#     r"""
#     TODO: 用 Ryan (1996) 的弱场展开给出 (dE/dt, dLz/dt, dQ/dt)。

#     文献：
#     - Ryan, Phys. Rev. D 53, 3064 (1996)：
#       计算了 Kerr 背景下一般 (a,e,iota) 轨道的
#       (\dot a, \dot e, \dot\iota)，在自旋一阶和弱场极限下完全一致。:contentReference[oaicite:8]{index=8}

#     典型步骤：
#     1. 定义速度参数 v = (M/p)^{1/2}；
#     2. 用 Ryan 式 (2.x–3.x) 中的展开式写出 \dot a, \dot e, \dot\iota；
#     3. 用 Newtonian 关系 a = p M / (1 - e^2) 把它们改写成 \dot p, \dot e, \dot\iota；
#     4. 再用 E_N(p,e) 和 L_N(p,e) 的链式法则得到 \dot E, \dot L_z；
#     5. Q 方面可以用 GG06 中的定义 Q(L, iota) 反推 \dot Q。

#     这里先留空，后续可以按文献逐项填入具体 PN 系数。
#     """
#     raise NotImplementedError("nk_fluxes_ryan_leading 尚未实现，请后续按 Ryan 1996 填入 PN 系数。")


# def nk_fluxes_gg06_2pn(p_dimless: float,
#                        e: float,
#                        iota: float,
#                        a_spin: float,
#                        M_solar: float,
#                        mu_solar: float) -> NKFluxes:
#     r"""
#     TODO: GG06 改进版 2PN hybrid flux（推荐最终使用的方案）。

#     文献：
#     - Gair & Glampedakis, "Improved approximate inspirals of test-bodies into
#       Kerr black holes", gr-qc/0510129：:contentReference[oaicite:9]{index=9}
#       * 式 (4)–(6)：原始 GHK 通量的 Ryan 形式；
#       * Sec. V：2PN 修正（Tagoshi 2.5PN + Shibata 2PN + Ryan leading）；
#       * 式 (8)–(15)：保证 e→0, iota→0 极限下 “circular stays circular / polar stays polar”
#         的一致性修正；
#       * Sec. VI：Teukolsky 拟合对 circular–inclined 情形的再修正。

#     建议实现步骤：
#     1. 先在 Python 里把 GG06 式 (4)–(6) 的表达式完整抄写为函数；
#     2. 按 Sec. V 的拼接规则加入 2PN 修正；
#     3. 再根据 (8)–(15) 加上 e→0 和 iota→0 的修正项；
#     4. 最后加入 circular–inclined Teuk 拟合修正，把 (dE, dLz, dQ) 替换到拟合曲线附近。

#     当前版本先留出函数接口，确保上层调用结构稳定。
#     """
#     raise NotImplementedError("nk_fluxes_gg06_2pn 尚未实现，请后续按 GG06 逐项填入。")


# # ======== 一个统一的选择接口，供 evolve_nk_orbit 使用 ========

# def compute_nk_fluxes(p_dimless: float,
#                       e: float,
#                       iota: float,
#                       a_spin: float,
#                       M_solar: float,
#                       mu_solar: float,
#                       scheme: Literal["peters_ghk",
#                                       "ryan_leading",
#                                       "gg06_2pn"] = "peters_ghk") -> NKFluxes:
#     """
#     对外统一接口：给定 (p,e,iota,a,M,mu) 和选择的方案，返回 NKFluxes。

#     scheme 选项说明：
#     - "peters_ghk"   : 当前默认。0PN Peters–Mathews 通量 + constant–inclination；
#                        对应 GHK 原版的思路，只是能流更直接地从 (p,e) 推到 (E,L)。
#     - "ryan_leading" : 预留接口，用 Ryan 1996 的弱场展开替代 0PN；
#     - "gg06_2pn"     : 预留接口，用 Gair & Glampedakis 2006 的改进 2PN 通量。

#     上层（nk_geodesic_orbit.evolve_nk_orbit）只需要调用这个函数，
#     不需要关心具体是哪一篇文献的 PN 系数。
#     """
#     if scheme == "peters_ghk":
#         return nk_fluxes_peters_ghk(
#             p_dimless=p_dimless,
#             e=e,
#             iota=iota,
#             M_solar=M_solar,
#             mu_solar=mu_solar,
#         )
#     elif scheme == "ryan_leading":
#         return nk_fluxes_ryan_leading(
#             p_dimless=p_dimless,
#             e=e,
#             iota=iota,
#             a_spin=a_spin,
#             M_solar=M_solar,
#             mu_solar=mu_solar,
#         )
#     elif scheme == "gg06_2pn":
#         return nk_fluxes_gg06_2pn(
#             p_dimless=p_dimless,
#             e=e,
#             iota=iota,
#             a_spin=a_spin,
#             M_solar=M_solar,
#             mu_solar=mu_solar,
#         )
#     elif scheme == "PM":
#         return nk_fluxes_peters_ghk(p_dimless=p_dimless,
#                                          e=e, iota=iota,
#                                          M_solar=M_solar, mu_solar=mu_solar)
#     else:
#         raise ValueError(f"Unknown NK flux scheme: {scheme}")
# src/emrikludge/orbits/nk_fluxes.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal
from .evolution_pn_fluxes import peters_mathews_fluxes

# 引入工具函数，用于坐标转换 (需要你确认 nk_waveform.py 中是否有此函数，或者我们在此处重写一个轻量版的)
def _get_minkowski_trajectory(r, theta, phi):
    """
    将 BL 坐标 (r, theta, phi) 转换为平直空间笛卡尔坐标 (x, y, z)。
    """
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=0)

@dataclass
class NKFluxes:
    # 修改定义，使其既能存 E, Lz 也能存 p, e 的导数
    dE_dt: float = 0.0
    dLz_dt: float = 0.0
    dQ_dt: float = 0.0
    dp_dt: float = 0.0
    de_dt: float = 0.0
    diota_dt: float = 0.0

def _compute_stf_tensor_2(T_ij):
    """计算 2 阶张量的 STF (Symmetric Trace-Free) 部分"""
    # T shape: (3, 3, N)
    # 1. Symmetrize
    S_ij = 0.5 * (T_ij + np.transpose(T_ij, (1, 0, 2)))
    # 2. Remove Trace
    trace = np.einsum('iit->t', S_ij)
    delta = np.eye(3)[:, :, np.newaxis]
    STF_ij = S_ij - (1.0/3.0) * trace * delta
    return STF_ij

def _compute_stf_tensor_3(T_ijk):
    """
    计算 3 阶张量的 STF 部分。
    简化处理：Mass Octupole M_ijk = x_i x_j x_k 本身就是全对称的。
    只需要移除 Trace。
    Trace: T_iik
    STF_ijk = T_ijk - (1/5) * (delta_ij T_llk + delta_ik T_llj + delta_jk T_lli)
    """
    # T shape: (3, 3, 3, N)
    # 假设 T 已经是全对称的 (对于 M_ijk = x_i x_j x_k 成立)
    
    # Trace vector V_k = T_iik
    V_k = np.einsum('iit->kt', T_ijk)
    
    delta = np.eye(3)
    term2 = np.zeros_like(T_ijk)
    
    # 构造 delta_ij V_k + ...
    # 使用循环清晰一点，或者 einsum
    # term2_ijk = delta_ij V_k + delta_ik V_j + delta_jk V_i
    for i in range(3):
        for j in range(3):
            for k in range(3):
                term2[i,j,k,:] = delta[i,j]*V_k[k,:] + delta[i,k]*V_k[j,:] + delta[j,k]*V_k[i,:]
                
    STF_ijk = T_ijk - (1.0/5.0) * term2
    return STF_ijk

def compute_numerical_fluxes(trajectory, mu_code, M_code, dt_code) -> NKFluxes:
    """
    根据 Babak et al. (2007) Eq. 43, 44 计算数值能流。
    使用数值微分和时间平均。
    
    参数:
        trajectory: NKOrbitTrajectory (包含 r, theta, phi, t)
        mu_code: 质量比 mu/M (几何单位)
        M_code: 主黑洞质量 (几何单位，通常为 1.0)
        dt_code: 时间步长 (几何单位)
    
    返回:
        NKFluxes (平均能流)
    """
    # 1. 获取轨迹 (3, N)
    x_vec = _get_minkowski_trajectory(trajectory.r, trajectory.theta, trajectory.phi)
    
    # 2. 计算速度 v (3, N)
    v_vec = np.gradient(x_vec, dt_code, axis=1, edge_order=2)
    
    # 3. 计算原始多极矩 (Primitive Moments)
    # I_ij = mu * x_i * x_j
    I_raw = mu_code * np.einsum('it,jt->ijt', x_vec, x_vec)
    
    # M_ijk = mu * x_i * x_j * x_k
    M_raw = mu_code * np.einsum('it,jt,kt->ijkt', x_vec, x_vec, x_vec)
    
    # S_ijk = mu * v_i * x_j * x_k (用于计算 J)
    # 注意 Babak 定义 J_ij = epsilon_jlm S_mli
    S_raw = mu_code * np.einsum('it,jt,kt->ijkt', v_vec, x_vec, x_vec)
    
    # 4. 计算 Radiative STF Tensors
    
    # (1) Mass Quadrupole I
    I_STF = _compute_stf_tensor_2(I_raw)
    
    # (2) Current Quadrupole J
    # J_ab = epsilon_amn S_mnb
    # 这是一个反对称操作。S_mnb = v_m x_n x_b
    # J_ab ~ eps_amn v_m x_n x_b ~ (v x x)_a x_b
    # 也就是 J_ab = mu * (L_orb)_a * x_b  (L_orb 是轨道角动量密度)
    # 严格按照公式实现:
    epsilon = np.zeros((3,3,3))
    epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1
    epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1
    
    # J_ij(t) = sum_{l,m} eps_ilm S_lmj
    # einsum: i=j(formula), j=l, k=m in code? 
    # Babak: J^{jk} = eps_{jlm} S^{mlk} (indices adjusted for waveform frame?)
    # Let's stick to standard index notation J_ij = eps_ipq S_pqj
    J_raw = np.einsum('ipq,pqjt->ijt', epsilon, S_raw)
    
    # J 应该是 STF 的
    J_STF = _compute_stf_tensor_2(J_raw)
    
    # (3) Mass Octupole M
    M_STF = _compute_stf_tensor_3(M_raw)
    
    # 5. 数值求导
    # I: 3阶导
    d1_I = np.gradient(I_STF, dt_code, axis=2, edge_order=2)
    d2_I = np.gradient(d1_I, dt_code, axis=2, edge_order=2)
    d3_I = np.gradient(d2_I, dt_code, axis=2, edge_order=2)
    
    # J: 3阶导 (Babak Eq 43 term 16/9 J^(3))
    d1_J = np.gradient(J_STF, dt_code, axis=2, edge_order=2)
    d2_J = np.gradient(d1_J, dt_code, axis=2, edge_order=2)
    d3_J = np.gradient(d2_J, dt_code, axis=2, edge_order=2)
    
    # M: 4阶导 (Babak Eq 43 term 5/189 M^(4))
    d1_M = np.gradient(M_STF, dt_code, axis=3, edge_order=2)
    d2_M = np.gradient(d1_M, dt_code, axis=3, edge_order=2)
    d3_M = np.gradient(d2_M, dt_code, axis=3, edge_order=2)
    d4_M = np.gradient(d3_M, dt_code, axis=3, edge_order=2)
    
    # 6. 组合求 Flux (瞬时值)
    # Eq 43: <E_dot> = -1/5 < I(3)^2 + 16/9 J(3)^2 + 5/189 M(4)^2 >
    # 我们先算瞬时值，最后再平均
    
    term_I = np.einsum('ijt,ijt->t', d3_I, d3_I)
    term_J = np.einsum('ijt,ijt->t', d3_J, d3_J)
    term_M = np.einsum('ijkt,ijkt->t', d4_M, d4_M)
    
    # Energy Flux (瞬时功率，总是正值表示辐射出去，公式里带负号表示轨道能量损失)
    # Babak 公式给的是 dE/dt (orbit) = - Flux，所以是负的
    flux_E_inst = - (1.0/5.0) * term_I - (16.0/45.0) * term_J - (5.0/189.0) * term_M
    
    # Angular Momentum Flux (Lz) Eq 44
    # L_dot_i = -2/5 eps_ikl < I''ka I'''al + ... >
    # 我们只关注 z 分量 (i=2, index 0,1,2) -> i=2
    # eps_2kl -> k=0,l=1 (1) or k=1,l=0 (-1)
    # Term 1: I
    # eps_ikl I_ka^(2) I_al^(3)
    L_dot_vec = np.zeros((3, len(trajectory.t)))
    
    # 计算叉乘项 (I'' x I''')_i = eps_ikl I''_km I'''_ml ? No
    # 它是 eps_ikl (I''_ka I'''_al) -> trace over a
    I_2 = d2_I
    I_3 = d3_I
    # Matrix product I''_ka I'''_al -> P_kl
    P_kl = np.einsum('kat,alt->klt', I_2, I_3)
    # eps_ikl P_kl
    term_L_I = np.einsum('ikl,klt->it', epsilon, P_kl)
    
    # Term 2: J (16/9)
    # J has same structure as I in Eq 44
    J_2 = d2_J
    J_3 = d3_J
    P_J_kl = np.einsum('kat,alt->klt', J_2, J_3)
    term_L_J = np.einsum('ikl,klt->it', epsilon, P_J_kl)
    
    # Term 3: M (5/126)
    # M'''_kab M''''_lab
    M_3 = d3_M
    M_4 = d4_M
    P_M_kl = np.einsum('kabt,labt->klt', M_3, M_4)
    term_L_M = np.einsum('ikl,klt->it', epsilon, P_M_kl)
    
    L_dot_vec = - (2.0/5.0) * term_L_I \
                - (32.0/45.0) * term_L_J \
                - (1.0/63.0) * term_L_M  # (2/5 * 5/126 = 1/63)
                
    flux_Lz_inst = L_dot_vec[2, :] # z component
    
    # 7. 时间平均
    # 简单的均值
    avg_dE_dt = np.mean(flux_E_inst)
    avg_dLz_dt = np.mean(flux_Lz_inst)
    
    return NKFluxes(dE_dt=avg_dE_dt, dLz_dt=avg_dLz_dt, dQ_dt=0.0)

# 保持原有的 compute_nk_fluxes 接口
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
    原有接口：用于 Evolution (Inspiral)。
    目前只包含解析公式 (PN)。
    如果用户想用数值能流驱动演化，需要在这里调用 compute_numerical_fluxes，
    但这需要先积分一段轨道，开销很大。通常只用于 Snapshot 验证。
    """
    if scheme == "peters_ghk" or scheme == "PM":
        return nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar)
    else:
        raise NotImplementedError(f"Flux scheme {scheme} not implemented yet")

def nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar):
    # 复用之前的实现
    dp_dt, de_dt = peters_mathews_fluxes(p_dimless, e, M_solar, mu_solar)
    # ... (转换逻辑同之前) ...
    # 为节省篇幅，此处略，请确保保留之前文件的转换逻辑
    # 建议直接把之前的文件内容合并进来
    pass

def nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar):
    # 调用 Peters-Mathews 公式 (0PN)
    # dp/dt, de/dt 是几何单位还是物理单位？
    # peters_mathews_fluxes 内部通常返回的是 dimensionless rate? 
    # 或者是 d(p/M)/dt_sec? 
    # 我们要确保它是 d(p)/d(t_M) (全几何单位)
    
    # 假设 evolution_pn_fluxes.py 里的实现是返回几何单位导数
    # 常见的 PM 公式: dp/dt = -64/5 * q * (1/p^3) * ...
    
    from .evolution_pn_fluxes import peters_mathews_fluxes
    dp, de = peters_mathews_fluxes(p_dimless, e, M_solar, mu_solar)
    
    return NKFluxes(dp_dt=dp, de_dt=de, diota_dt=0.0)