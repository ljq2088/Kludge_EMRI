# # # src/emrikludge/orbits/nk_fluxes.py
# # """
# # NK 后牛顿能流模块（radiation–reaction / fluxes）。

# # 设计思路（对应 GG06, Ryan, Babak 等文献）：
# # ------------------------------------------------
# # 1. 输入轨道元素 (p, e, iota) 以及 BH 参数 (M, mu, a)；
# # 2. 计算能量、角动量、Carter 常数的通量 (dE/dt, dLz/dt, dQ/dt)：
# #    - 当前实现：Peters–Mathews 0PN 通量 + constant–inclination (GHK 风格)；
# #    - 预留接口：Ryan 0PN+1.5PN, GG06 2PN 修正, 基于 Teukolsky 的拟合；
# # 3. 上层（nk_geodesic_orbit.evolve_nk_orbit）通过
# #    (dE/dt, dLz/dt, dQ/dt) 和 Jacobian ∂(E,Lz,Q)/∂(p,e,iota) 反推
# #    (dp/dt, de/dt, diota/dt)。

# # 注意：
# # - 这里不做 geodesic 求解，只做“弱场近似的通量”。
# # """

# # from dataclasses import dataclass
# # from typing import Literal

# # import numpy as np

# # from ..constants import G_SI, C_SI, M_SUN_SI
# # from .evolution_pn_fluxes import peters_mathews_fluxes


# # @dataclass
# # class NKFluxes:
# #     """
# #     存放 NK 轨道在某个时刻的能流。

# #     所有通量都是 SI 单位：
# #     - dE_dt:   能量通量 [J / s]
# #     - dLz_dt:  z 方向角动量通量 [kg m^2 / s^2]
# #     - dQ_dt:   Carter 常数通量 [几何单位下的 Q / s]（当前版本可以先置 0）
# #     """
# #     dE_dt: float
# #     dLz_dt: float
# #     dQ_dt: float


# # # ======== 一些帮助函数：把 (p,e) 映射为牛顿椭圆轨道的 a, E, L ========

# # def _newtonian_orbital_elements(M_solar: float,
# #                                 mu_solar: float,
# #                                 p_dimless: float,
# #                                 e: float):
# #     """
# #     基于牛顿两体问题，把 (p, e) 映射为半长轴 a、能量 E_N 和角动量 L_N。

# #     - 这里采用 Schwarzschild-like 的近似：
# #       a = p M / (1 - e^2)，其中 M 用几何单位的长度。
# #     - 用的是经典 Kepler 能量/角动量：
# #       E_N = - G M_tot mu / (2 a)
# #       L_N = mu * sqrt(G M_tot a (1 - e^2))

# #     只在 0PN 通量中用到，作为基线近似。
# #     """
# #     # 质量：太阳质量 -> kg
# #     M_kg = M_solar * M_SUN_SI
# #     mu_kg = mu_solar * M_SUN_SI
# #     M_tot = M_kg + mu_kg

# #     # 把 M 转成几何单位长度 M_geom = GM/c^2
# #     M_geom = G_SI * M_kg / C_SI**2
# #     # 半长轴（几何单位长度）
# #     a_geom = p_dimless * M_geom / (1.0 - e**2)

# #     # Newtonian 能量和角动量
# #     E_N = - G_SI * M_tot * mu_kg / (2.0 * a_geom)        # [J]
# #     L_N = mu_kg * np.sqrt(G_SI * M_tot * a_geom * (1.0 - e**2))  # [kg m^2 / s]

# #     return a_geom, E_N, L_N


# # # ======== 0PN：Peters–Mathews 型的 baseline flux（GHK 原味） ========

# # def nk_fluxes_peters_ghk(p_dimless: float,
# #                          e: float,
# #                          iota: float,
# #                          M_solar: float,
# #                          mu_solar: float) -> NKFluxes:
# #     r"""
# #     基于 Peters–Mathews 0PN 通量的 GHK 风格能流近似：

# #     1. 使用你 AAK 那边已经实现的 Peters–Mathews 通量：
# #        dp/dt, de/dt = peters_mathews_fluxes(p, e; M, mu)
# #        这是 Schwarzschild 双体系统的 0PN 结果；
# #     2. 用牛顿轨道能量 E_N(p,e) 和角动量 L_N(p,e) 的链式法则得到
# #        dE/dt, dL/dt（见下方实现）；
# #     3. 采用 GHK 的 constant–inclination 近似：
# #        - 倾角 iota 视为常数；
# #        - 这相当于把 dQ/dt 选成使得 iota(t) 恒定（精确表达式可参考 GG06 式 (7),(14)）。

# #     文献对应关系：
# #     - Peters & Mathews 1963：给出 dp/dt, de/dt 的 0PN 表达式；
# #     - GHK / GG06：说明在 hybrid scheme 中，用 “强场定义的 p,e”
# #       把这些弱场通量 eval 在真实 Kerr 轨道上会有更好表现。:contentReference[oaicite:7]{index=7}

# #     目前实现只做“baseline”，高阶 PN 和 GG06 修正留到 nk_fluxes_gg06_2pn。
# #     """
# #     # 1. 0PN：从 Peters–Mathews 得到 (dp/dt, de/dt)，你 AAK 那边已实现
# #     dp_dt, de_dt = peters_mathews_fluxes(
# #         p_dimless=p_dimless,
# #         e=e,
# #         M_solar=M_solar,
# #         mu_solar=mu_solar,
# #     )

# #     # 2. 把 (dp/dt, de/dt) 转成 (dE/dt, dL/dt)：
# #     a_geom, E_N, L_N = _newtonian_orbital_elements(
# #         M_solar=M_solar,
# #         mu_solar=mu_solar,
# #         p_dimless=p_dimless,
# #         e=e,
# #     )

# #     # 对 E_N(p,e) 做链式求导：
# #     #   E_N = - G M_tot mu / (2 a)，  a = p M / (1 - e^2)
# #     M_kg = M_solar * M_SUN_SI
# #     mu_kg = mu_solar * M_SUN_SI
# #     M_tot = M_kg + mu_kg
# #     M_geom = G_SI * M_kg / C_SI**2

# #     # ∂a/∂p, ∂a/∂e
# #     da_dp = M_geom / (1.0 - e**2)
# #     da_de = a_geom * (2.0 * e / (1.0 - e**2))

# #     # ∂E/∂a
# #     dE_da = G_SI * M_tot * mu_kg / (2.0 * a_geom**2)

# #     dE_dp = dE_da * da_dp
# #     dE_de = dE_da * da_de

# #     dE_dt = dE_dp * dp_dt + dE_de * de_dt  # [J/s]

# #     # 对 L_N(p,e) 做链式求导：
# #     #   L_N = mu * sqrt(G M_tot a (1 - e^2))
# #     # 写成 L_N^2 = mu^2 G M_tot a (1 - e^2)
# #     #   ⇒ dL/dx = (1/(2 L_N)) d(L^2)/dx
# #     L_sq = (mu_kg**2) * G_SI * M_tot * a_geom * (1.0 - e**2)
# #     # 检查 L_sq 是否一致（只是 sanity check）
# #     # assert np.isclose(L_sq, L_N**2, rtol=1e-8)

# #     dLsq_da = (mu_kg**2) * G_SI * M_tot * (1.0 - e**2)
# #     dLsq_de = (mu_kg**2) * G_SI * M_tot * a_geom * (-2.0 * e)

# #     dLsq_dp = dLsq_da * da_dp
# #     dLsq_de = dLsq_de + dLsq_da * da_de  # 注意 a(e) 的贡献

# #     dL_dp = 0.5 * dLsq_dp / L_N
# #     dL_de = 0.5 * dLsq_de / L_N

# #     dL_dt = dL_dp * dp_dt + dL_de * de_dt  # [kg m^2 / s^2]

# #     # 3. constant–inclination 近似：暂时不演化 iota ⇒ 近似 dQ/dt = 0
# #     dQ_dt = 0.0

# #     return NKFluxes(dE_dt=dE_dt, dLz_dt=dL_dt, dQ_dt=dQ_dt)


# # # ======== 预留：Ryan / GG06 高阶 PN 能流 ========

# # def nk_fluxes_ryan_leading(p_dimless: float,
# #                            e: float,
# #                            iota: float,
# #                            a_spin: float,
# #                            M_solar: float,
# #                            mu_solar: float) -> NKFluxes:
# #     r"""
# #     TODO: 用 Ryan (1996) 的弱场展开给出 (dE/dt, dLz/dt, dQ/dt)。

# #     文献：
# #     - Ryan, Phys. Rev. D 53, 3064 (1996)：
# #       计算了 Kerr 背景下一般 (a,e,iota) 轨道的
# #       (\dot a, \dot e, \dot\iota)，在自旋一阶和弱场极限下完全一致。:contentReference[oaicite:8]{index=8}

# #     典型步骤：
# #     1. 定义速度参数 v = (M/p)^{1/2}；
# #     2. 用 Ryan 式 (2.x–3.x) 中的展开式写出 \dot a, \dot e, \dot\iota；
# #     3. 用 Newtonian 关系 a = p M / (1 - e^2) 把它们改写成 \dot p, \dot e, \dot\iota；
# #     4. 再用 E_N(p,e) 和 L_N(p,e) 的链式法则得到 \dot E, \dot L_z；
# #     5. Q 方面可以用 GG06 中的定义 Q(L, iota) 反推 \dot Q。

# #     这里先留空，后续可以按文献逐项填入具体 PN 系数。
# #     """
# #     raise NotImplementedError("nk_fluxes_ryan_leading 尚未实现，请后续按 Ryan 1996 填入 PN 系数。")


# # def nk_fluxes_gg06_2pn(p_dimless: float,
# #                        e: float,
# #                        iota: float,
# #                        a_spin: float,
# #                        M_solar: float,
# #                        mu_solar: float) -> NKFluxes:
# #     r"""
# #     TODO: GG06 改进版 2PN hybrid flux（推荐最终使用的方案）。

# #     文献：
# #     - Gair & Glampedakis, "Improved approximate inspirals of test-bodies into
# #       Kerr black holes", gr-qc/0510129：:contentReference[oaicite:9]{index=9}
# #       * 式 (4)–(6)：原始 GHK 通量的 Ryan 形式；
# #       * Sec. V：2PN 修正（Tagoshi 2.5PN + Shibata 2PN + Ryan leading）；
# #       * 式 (8)–(15)：保证 e→0, iota→0 极限下 “circular stays circular / polar stays polar”
# #         的一致性修正；
# #       * Sec. VI：Teukolsky 拟合对 circular–inclined 情形的再修正。

# #     建议实现步骤：
# #     1. 先在 Python 里把 GG06 式 (4)–(6) 的表达式完整抄写为函数；
# #     2. 按 Sec. V 的拼接规则加入 2PN 修正；
# #     3. 再根据 (8)–(15) 加上 e→0 和 iota→0 的修正项；
# #     4. 最后加入 circular–inclined Teuk 拟合修正，把 (dE, dLz, dQ) 替换到拟合曲线附近。

# #     当前版本先留出函数接口，确保上层调用结构稳定。
# #     """
# #     raise NotImplementedError("nk_fluxes_gg06_2pn 尚未实现，请后续按 GG06 逐项填入。")


# # # ======== 一个统一的选择接口，供 evolve_nk_orbit 使用 ========

# # def compute_nk_fluxes(p_dimless: float,
# #                       e: float,
# #                       iota: float,
# #                       a_spin: float,
# #                       M_solar: float,
# #                       mu_solar: float,
# #                       scheme: Literal["peters_ghk",
# #                                       "ryan_leading",
# #                                       "gg06_2pn"] = "peters_ghk") -> NKFluxes:
# #     """
# #     对外统一接口：给定 (p,e,iota,a,M,mu) 和选择的方案，返回 NKFluxes。

# #     scheme 选项说明：
# #     - "peters_ghk"   : 当前默认。0PN Peters–Mathews 通量 + constant–inclination；
# #                        对应 GHK 原版的思路，只是能流更直接地从 (p,e) 推到 (E,L)。
# #     - "ryan_leading" : 预留接口，用 Ryan 1996 的弱场展开替代 0PN；
# #     - "gg06_2pn"     : 预留接口，用 Gair & Glampedakis 2006 的改进 2PN 通量。

# #     上层（nk_geodesic_orbit.evolve_nk_orbit）只需要调用这个函数，
# #     不需要关心具体是哪一篇文献的 PN 系数。
# #     """
# #     if scheme == "peters_ghk":
# #         return nk_fluxes_peters_ghk(
# #             p_dimless=p_dimless,
# #             e=e,
# #             iota=iota,
# #             M_solar=M_solar,
# #             mu_solar=mu_solar,
# #         )
# #     elif scheme == "ryan_leading":
# #         return nk_fluxes_ryan_leading(
# #             p_dimless=p_dimless,
# #             e=e,
# #             iota=iota,
# #             a_spin=a_spin,
# #             M_solar=M_solar,
# #             mu_solar=mu_solar,
# #         )
# #     elif scheme == "gg06_2pn":
# #         return nk_fluxes_gg06_2pn(
# #             p_dimless=p_dimless,
# #             e=e,
# #             iota=iota,
# #             a_spin=a_spin,
# #             M_solar=M_solar,
# #             mu_solar=mu_solar,
# #         )
# #     elif scheme == "PM":
# #         return nk_fluxes_peters_ghk(p_dimless=p_dimless,
# #                                          e=e, iota=iota,
# #                                          M_solar=M_solar, mu_solar=mu_solar)
# #     else:
# #         raise ValueError(f"Unknown NK flux scheme: {scheme}")
# # src/emrikludge/orbits/nk_fluxes.py
# # src/emrikludge/orbits/nk_fluxes.py

# import numpy as np
# from dataclasses import dataclass
# from typing import Literal
# # 假设 evolution_pn_fluxes 已存在且包含 peters_mathews_fluxes
# from .evolution_pn_fluxes import peters_mathews_fluxes

# @dataclass
# class NKFluxes:
#     """
#     存放 NK 轨道能流。包含守恒量通量(用于诊断)和轨道参数导数(用于演化)。
#     """
#     dE_dt: float = 0.0    # 能量通量 (GW 辐射带走, >0)
#     dLz_dt: float = 0.0   # 角动量通量
#     dQ_dt: float = 0.0    # Carter 常数通量
#     dp_dt: float = 0.0    # 半通径变化率 (通常 <0)
#     de_dt: float = 0.0    # 偏心率变化率 (通常 <0)
#     diota_dt: float = 0.0 # 倾角变化率

# def _get_minkowski_trajectory(r, theta, phi):
#     """将 BL 坐标 (r, theta, phi) 转换为平直空间笛卡尔坐标 (x, y, z)。"""
#     sin_theta = np.sin(theta)
#     x = r * sin_theta * np.cos(phi)
#     y = r * sin_theta * np.sin(phi)
#     z = r * np.cos(theta)
#     return np.stack([x, y, z], axis=0)

# def _compute_stf_tensor_2(T_ij):
#     """计算 2 阶张量的 STF (Symmetric Trace-Free) 部分"""
#     S_ij = 0.5 * (T_ij + np.transpose(T_ij, (1, 0, 2)))
#     trace = np.einsum('iit->t', S_ij)
#     delta = np.eye(3)[:, :, np.newaxis]
#     return S_ij - (1.0/3.0) * trace * delta

# def _compute_stf_tensor_3(T_ijk):
#     """计算 3 阶张量的 STF 部分 (Mass Octupole)"""
#     V_k = np.einsum('iit->kt', T_ijk)
#     delta = np.eye(3)
#     term2 = np.zeros_like(T_ijk)
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 term2[i,j,k,:] = delta[i,j]*V_k[k,:] + delta[i,k]*V_k[j,:] + delta[j,k]*V_k[i,:]
#     return T_ijk - (1.0/5.0) * term2

# def compute_numerical_fluxes(trajectory, mu_code, M_code, dt_code) -> NKFluxes:
#     """
#     根据 Babak et al. (2007) Eq. 43, 44 计算数值能流 <dE/dt>, <dLz/dt>。
#     注意：此函数只计算守恒量的通量，不直接提供 dp/dt, de/dt。
#     """
#     x_vec = _get_minkowski_trajectory(trajectory.r, trajectory.theta, trajectory.phi)
#     v_vec = np.gradient(x_vec, dt_code, axis=1, edge_order=2)
    
#     # 原始多极矩
#     I_raw = mu_code * np.einsum('it,jt->ijt', x_vec, x_vec)
#     M_raw = mu_code * np.einsum('it,jt,kt->ijkt', x_vec, x_vec, x_vec)
#     S_raw = mu_code * np.einsum('it,jt,kt->ijkt', v_vec, x_vec, x_vec)
    
#     # STF 多极矩
#     I_STF = _compute_stf_tensor_2(I_raw)
#     M_STF = _compute_stf_tensor_3(M_raw)
    
#     # Current Quadrupole J_ij
#     epsilon = np.zeros((3,3,3))
#     epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1
#     epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1
#     J_raw = np.einsum('ipq,pqjt->ijt', epsilon, S_raw)
#     J_STF = _compute_stf_tensor_2(J_raw)
    
#     # 数值求导
#     d1_I = np.gradient(I_STF, dt_code, axis=2, edge_order=2)
#     d2_I = np.gradient(d1_I, dt_code, axis=2, edge_order=2)
#     d3_I = np.gradient(d2_I, dt_code, axis=2, edge_order=2)
    
#     d1_J = np.gradient(J_STF, dt_code, axis=2, edge_order=2)
#     d2_J = np.gradient(d1_J, dt_code, axis=2, edge_order=2)
#     d3_J = np.gradient(d2_J, dt_code, axis=2, edge_order=2)
    
#     d1_M = np.gradient(M_STF, dt_code, axis=3, edge_order=2)
#     d2_M = np.gradient(d1_M, dt_code, axis=3, edge_order=2)
#     d3_M = np.gradient(d2_M, dt_code, axis=3, edge_order=2)
#     d4_M = np.gradient(d3_M, dt_code, axis=3, edge_order=2)
    
#     # Energy Flux Eq 43
#     term_I = np.einsum('ijt,ijt->t', d3_I, d3_I)
#     term_J = np.einsum('ijt,ijt->t', d3_J, d3_J)
#     term_M = np.einsum('ijkt,ijkt->t', d4_M, d4_M)
#     flux_E_inst = - (1.0/5.0) * term_I - (16.0/45.0) * term_J - (5.0/189.0) * term_M
    
#     # Angular Momentum Flux Eq 44 (Lz component)
#     I_2, I_3 = d2_I, d3_I
#     P_kl = np.einsum('kat,alt->klt', I_2, I_3)
#     term_L_I = np.einsum('ikl,klt->it', epsilon, P_kl)
    
#     J_2, J_3 = d2_J, d3_J
#     P_J_kl = np.einsum('kat,alt->klt', J_2, J_3)
#     term_L_J = np.einsum('ikl,klt->it', epsilon, P_J_kl)
    
#     M_3, M_4 = d3_M, d4_M
#     P_M_kl = np.einsum('kabt,labt->klt', M_3, M_4)
#     term_L_M = np.einsum('ikl,klt->it', epsilon, P_M_kl)
    
#     L_dot_vec = - (2.0/5.0) * term_L_I - (32.0/45.0) * term_L_J - (1.0/63.0) * term_L_M
#     flux_Lz_inst = L_dot_vec[2, :]
    
#     return NKFluxes(dE_dt=np.mean(flux_E_inst), dLz_dt=np.mean(flux_Lz_inst), dQ_dt=0.0)

# def nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar):
#     """
#     0PN Peters-Mathews 通量。
#     直接返回 dp/dt, de/dt 以驱动 Inspiral 演化。
#     """
#     # 调用 evolution_pn_fluxes 中已实现的公式
#     # 确保该模块返回的是几何单位下的 dp/dt (即 d(p)/d(t_M))
#     dp, de = peters_mathews_fluxes(p_dimless, e, M_solar, mu_solar)
    
#     # 对于 0PN 近似，通常假设 dE/dt, dL/dt 也可以由 Newtonian 关系推导
#     # 但对于驱动 Inspiral，dp_dt 和 de_dt 是必须的
#     return NKFluxes(dp_dt=dp, de_dt=de, diota_dt=0.0)
# # src/emrikludge/orbits/nk_fluxes.py

# import numpy as np
# from dataclasses import dataclass
# from typing import Literal, Tuple

# # 引入 Mapping 模块用于计算雅可比矩阵
# from .nk_mapping import get_conserved_quantities, KerrConstants

# @dataclass
# class NKFluxes:
#     """
#     存放 NK 轨道能流。
#     既包含物理守恒量通量 (用于诊断)，也包含几何参数导数 (用于驱动演化)。
#     """
#     # 守恒量通量 (单位质量 Specific Fluxes)
#     dE_dt: float = 0.0    # d(E_spec)/dt
#     dLz_dt: float = 0.0   # d(Lz_spec)/dt
#     dQ_dt: float = 0.0    # d(Q_spec)/dt
    
#     # 轨道几何参数导数 (用于积分器)
#     dp_dt: float = 0.0
#     de_dt: float = 0.0
#     diota_dt: float = 0.0

# # ==============================================================================
# # Gair & Glampedakis (2006) 2PN Flux Coefficients
# # ==============================================================================

# def _g_coeffs(e: float) -> dict:
#     """
#     计算 GG06 Eq. (39) 和 Eq. (46) 中的 g_n(e) 系数。
#     """
#     e2 = e*e
#     e4 = e2*e2
#     e6 = e4*e2
    
#     g = {}
#     g[1] = 1 + (73/24)*e2 + (37/96)*e4
#     g[2] = (73/12) + (823/24)*e2 + (949/32)*e4 + (491/192)*e6
#     g[3] = (1247/336) + (9181/672)*e2
#     g[4] = 4 + (1375/48)*e2
#     g[5] = (44711/9072) + (172157/2592)*e2
#     g[6] = (33/16) + (359/32)*e2
#     g[7] = (8191/672) + (44531/336)*e2
#     g[8] = (3749/336) - (5143/168)*e2
#     g[9] = 1 + (7/8)*e2
#     g[10] = (61/12) + (119/8)*e2 + (183/32)*e4 # Eq 39 for equatorial
#     g[11] = (1247/336) + (425/336)*e2
#     g[12] = 4 + (97/8)*e2
#     g[13] = (44711/9072) + (302893/6048)*e2
#     g[14] = (33/16) + (95/16)*e2
#     g[15] = (8191/672) + (48361/1344)*e2
#     g[16] = (417/56) - (37241/672)*e2
    
#     # Eq 46 (Split for inclined Lz)
#     g['10a'] = (61/24) + (63/8)*e2 + (95/64)*e4
#     g['10b'] = (61/8) + (91/4)*e2 + (461/64)*e4
    
#     return g

# def _N_coeffs(p: float, M: float, a: float, E_circ: float, L_circ: float, iota: float) -> Tuple[float, float, float]:
#     """
#     计算 GG06 Eq. (10) 和 (15) 中的 N 系数，用于 Circular Fix。
#     这些系数需要在 e -> 0 的极限下计算 (即使用圆轨道的 E, L)。
#     """
#     # 注意：公式中的 N 是针对 r 的函数，但在 Eq 13/14 极限下，r -> p (对于 e=0)
#     # 实际上 Eq 11 提到 N1(p, iota) = N1(ra) = N2(rp) at e=0
    
#     # Eq 10 N1 (at r=p)
#     # N1 = E*r^4 + a^2*E*r^2 - 2*a*M*(L - a*E)*r
#     # 使用 M_code = 1.0
#     N1 = E_circ * p**4 + (a**2) * E_circ * p**2 - 2*a*M * (L_circ - a*E_circ) * p
    
#     # Eq 15 N4, N5
#     N4 = (2*M*p - p**2) * L_circ - 2*M*a*E_circ * p
#     N5 = (2*M*p - p**2 - a**2) / 2.0
    
#     return N1, N4, N5

# # ==============================================================================
# # Core Flux Calculation (2PN + Corrections)
# # ==============================================================================

# def _calc_gg06_fluxes_raw(p: float, e: float, iota: float, a: float, M: float, mu: float) -> Tuple[float, float, float]:
#     """
#     计算 GG06 2PN 通量 (dE/dt, dLz/dt, dQ/dt)。
    
#     返回的是【粒子总能量/角动量】的通量 (Total Fluxes)，量级为 mu^2。
#     遵循 Gair & Glampedakis (2006) Eqs 44, 45, 56。
#     """
#     # 预计算
#     q = a / M
#     v2 = M / p
#     v = np.sqrt(v2)
#     Mp = v2 # (M/p)
#     safe_e = e
#     if e >= 1.0:
#         # 策略 A: 强制钳位到接近 1 (例如 0.999) 继续算 (不推荐，物理上不正确)
#         # 策略 B: 认为已经 Plunge，通量设为 0 或其他标记值
#         # 这里为了不报错，我们暂时钳位，但最好是在积分器里停止
#         safe_e = 0.999
#     # 角度函数
#     cos_i = np.cos(iota)
#     sin_i = np.sin(iota)
#     sin2_i = sin_i**2
#     cos2_i = cos_i**2
    
#     g = _g_coeffs(safe_e)
#     prefix = (1 - safe_e*safe_e)**1.5
    
#     # --- Energy Flux (Eq 44) ---
#     # E_dot_2PN / factor
#     # factor = - (32/5) * (mu/M)^2 * (M/p)^5
#     factor_E = -(32.0/5.0) * (mu/M)**2 * (Mp**5)
    
#     term_E = (
#         g[1] 
#         - q * (Mp**1.5) * g[2] * cos_i
#         - Mp * g[3]
#         + np.pi * (Mp**1.5) * g[4]
#         - (Mp**2) * g[5]
#         + (q**2) * (Mp**2) * g[6]
#         - (527.0/96.0) * (q**2) * (Mp**2) * sin2_i
#     )
#     E_dot_2PN = factor_E * prefix * term_E
    
#     # --- Angular Momentum Flux (Eq 45) ---
#     # L_dot_2PN / factor_L
#     # factor_L = - (32/5) * (mu/M) * (M/p)^3.5
#     factor_L = -(32.0/5.0) * (mu/M) * (Mp**3.5)
    
#     term_L = (
#         g[9] * cos_i
#         + q * (Mp**1.5) * (g['10a'] - cos2_i * g['10b'])
#         - Mp * g[11] * cos_i
#         + np.pi * (Mp**1.5) * g[12] * cos_i
#         - (Mp**2) * g[13] * cos_i
#         + (q**2) * (Mp**2) * cos_i * (g[14] - (45.0/8.0)*sin2_i)
#     )
#     L_dot_2PN = factor_L * prefix * term_L
    
#     # --- Carter Constant Flux (Eq 56) ---
#     # 计算 dQ/dt。注意 Eq 56 给出的是 dQ/dt / sqrt(Q)。
#     # 我们需要计算出 Q 的值来恢复 dQ/dt。
#     # 这里需要调用 mapping 得到当前轨道的 Q_spec。
#     # 注意：Flux 公式里的 Q 通常指 Specific Q 还是 Total Q?
#     # Eq 56 右边系数: -(64/5) * (mu^2/M) ...
#     # 如果 Q 是 Specific (M^2)，sqrt(Q) ~ M. RHS ~ mu^2. dQ/dt ~ mu^2.
#     # Q_spec 变化率应该是 mu. 所以这里的公式算的是 d(Q_total)/dt ?
#     # 让我们看 Eq 23: Q_dot = ... Lz * Lz_dot ...
#     # 如果 Lz 是 total (~mu), Lz_dot (~mu^2) -> Q_dot ~ mu^3.
#     # Eq 56 RHS: mu^2 * sqrt(Q). 如果 Q is total (~mu^2), RHS ~ mu^3.
#     # 所以 Eq 56 是 Total Q 的公式。
    
#     # 为了计算方便，我们先算出 Specific Q，再转为 Total Q
#     try:
#         k_consts = get_conserved_quantities(M, a, p, safe_e, iota)
#         Q_spec = k_consts.Q
#     except:
#         # Mapping 失败时的回退 (通常不应该发生，除非 plunge)
#         Q_spec = 0.0 

#     # Total Q = mu^2 * Q_spec
#     Q_total = (mu**2) * Q_spec
#     sqrt_Q_total = np.sqrt(np.abs(Q_total))
    
#     # Eq 56 RHS (without sqrt(Q) factor)
#     # Coeff = - (64/5) * (mu^2/M) * (M/p)^3.5 * sin(i) * (1-e^2)^1.5
#     factor_Q = -(64.0/5.0) * (mu**2/M) * (Mp**3.5) * sin_i * prefix
    
#     term_Q = (
#         g[9] 
#         - q * (Mp**1.5) * cos_i * g['10b']
#         - Mp * g[11]
#         + np.pi * (Mp**1.5) * g[12]
#         - (Mp**2) * g[13]
#         + (q**2) * (Mp**2) * (g[14] - (45.0/8.0)*sin2_i)
#     )
    
#     Q_dot_2PN = factor_Q * sqrt_Q_total * term_Q
    
#     return E_dot_2PN, L_dot_2PN, Q_dot_2PN

# def nk_fluxes_gg06_2pn(p: float, e: float, iota: float, a_spin: float, M_solar: float, mu_solar: float) -> NKFluxes:
#     """
#     计算 GG06 改进版能流，包含 Circular Fix (Eq 20)。
#     """
#     # 1. 转换单位到几何单位 (Code Units M=1)
#     # 输入 p 已经是 p/M。a_spin 是 a/M。
#     # 这里的 M_solar, mu_solar 仅用于计算质量比 q_mass
#     q_mass = mu_solar / M_solar
#     M_code = 1.0
#     mu_code = q_mass * M_code # 在 M=1 单位制下，mu 就是质量比
    
#     # 2. 计算当前轨道的 2PN Fluxes (Total Fluxes in Code Units)
#     E_dot, L_dot, Q_dot = _calc_gg06_fluxes_raw(p, e, iota, a_spin, M_code, mu_code)
    
#     # 3. 应用 Near-Circular Correction (GG06 Sec III, Eq 20)
#     # 修正 E_dot 以满足圆轨道一致性条件
    
#     # 计算对应的圆轨道 (e=0) 的通量
#     E_dot_circ, L_dot_circ, Q_dot_circ = _calc_gg06_fluxes_raw(p, 0.0, iota, a_spin, M_code, mu_code)
    
#     # 计算 circular orbit 的 N 系数 (需要 Mapping 得到圆轨道的 E, L, Q)
#     try:
#         k_circ = get_conserved_quantities(M_code, a_spin, p, 0.0, iota)
#         N1, N4, N5 = _N_coeffs(p, M_code, a_spin, k_circ.E, k_circ.Lz, iota)
        
#         # Eq 20: E_dot_mod = (1-e^2)^1.5 * [ ... ]
#         prefix = (1 - e*e)**1.5
        
#         # 括号内的项
#         term1 = E_dot / prefix # 移除 (1-e^2)^1.5 因子
#         term2 = E_dot_circ     # 已经是 e=0
#         term3 = (N4/N1) * L_dot_circ
#         # 注意：公式里是用 Q_dot 还是 iota_dot?
#         # Eq 20 原文是用 Q_dot (Eq 20 last term: - (N5/N1) * Q_dot_circ)
#         # 我们用的是 Total Q_dot，N1/N5 是基于 Specific 量的吗？
#         # N1, N4, N5 量级：N1 ~ E*p^4 ~ p^4. N4 ~ p*L ~ p*sqrt(p) ~ p^1.5.
#         # Eq 14: N1*E_dot + N4*L_dot + N5*Q_dot = 0
#         # 这是一个齐次方程。如果我们用 Total Fluxes (E_tot ~ mu, L_tot ~ mu, Q_tot ~ mu^2)
#         # 那么各项量级：
#         # N1*E ~ p^4 * mu
#         # N4*L ~ p^1.5 * mu
#         # N5*Q ~ p * mu^2
#         # 明显量级不对！N 系数是基于 Specific quantities (e.g. D(r) Eq 10) 推导的。
#         # 所以 Eq 14 成立的前提是 E, L, Q 都是 Specific Fluxes。
        
#         # >>> 关键单位转换 <<<
#         # 将 Total Fluxes 转为 Specific Fluxes
#         # E_spec_dot = E_tot_dot / mu
#         # L_spec_dot = L_tot_dot / mu
#         # Q_spec_dot = Q_tot_dot / mu^2
        
#         E_spec_dot = E_dot / mu_code
#         E_spec_dot_circ = E_dot_circ / mu_code
#         L_spec_dot_circ = L_dot_circ / mu_code
#         Q_spec_dot_circ = Q_dot_circ / (mu_code**2)
        
#         # 在 Specific 层面应用 Eq 20
#         term1_s = E_spec_dot / prefix
#         correction = E_spec_dot_circ + (N4/N1)*L_spec_dot_circ + (N5/N1)*Q_spec_dot_circ
        
#         E_spec_dot_mod = prefix * (term1_s - correction)
        
#         # 最终 Fluxes (Specific)
#         # 这里我们直接返回 Specific Fluxes，因为 Mapping 也是 Specific 的
#         dE_spec = E_spec_dot_mod
#         dL_spec = L_dot / mu_code
#         dQ_spec = Q_dot / (mu_code**2)
        
#     except Exception as err:
#         # 如果 Mapping 失败 (例如 ISCO 附近)，降级为未修正通量
#         # print(f"Warning: Circular fix failed ({err}), using raw 2PN.")
#         dE_spec = E_dot / mu_code
#         dL_spec = L_dot / mu_code
#         dQ_spec = Q_dot / (mu_code**2)

#     # ==========================================================================
#     # 4. 坐标转换 (Jacobian): (E, L, Q) -> (p, e, iota)
#     # ==========================================================================
    
#     # 我们需要计算 Jacobian J = d(E,L,Q)/d(p,e,iota)
#     # 然后解 J * [dp, de, diota]^T = [dE, dL, dQ]^T
    
#     # 4.1 数值计算 Jacobian (中心差分)
#     # 步长
#     dp = 1e-4 * p
#     de_step = 1e-5 if e > 1e-4 else 1e-6
#     di = 1e-5
    
#     # 基准点
#     k0 = get_conserved_quantities(M_code, a_spin, p, e, iota)
#     y0 = np.array([k0.E, k0.Lz, k0.Q])
    
#     # 对 p 扰动
#     kp = get_conserved_quantities(M_code, a_spin, p+dp, e, iota)
#     km = get_conserved_quantities(M_code, a_spin, p-dp, e, iota)
#     dydp = (np.array([kp.E, kp.Lz, kp.Q]) - np.array([km.E, km.Lz, km.Q])) / (2*dp)
    
#     # 对 e 扰动
#     ke_p = get_conserved_quantities(M_code, a_spin, p, e+de_step, iota)
#     ke_m = get_conserved_quantities(M_code, a_spin, p, e-de_step, iota)
#     dyde = (np.array([ke_p.E, ke_p.Lz, ke_p.Q]) - np.array([ke_m.E, ke_m.Lz, ke_m.Q])) / (2*de_step)
    
#     # 对 iota 扰动
#     ki_p = get_conserved_quantities(M_code, a_spin, p, e, iota+di)
#     ki_m = get_conserved_quantities(M_code, a_spin, p, e, iota-di)
#     dydi = (np.array([ki_p.E, ki_p.Lz, ki_p.Q]) - np.array([ki_m.E, ki_m.Lz, ki_m.Q])) / (2*di)
    
#     # 组装 Jacobian Matrix
#     # J = [[dE/dp, dE/de, dE/di],
#     #      [dL/dp, ...         ],
#     #      [dQ/dp, ...         ]]
#     J = np.column_stack((dydp, dyde, dydi))
    
#     # 4.2 求解线性方程组
#     # Y_dot = [dE_spec, dL_spec, dQ_spec]
#     Y_dot = np.array([dE_spec, dL_spec, dQ_spec])
    
#     # X_dot = J^-1 * Y_dot
#     try:
#         X_dot = np.linalg.solve(J, Y_dot)
#         dp_dt_val, de_dt_val, diota_dt_val = X_dot
#     except np.linalg.LinAlgError:
#         # 奇异矩阵 (例如圆轨道或极轨道极限)，回退或置零
#         dp_dt_val, de_dt_val, diota_dt_val = 0.0, 0.0, 0.0

#     return NKFluxes(
#         dE_dt=dE_spec, dLz_dt=dL_spec, dQ_dt=dQ_spec,
#         dp_dt=dp_dt_val, de_dt=de_dt_val, diota_dt=diota_dt_val
#     )

# # ... (保留 compute_numerical_fluxes, nk_fluxes_peters_ghk, compute_nk_fluxes) ...
# # 记得在 compute_nk_fluxes 中添加 scheme="gg06_2pn" 的分支调用
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
#     对外接口：计算驱动轨道演化的通量 (dp/dt, de/dt)。
#     """
#     if scheme == "peters_ghk" or scheme == "PM":
#         return nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar)
#     elif scheme == "ryan_leading":
#         return nk_fluxes_ryan_leading(p_dimless, e, iota, a_spin, M_solar, mu_solar)
#     elif scheme == "gg06_2pn":
#         return nk_fluxes_gg06_2pn(p_dimless, e, iota, a_spin, M_solar, mu_solar)
#     else:
#         raise NotImplementedError(f"Flux scheme {scheme} not implemented yet")
# src/emrikludge/orbits/nk_fluxes.py

import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple

# 引入 Mapping 模块用于计算雅可比矩阵
from .nk_mapping import get_conserved_quantities, KerrConstants

@dataclass
class NKFluxes:
    """
    存放 NK 轨道能流。
    既包含物理守恒量通量 (用于诊断)，也包含几何参数导数 (用于驱动演化)。
    """
    # 守恒量通量 (单位质量 Specific Fluxes)
    dE_dt: float = 0.0    # d(E_spec)/dt
    dLz_dt: float = 0.0   # d(Lz_spec)/dt
    dQ_dt: float = 0.0    # d(Q_spec)/dt
    
    # 轨道几何参数导数 (用于积分器)
    dp_dt: float = 0.0
    de_dt: float = 0.0
    diota_dt: float = 0.0

# ==============================================================================
# Gair & Glampedakis (2006) 2PN Flux Coefficients
# ==============================================================================

def _g_coeffs(e: float) -> dict:
    """
    计算 GG06 Eq. (39) 和 Eq. (46) 中的 g_n(e) 系数。
    """
    e2 = e*e
    e4 = e2*e2
    e6 = e4*e2
    
    g = {}
    g[1] = 1 + (73/24)*e2 + (37/96)*e4
    g[2] = (73/12) + (823/24)*e2 + (949/32)*e4 + (491/192)*e6
    g[3] = (1247/336) + (9181/672)*e2
    g[4] = 4 + (1375/48)*e2
    g[5] = (44711/9072) + (172157/2592)*e2
    g[6] = (33/16) + (359/32)*e2
    g[7] = (8191/672) + (44531/336)*e2
    g[8] = (3749/336) - (5143/168)*e2
    g[9] = 1 + (7/8)*e2
    g[10] = (61/12) + (119/8)*e2 + (183/32)*e4 # Eq 39 for equatorial
    g[11] = (1247/336) + (425/336)*e2
    g[12] = 4 + (97/8)*e2
    g[13] = (44711/9072) + (302893/6048)*e2
    g[14] = (33/16) + (95/16)*e2
    g[15] = (8191/672) + (48361/1344)*e2
    g[16] = (417/56) - (37241/672)*e2
    
    # Eq 46 (Split for inclined Lz)
    g['10a'] = (61/24) + (63/8)*e2 + (95/64)*e4
    g['10b'] = (61/8) + (91/4)*e2 + (461/64)*e4
    
    return g

def _N_coeffs(p: float, M: float, a: float, E_circ: float, L_circ: float, iota: float) -> Tuple[float, float, float]:
    """
    计算 GG06 Eq. (10) 和 (15) 中的 N 系数，用于 Circular Fix。
    这些系数需要在 e -> 0 的极限下计算 (即使用圆轨道的 E, L)。
    """
    # Eq 10 N1 (at r=p)
    N1 = E_circ * p**4 + (a**2) * E_circ * p**2 - 2*a*M * (L_circ - a*E_circ) * p
    
    # Eq 15 N4, N5
    N4 = (2*M*p - p**2) * L_circ - 2*M*a*E_circ * p
    N5 = (2*M*p - p**2 - a**2) / 2.0
    
    return N1, N4, N5

# ==============================================================================
# Core Flux Calculation (2PN + Corrections)
# ==============================================================================

def _calc_gg06_fluxes_raw(p: float, e: float, iota: float, a: float, M: float, mu: float) -> Tuple[float, float, float]:
    """
    计算 GG06 2PN 通量 (dE/dt, dLz/dt, dQ/dt)。
    """
    # 预计算
    q = a / M
    v2 = M / p
    Mp = v2 # (M/p)
    
    # 🛡️【增强版安全钳位】
    # 积分器可能会尝试 e > 1 或者 e < 0 的值，必须强制钳位在物理范围内
    safe_e = e
    if e > 0.999:
        safe_e = 0.999
    elif e < 0.0:
        safe_e = 0.0
        
    # 角度函数
    cos_i = np.cos(iota)
    sin_i = np.sin(iota)
    sin2_i = sin_i**2
    cos2_i = cos_i**2
    
    # 使用 safe_e 计算系数
    g = _g_coeffs(safe_e)
    
    # 计算前缀：现在保证 (1 - safe_e^2) 永远大于 0
    prefix = (1.0 - safe_e*safe_e)**1.5
    
    # --- Energy Flux (Eq 44) ---
    factor_E = -(32.0/5.0) * (mu/M)**2 * (Mp**5)
    
    term_E = (
        g[1] 
        - q * (Mp**1.5) * g[2] * cos_i
        - Mp * g[3]
        + np.pi * (Mp**1.5) * g[4]
        - (Mp**2) * g[5]
        + (q**2) * (Mp**2) * g[6]
        - (527.0/96.0) * (q**2) * (Mp**2) * sin2_i
    )
    E_dot_2PN = factor_E * prefix * term_E
    
    # --- Angular Momentum Flux (Eq 45) ---
    factor_L = -(32.0/5.0) * (mu/M) * (Mp**3.5)
    
    term_L = (
        g[9] * cos_i
        + q * (Mp**1.5) * (g['10a'] - cos2_i * g['10b'])
        - Mp * g[11] * cos_i
        + np.pi * (Mp**1.5) * g[12] * cos_i
        - (Mp**2) * g[13] * cos_i
        + (q**2) * (Mp**2) * cos_i * (g[14] - (45.0/8.0)*sin2_i)
    )
    L_dot_2PN = factor_L * prefix * term_L
    
    # --- Carter Constant Flux (Eq 56) ---
    try:
        k_consts = get_conserved_quantities(M, a, p, safe_e, iota)
        Q_spec = k_consts.Q
    except:
        Q_spec = 0.0 

    Q_total = (mu**2) * Q_spec
    sqrt_Q_total = np.sqrt(np.abs(Q_total))
    
    factor_Q = -(64.0/5.0) * (mu**2/M) * (Mp**3.5) * sin_i * prefix
    
    term_Q = (
        g[9] 
        - q * (Mp**1.5) * cos_i * g['10b']
        - Mp * g[11]
        + np.pi * (Mp**1.5) * g[12]
        - (Mp**2) * g[13]
        + (q**2) * (Mp**2) * (g[14] - (45.0/8.0)*sin2_i)
    )
    
    Q_dot_2PN = factor_Q * sqrt_Q_total * term_Q
    
    return E_dot_2PN, L_dot_2PN, Q_dot_2PN

def nk_fluxes_gg06_2pn(p: float, e: float, iota: float, a_spin: float, M_solar: float, mu_solar: float) -> NKFluxes:
    """
    计算 GG06 改进版能流，包含 Circular Fix (Eq 20)。
    """
    # 1. 转换单位到几何单位 (Code Units M=1)
    q_mass = mu_solar / M_solar
    M_code = 1.0
    mu_code = q_mass * M_code
    
    safe_e = e
    if e > 0.999:
        safe_e = 0.999
    elif e < 0.0:
        safe_e = 0.0
    
    # 2. 计算当前轨道的 2PN Fluxes (传入 safe_e)
    E_dot, L_dot, Q_dot = _calc_gg06_fluxes_raw(p, safe_e, iota, a_spin, M_code, mu_code)
    
    # 3. 应用 Near-Circular Correction (GG06 Sec III, Eq 20)
    E_dot_circ, L_dot_circ, Q_dot_circ = _calc_gg06_fluxes_raw(p, 0.0, iota, a_spin, M_code, mu_code)
    
    try:
        k_circ = get_conserved_quantities(M_code, a_spin, p, 0.0, iota)
        N1, N4, N5 = _N_coeffs(p, M_code, a_spin, k_circ.E, k_circ.Lz, iota)
        
        # 🛡️【修正】使用 safe_e 计算 prefix
        prefix = (1.0 - safe_e*safe_e)**1.5
        
        E_spec_dot = E_dot / mu_code
        E_spec_dot_circ = E_dot_circ / mu_code
        L_spec_dot_circ = L_dot_circ / mu_code
        Q_spec_dot_circ = Q_dot_circ / (mu_code**2)
        
        term1_s = E_spec_dot / prefix
        correction = E_spec_dot_circ + (N4/N1)*L_spec_dot_circ + (N5/N1)*Q_spec_dot_circ
        
        E_spec_dot_mod = prefix * (term1_s - correction)
        
        dE_spec = E_spec_dot_mod
        dL_spec = L_dot / mu_code
        dQ_spec = Q_dot / (mu_code**2)
        
    except Exception as err:
        dE_spec = E_dot / mu_code
        dL_spec = L_dot / mu_code
        dQ_spec = Q_dot / (mu_code**2)

    # ==========================================================================
    # 4. 坐标转换 (Jacobian): (E, L, Q) -> (p, e, iota)
    # ==========================================================================
    
    # Jacobian 计算使用 safe_e
    dp = 1e-4 * p
    de_step = 1e-5 if safe_e > 1e-4 else 1e-6
    di = 1e-5
    
    # 基准点
    k0 = get_conserved_quantities(M_code, a_spin, p, safe_e, iota)
    y0 = np.array([k0.E, k0.Lz, k0.Q])
    
    # 对 p 扰动
    kp = get_conserved_quantities(M_code, a_spin, p+dp, safe_e, iota)
    km = get_conserved_quantities(M_code, a_spin, p-dp, safe_e, iota)
    dydp = (np.array([kp.E, kp.Lz, kp.Q]) - np.array([km.E, km.Lz, km.Q])) / (2*dp)
    
    # 对 e 扰动 (注意边界)
    e_plus = min(safe_e + de_step, 0.999)
    e_minus = max(safe_e - de_step, 0.0)
    ke_p = get_conserved_quantities(M_code, a_spin, p, e_plus, iota)
    ke_m = get_conserved_quantities(M_code, a_spin, p, e_minus, iota)
    dyde = (np.array([ke_p.E, ke_p.Lz, ke_p.Q]) - np.array([ke_m.E, ke_m.Lz, ke_m.Q])) / (e_plus - e_minus)
    
    # 对 iota 扰动
    ki_p = get_conserved_quantities(M_code, a_spin, p, safe_e, iota+di)
    ki_m = get_conserved_quantities(M_code, a_spin, p, safe_e, iota-di)
    dydi = (np.array([ki_p.E, ki_p.Lz, ki_p.Q]) - np.array([ki_m.E, ki_m.Lz, ki_m.Q])) / (2*di)
    
    J = np.column_stack((dydp, dyde, dydi))
    
    Y_dot = np.array([dE_spec, dL_spec, dQ_spec])
    
    try:
        X_dot = np.linalg.solve(J, Y_dot)
        dp_dt_val, de_dt_val, diota_dt_val = X_dot
    except np.linalg.LinAlgError:
        dp_dt_val, de_dt_val, diota_dt_val = 0.0, 0.0, 0.0

    return NKFluxes(
        dE_dt=dE_spec, dLz_dt=dL_spec, dQ_dt=dQ_spec,
        dp_dt=dp_dt_val, de_dt=de_dt_val, diota_dt=diota_dt_val
    )

def nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar):
    """
    0PN Peters-Mathews 通量。
    """
    from .evolution_pn_fluxes import peters_mathews_fluxes
    dp, de = peters_mathews_fluxes(p_dimless, e, M_solar, mu_solar)
    return NKFluxes(dp_dt=dp, de_dt=de, diota_dt=0.0)

def nk_fluxes_ryan_leading(p_dimless, e, iota, a_spin, M_solar, mu_solar):
    raise NotImplementedError("Ryan leading flux not implemented")

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
    对外接口：计算驱动轨道演化的通量 (dp/dt, de/dt)。
    """
    if scheme == "peters_ghk" or scheme == "PM":
        return nk_fluxes_peters_ghk(p_dimless, e, iota, M_solar, mu_solar)
    elif scheme == "ryan_leading":
        return nk_fluxes_ryan_leading(p_dimless, e, iota, a_spin, M_solar, mu_solar)
    elif scheme == "gg06_2pn":
        return nk_fluxes_gg06_2pn(p_dimless, e, iota, a_spin, M_solar, mu_solar)
    else:
        raise NotImplementedError(f"Flux scheme {scheme} not implemented yet")

# 保留 compute_numerical_fluxes
def compute_numerical_fluxes(trajectory, mu_code, M_code, dt_code) -> NKFluxes:
    # 引入工具函数，用于坐标转换
    def _get_minkowski_trajectory(r, theta, phi):
        sin_theta = np.sin(theta)
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=0)

    def _compute_stf_tensor_2(T_ij):
        S_ij = 0.5 * (T_ij + np.transpose(T_ij, (1, 0, 2)))
        trace = np.einsum('iit->t', S_ij)
        delta = np.eye(3)[:, :, np.newaxis]
        return S_ij - (1.0/3.0) * trace * delta

    def _compute_stf_tensor_3(T_ijk):
        V_k = np.einsum('iit->kt', T_ijk)
        delta = np.eye(3)
        term2 = np.zeros_like(T_ijk)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    term2[i,j,k,:] = delta[i,j]*V_k[k,:] + delta[i,k]*V_k[j,:] + delta[j,k]*V_k[i,:]
        return T_ijk - (1.0/5.0) * term2

    x_vec = _get_minkowski_trajectory(trajectory.r, trajectory.theta, trajectory.phi)
    v_vec = np.gradient(x_vec, dt_code, axis=1, edge_order=2)
    
    I_raw = mu_code * np.einsum('it,jt->ijt', x_vec, x_vec)
    M_raw = mu_code * np.einsum('it,jt,kt->ijkt', x_vec, x_vec, x_vec)
    S_raw = mu_code * np.einsum('it,jt,kt->ijkt', v_vec, x_vec, x_vec)
    
    I_STF = _compute_stf_tensor_2(I_raw)
    M_STF = _compute_stf_tensor_3(M_raw)
    
    epsilon = np.zeros((3,3,3))
    epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1
    epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1
    J_raw = np.einsum('ipq,pqjt->ijt', epsilon, S_raw)
    J_STF = _compute_stf_tensor_2(J_raw)
    
    d1_I = np.gradient(I_STF, dt_code, axis=2, edge_order=2)
    d2_I = np.gradient(d1_I, dt_code, axis=2, edge_order=2)
    d3_I = np.gradient(d2_I, dt_code, axis=2, edge_order=2)
    
    d1_J = np.gradient(J_STF, dt_code, axis=2, edge_order=2)
    d2_J = np.gradient(d1_J, dt_code, axis=2, edge_order=2)
    d3_J = np.gradient(d2_J, dt_code, axis=2, edge_order=2)
    
    d1_M = np.gradient(M_STF, dt_code, axis=3, edge_order=2)
    d2_M = np.gradient(d1_M, dt_code, axis=3, edge_order=2)
    d3_M = np.gradient(d2_M, dt_code, axis=3, edge_order=2)
    d4_M = np.gradient(d3_M, dt_code, axis=3, edge_order=2)
    
    term_I = np.einsum('ijt,ijt->t', d3_I, d3_I)
    term_J = np.einsum('ijt,ijt->t', d3_J, d3_J)
    term_M = np.einsum('ijkt,ijkt->t', d4_M, d4_M)
    flux_E_inst = - (1.0/5.0) * term_I - (16.0/45.0) * term_J - (5.0/189.0) * term_M
    
    I_2, I_3 = d2_I, d3_I
    P_kl = np.einsum('kat,alt->klt', I_2, I_3)
    term_L_I = np.einsum('ikl,klt->it', epsilon, P_kl)
    J_2, J_3 = d2_J, d3_J
    P_J_kl = np.einsum('kat,alt->klt', J_2, J_3)
    term_L_J = np.einsum('ikl,klt->it', epsilon, P_J_kl)
    M_3, M_4 = d3_M, d4_M
    P_M_kl = np.einsum('kabt,labt->klt', M_3, M_4)
    term_L_M = np.einsum('ikl,klt->it', epsilon, P_M_kl)
    
    L_dot_vec = - (2.0/5.0) * term_L_I - (32.0/45.0) * term_L_J - (1.0/63.0) * term_L_M
    flux_Lz_inst = L_dot_vec[2, :]
    
    return NKFluxes(dE_dt=np.mean(flux_E_inst), dLz_dt=np.mean(flux_Lz_inst), dQ_dt=0.0)