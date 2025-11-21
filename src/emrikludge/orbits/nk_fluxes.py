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
def _calc_jacobian_analytical(M, a, p, e, iota, E, Lz, Q):
    """
    使用隐式微分计算 Jacobian d(E,L,Q)/d(p,e,iota)。
    替代昂贵的数值差分。
    
    方程组 F(E, L, Q; p, e, iota) = 0:
    1. V_r(r_p) = 0
    2. V_r(r_a) = 0
    3. Q * cos^2(iota) - Lz^2 * sin^2(iota) = 0
    """
    # 辅助变量
    r_p = p / (1.0 + e)
    r_a = p / (1.0 - e)
    
    # V_r 对参数的偏导数
    # V_r = [E(r^2+a^2) - Lz*a]^2 - Delta * [r^2 + (Lz-aE)^2 + Q]
    # 我们需要 dV/dE, dV/dL, dV/dQ, dV/dr
    
    def get_V_derivs(r):
        Delta = r**2 - 2*M*r + a**2
        # dDelta/dr = 2r - 2M
        
        # Term 1 = X^2, X = E(r^2+a^2) - Lz*a
        X = E * (r**2 + a**2) - Lz * a
        # Term 2 = Delta * Y, Y = r^2 + (Lz-aE)^2 + Q
        Y = r**2 + (Lz - a*E)**2 + Q
        
        # --- 对守恒量 (E, L, Q) 的偏导 ---
        dX_dE = r**2 + a**2
        dX_dL = -a
        # dX_dQ = 0
        
        dY_dE = 2*(Lz - a*E) * (-a)
        dY_dL = 2*(Lz - a*E)
        dY_dQ = 1.0
        
        dV_dE = 2*X*dX_dE - Delta*dY_dE
        dV_dL = 2*X*dX_dL - Delta*dY_dL
        dV_dQ = 0         - Delta*dY_dQ
        
        # --- 对几何参数 (r) 的偏导 ---
        # 注意：我们只需要 dV/dr，因为 dr/dp 和 dr/de 是已知的
        # 在转折点 V=0，但 dV/dr 一般不为 0 (除非圆轨道)
        # 这里的导数比较繁琐，但为了速度是值得的
        dDelta_dr = 2*r - 2*M
        dX_dr = E * 2*r
        dY_dr = 2*r
        
        dV_dr = 2*X*dX_dr - (dDelta_dr*Y + Delta*dY_dr)
        
        return dV_dE, dV_dL, dV_dQ, dV_dr

    # 1. 计算近星点和远星点的导数
    Vp_E, Vp_L, Vp_Q, Vp_r = get_V_derivs(r_p)
    Va_E, Va_L, Va_Q, Va_r = get_V_derivs(r_a)
    
    # 2. 极向方程 G = Q*c2 - L^2*s2 = 0
    cos2 = np.cos(iota)**2
    sin2 = np.sin(iota)**2
    # dG/dE = 0
    dG_dE = 0.0
    # dG/dL = -2*L*s2
    dG_dL = -2 * Lz * sin2
    # dG/dQ = c2
    dG_dQ = cos2
    
    # 3. 构建左边矩阵 A = dF/d(E,L,Q)
    # Row 1: dV(rp)/d...
    # Row 2: dV(ra)/d...
    # Row 3: dG/d...
    A = np.array([
        [Vp_E, Vp_L, Vp_Q],
        [Va_E, Va_L, Va_Q],
        [dG_dE, dG_dL, dG_dQ]
    ])
    
    # 4. 构建右边矩阵 B = - dF/d(p, e, iota)
    # 我们需要 drp/dp, drp/de, dra/dp, dra/de
    drp_dp = 1.0 / (1.0 + e)
    drp_de = -p / (1.0 + e)**2
    
    dra_dp = 1.0 / (1.0 - e)
    dra_de = p / (1.0 - e)**2
    
    # dG/diota = Q*(-2sc) - L^2*(2sc) = -2*sin(i)*cos(i)*(Q + L^2)
    dG_diota = -2 * np.sin(iota) * np.cos(iota) * (Q + Lz**2)
    
    # Row 1 (Vp): - [dV/dr * dr/dp, dV/dr * dr/de, 0]
    row1 = - np.array([Vp_r * drp_dp, Vp_r * drp_de, 0.0])
    
    # Row 2 (Va): - [dV/dr * dra/dp, dV/dr * dra/de, 0]
    row2 = - np.array([Va_r * dra_dp, Va_r * dra_de, 0.0])
    
    # Row 3 (G):  - [0, 0, dG/diota]
    row3 = - np.array([0.0, 0.0, dG_diota])
    
    B = np.stack([row1, row2, row3])
    
    # 5. 求解 J = A^-1 * B
    # A * J = B  => J = solve(A, B)
    try:
        Jac = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        # 奇异矩阵 (例如 e=0 时 r_p=r_a，A的前两行相同)
        # 在这里我们简单置零或回退到数值差分（但通常外层有 safe_e 保护）
        Jac = np.zeros((3,3))
        
    return Jac
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
    # dp = 1e-4 * p
    # de_step = 1e-5 if safe_e > 1e-4 else 1e-6
    # di = 1e-5
    
    # # 基准点
    # k0 = get_conserved_quantities(M_code, a_spin, p, safe_e, iota)
    # y0 = np.array([k0.E, k0.Lz, k0.Q])
    
    # # 对 p 扰动
    # kp = get_conserved_quantities(M_code, a_spin, p+dp, safe_e, iota)
    # km = get_conserved_quantities(M_code, a_spin, p-dp, safe_e, iota)
    # dydp = (np.array([kp.E, kp.Lz, kp.Q]) - np.array([km.E, km.Lz, km.Q])) / (2*dp)
    
    # # 对 e 扰动 (注意边界)
    # e_plus = min(safe_e + de_step, 0.999)
    # e_minus = max(safe_e - de_step, 0.0)
    # ke_p = get_conserved_quantities(M_code, a_spin, p, e_plus, iota)
    # ke_m = get_conserved_quantities(M_code, a_spin, p, e_minus, iota)
    # dyde = (np.array([ke_p.E, ke_p.Lz, ke_p.Q]) - np.array([ke_m.E, ke_m.Lz, ke_m.Q])) / (e_plus - e_minus)
    
    # # 对 iota 扰动
    # ki_p = get_conserved_quantities(M_code, a_spin, p, safe_e, iota+di)
    # ki_m = get_conserved_quantities(M_code, a_spin, p, safe_e, iota-di)
    # dydi = (np.array([ki_p.E, ki_p.Lz, ki_p.Q]) - np.array([ki_m.E, ki_m.Lz, ki_m.Q])) / (2*di)
    
    # J = np.column_stack((dydp, dyde, dydi))
    
    # Y_dot = np.array([dE_spec, dL_spec, dQ_spec])
    
    # try:
    #     X_dot = np.linalg.solve(J, Y_dot)
    #     dp_dt_val, de_dt_val, diota_dt_val = X_dot
    # except np.linalg.LinAlgError:
    #     dp_dt_val, de_dt_val, diota_dt_val = 0.0, 0.0, 0.0
    try:
        k_current = get_conserved_quantities(M_code, a_spin, p, safe_e, iota)
        
        # 使用解析雅可比
        J_matrix = _calc_jacobian_analytical(
            M_code, a_spin, p, safe_e, iota, 
            k_current.E, k_current.Lz, k_current.Q
        )
        
        # J_matrix 实际上是 d(E,L,Q)/d(p,e,iota)
        # 我们需要解 [dE, dL, dQ]^T = J * [dp, de, di]^T
        # 所以 [dp, de, di]^T = J^-1 * [dE, dL, dQ]^T
        # 等等，_calc_jacobian_analytical 返回的是 J = d(E,L,Q)/d(p,e,i) ?
        # 不，我在函数里写的是 Jac = solve(A, B)，其中 A*dY + B*dX = 0
        # 所以 A * (dY/dX) = -B  => dY/dX = - A^-1 B
        # 是的，函数返回的就是 d(Conserved)/d(Geom)。
        
        Y_dot = np.array([dE_spec, dL_spec, dQ_spec])
        
        # 解线性方程: J * X_dot = Y_dot
        X_dot = np.linalg.solve(J_matrix, Y_dot)
        
        dp_dt_val, de_dt_val, diota_dt_val = X_dot
        
    except Exception:
        # 如果解析计算失败 (奇异性)，回退到安全值
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