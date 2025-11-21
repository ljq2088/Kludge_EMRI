# src/emrikludge/orbits/nk_mapping.py

import numpy as np
from scipy.optimize import root
from dataclasses import dataclass

@dataclass
class KerrConstants:
    """存储计算好的 Kerr 测地线常数和辅助根"""
    E: float
    Lz: float
    Q: float
    r3: float
    r4: float
    z_minus: float
    z_plus: float
    beta: float

def _radial_potential(r, M, a, E, Lz, Q):
    """Babak Eq (2a): V_r(r)"""
    Delta = r**2 - 2*M*r + a**2
    term1 = (E * (r**2 + a**2) - Lz * a)**2
    term2 = Delta * (r**2 + (Lz - a * E)**2 + Q)
    return term1 - term2

def _radial_potential_deriv(r, M, a, E, Lz, Q):
    """
    V_r(r) 对 r 的导数。用于圆轨道条件 V'(r)=0。
    这里使用中心差分近似，精度对于工程实现通常足够。
    """
    eps = 1e-5 * r
    v_plus = _radial_potential(r + eps, M, a, E, Lz, Q)
    v_minus = _radial_potential(r - eps, M, a, E, Lz, Q)
    return (v_plus - v_minus) / (2 * eps)

def get_conserved_quantities(M, a, p, e, iota) -> KerrConstants:
    """
    数值求解 (E, Lz, Q) 给定 (p, e, iota)。
    """
    r_p = p / (1.0 + e)
    r_a = p / (1.0 - e)
    
    # 初始猜测：使用弱场近似
    # E ~ 0.95, Lz ~ sqrt(p)*cos(i), Q ~ p*sin^2(i)
    x0 = [
        0.93, 
        np.sqrt(p * M) * np.cos(iota), 
        p * M * np.sin(iota)**2
    ]

    def residuals(x):
        # 【关键修正】必须在这里解包 x，定义局部的 E, Lz, Q
        E_local, Lz_local, Q_local = x 
        
        # 分支处理：圆轨道 vs 偏心轨道
        if e < 1e-4:
            # 圆轨道条件：V(r) = 0 且 V'(r) = 0，此处 r = p
            eq1 = _radial_potential(p, M, a, E_local, Lz_local, Q_local)
            eq2 = _radial_potential_deriv(p, M, a, E_local, Lz_local, Q_local)
        else:
            # 偏心轨道条件：近星点和远星点势能为 0
            eq1 = _radial_potential(r_p, M, a, E_local, Lz_local, Q_local)
            eq2 = _radial_potential_deriv(r_a, M, a, E_local, Lz_local, Q_local) 
            # 注意：上面这一行是错的！偏心轨道应该是势能为0，而不是导数为0
            # 修正如下：
            eq2 = _radial_potential(r_a, M, a, E_local, Lz_local, Q_local)
            
        # 倾角条件 (Babak Eq 4)
        eq3 = Q_local * (np.cos(iota)**2) - (Lz_local**2) * (np.sin(iota)**2)
        
        return [eq1, eq2, eq3]

    # 使用 'lm' 求解器
    sol = root(residuals, x0, method='lm', options={'maxiter': 2000})
    
    if not sol.success:
        # 如果 LM 失败，尝试 hybr
        sol = root(residuals, x0, method='hybr')
        if not sol.success:
             raise ValueError(f"Mapping failed: {sol.message}")

    E, Lz, Q = sol.x

    # 物理性检查
    if E < 0.1 or E > 1.5:
         raise ValueError(f"Unphysical Energy derived: E={E:.4f}.")

    # --- 计算辅助变量 (r3, r4, z_minus, z_plus) ---
    # 系数计算基于韦达定理
    # V_r / (1-E^2) = - (r - r_p)(r - r_a)(r - r_3)(r - r_4)
    
    inv_E_factor = 1.0 / (1.0 - E**2)
    sum_r3r4 = 2*M * inv_E_factor - (r_p + r_a)
    
    # 注意：这里处理 r_p * r_a 可能为 0 的极端情况（虽然 EMRI 不会发生）
    denom_prod = r_p * r_a
    if denom_prod < 1e-8: denom_prod = 1e-8
    prod_r3r4 = (a**2 * Q * inv_E_factor) / denom_prod
    
    delta_r_sq = sum_r3r4**2 - 4*prod_r3r4
    # 保护负根（数值误差导致）
    if delta_r_sq < 0: delta_r_sq = 0
    
    delta_r = np.sqrt(delta_r_sq)
    r3 = (sum_r3r4 + delta_r) / 2.0
    r4 = (sum_r3r4 - delta_r) / 2.0
    if r4 > r3: r3, r4 = r4, r3

    # 极向根 z_minus, z_plus
    beta = (a**2) * (1.0 - E**2)
    term_b = Q + (Lz**2) + (a**2)*(1.0 - E**2)
    
    if abs(beta) > 1e-12:
        delta_z = np.sqrt(term_b**2 - 4 * beta * Q)
        z_plus = (term_b + delta_z) / (2 * beta)
        z_minus = (term_b - delta_z) / (2 * beta)
    else:
        z_minus = 0.0
        z_plus = 0.0

    return KerrConstants(E, Lz, Q, r3, r4, z_minus, z_plus, beta)