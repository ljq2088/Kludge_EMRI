# src/emrikludge/orbits/nk_mapping.py

import numpy as np
from scipy.optimize import root
from dataclasses import dataclass

@dataclass
class KerrConstants:
    E: float
    Lz: float
    Q: float
    r3: float
    r4: float
    r_p: float  # <--- 新增
    r_a: float  # <--- 新增
    z_minus: float
    z_plus: float
    beta: float

def _radial_potential(r, M, a, E, Lz, Q):
    """
    径向势 V_r。
    """
    Delta = r**2 - 2*M*r + a**2
    term1 = (E * (r**2 + a**2) - Lz * a)**2
    term2 = Delta * (r**2 + (Lz - a * E)**2 + Q)
    return term1 - term2

def _radial_potential_deriv(r, M, a, E, Lz, Q):
    """V_r 对 r 的导数，用于圆轨道条件"""
    eps = 1e-5 * r
    v_plus = _radial_potential(r + eps, M, a, E, Lz, Q)
    v_minus = _radial_potential(r - eps, M, a, E, Lz, Q)
    return (v_plus - v_minus) / (2 * eps)
try:
    from cpp.emrikludge import get_conserved_quantities_cpp
    USE_CPP = True
except ImportError:
    USE_CPP = False

def get_conserved_quantities(M, a, p, e, iota)-> KerrConstants:
    if USE_CPP:
        # 调用 C++ 版本 (速度 x100)
        return get_conserved_quantities_cpp(M, a, p, e, iota)
    else:

        """
        计算运动常数。
        内部强制使用 M=1.0 进行归一化计算。
        """
        
        M_code = 1.0
        
        r_p = p / (1.0 + e)
        r_a = p / (1.0 - e)
        
        # 初始猜测 (基于 M=1)
        x0 = [0.93, np.sqrt(p)*np.cos(iota), p*np.sin(iota)**2]

        def residuals(x):
            E, Lz, Q = x
            if e < 1e-4:
                eq1 = _radial_potential(p, M_code, a, E, Lz, Q)
                eq2 = _radial_potential_deriv(p, M_code, a, E, Lz, Q)
            else:
                eq1 = _radial_potential(r_p, M_code, a, E, Lz, Q)
                eq2 = _radial_potential(r_a, M_code, a, E, Lz, Q)
            eq3 = Q * (np.cos(iota)**2) - (Lz**2) * (np.sin(iota)**2)
            return [eq1, eq2, eq3]

        sol = root(residuals, x0, method='lm', options={'maxiter': 2000})
        if not sol.success:
            sol = root(residuals, x0, method='hybr')
            if not sol.success:
                raise ValueError(f"Mapping failed: {sol.message}")

        E, Lz, Q = sol.x
        
        if E < 0.1 or E > 1.5:
            raise ValueError(f"Unphysical Energy derived: E={E:.4f}")

        # 辅助变量计算
        inv_E_factor = 1.0 / (1.0 - E**2)
        sum_r3r4 = 2*M_code * inv_E_factor - (r_p + r_a)
        
        denom_prod = r_p * r_a
        if denom_prod < 1e-8: denom_prod = 1e-8
        prod_r3r4 = (a**2 * Q * inv_E_factor) / denom_prod
        
        delta_r_sq = sum_r3r4**2 - 4*prod_r3r4
        if delta_r_sq < 0: delta_r_sq = 0
        delta_r = np.sqrt(delta_r_sq)
        
        r3 = (sum_r3r4 + delta_r) / 2.0
        r4 = (sum_r3r4 - delta_r) / 2.0
        if r4 > r3: r3, r4 = r4, r3

        beta = (a**2) * (1.0 - E**2)
        term_b = Q + (Lz**2) + (a**2)*(1.0 - E**2)
        
        if abs(beta) > 1e-12:
            delta_z = np.sqrt(term_b**2 - 4 * beta * Q)
            z_plus = (term_b + delta_z) / (2 * beta)
            z_minus = (term_b - delta_z) / (2 * beta)
        else:
            z_minus = 0.0
            z_plus = 0.0

        # 返回完整的 KerrConstants
        return KerrConstants(E, Lz, Q, r3, r4, r_p, r_a, z_minus, z_plus, beta)