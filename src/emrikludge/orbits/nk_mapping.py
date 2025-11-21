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

def get_conserved_quantities(M, a, p, e, iota) -> KerrConstants:
    """
    数值求解 (E, Lz, Q) 给定 (p, e, iota)。
    基于 Babak et al. (2007) 的定义。
    """
    r_p = p / (1.0 + e)
    r_a = p / (1.0 - e)
    
    # 1. 定义残差函数
    def residuals(x):
        E_guess, Lz_guess, Q_guess = x
        
        # 径向根条件
        res1 = _radial_potential(r_p, M, a, E_guess, Lz_guess, Q_guess)
        res2 = _radial_potential(r_a, M, a, E_guess, Lz_guess, Q_guess)
        
        # 倾角条件 (Babak Eq 4: tan^2(iota) = Q / Lz^2)
        # 变形为 Q * cos^2(iota) - Lz^2 * sin^2(iota) = 0 以避免除零
        res3 = Q_guess * (np.cos(iota)**2) - (Lz_guess**2) * (np.sin(iota)**2)
        
        return [res1, res2, res3]

    # 2. 提供初始猜测 (牛顿近似)
    # 在弱场下: E ~ 1, Lz ~ sqrt(p) * cos(i), Q ~ p * sin^2(i)
    x0 = [
        0.95, 
        np.sqrt(p*M) * np.cos(iota), 
        p*M * np.sin(iota)**2
    ]

    # 3. 求解
    sol = root(residuals, x0, method='lm') # Levenberg-Marquardt 通常较稳健
    
    if not sol.success:
        raise ValueError(f"Mapping failed for p={p}, e={e}, iota={iota}: {sol.message}")
    
    E, Lz, Q = sol.x

    # 4. 计算辅助变量 (r3, r4, z_minus, z_plus)
    # 我们利用韦达定理或直接求多项式根来找到 r3, r4
    
    # 径向势 V_r / (1-E^2) = - (r - r_p)(r - r_a)(r - r_3)(r - r_4)
    # V_r 的系数:
    # r^4: (E^2 - 1)
    # r^3: 2M
    # r^2: [a^2(E^2-1) + Lz^2(1-E^2) + (Lz-aE)^2 + Q] ... 比较复杂
    # 最简单的是利用 r^3 系数关系:
    # sum(roots) = - (coeff_r3) / (coeff_r4)
    # sum(roots) = r_p + r_a + r_3 + r_4 = - (2M) / (E^2 - 1)
    # => r_3 + r_4 = 2M / (1 - E^2) - (r_p + r_a)
    
    # 还需要 r_3 * r_4。利用常数项系数关系:
    # r_p * r_a * r_3 * r_4 = - (Coeff_r0) / (E^2 - 1)
    # Coeff_r0 = -a^2 * Q 
    # => r_p * r_a * r_3 * r_4 = (a^2 * Q) / (1 - E^2)
    
    inv_E_factor = 1.0 / (1.0 - E**2)
    sum_r3r4 = 2*M * inv_E_factor - (r_p + r_a)
    prod_r3r4 = (a**2 * Q * inv_E_factor) / (r_p * r_a)
    
    # 解一元二次方程 x^2 - S*x + P = 0 得到 r3, r4
    delta_r = np.sqrt(sum_r3r4**2 - 4*prod_r3r4)
    r3 = (sum_r3r4 - delta_r) / 2.0
    r4 = (sum_r3r4 + delta_r) / 2.0
    # 确保 r3 > r4 (Babak 论文隐含约定，见 Eq 214 附近)
    if r4 > r3: r3, r4 = r4, r3

    # 计算极向根 z_minus, z_plus (Babak Eq 212-213)
    # beta * z^2 - z[...] + Q = 0
    # beta = a^2 * (1 - E^2)
    beta = (a**2) * (1.0 - E**2)
    term_b = Q + (Lz**2) + (a**2)*(1.0 - E**2)
    
    if beta != 0:
        delta_z = np.sqrt(term_b**2 - 4 * beta * Q)
        z_plus = (term_b + delta_z) / (2 * beta)
        z_minus = (term_b - delta_z) / (2 * beta)
    else:
        # Schwarzschild limit or a=0
        z_minus = 0.0
        z_plus = 0.0

    return KerrConstants(E, Lz, Q, r3, r4, z_minus, z_plus, beta)