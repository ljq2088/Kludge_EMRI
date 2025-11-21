# src/emrikludge/orbits/nk_geodesic_orbit.py

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from .nk_mapping import get_conserved_quantities

@dataclass
class NKOrbitTrajectory:
    t: np.ndarray
    r_over_M: np.ndarray  # 明确标识这是 r/M
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    chi: np.ndarray
    
    # 为了兼容波形代码，添加一个 property 'r'
    @property
    def r(self):
        return self.r_over_M

class BabakNKOrbit:
    def __init__(self, M, a, p, e, iota):
        """
        M: 黑洞质量 (物理单位，如 1e6 或 1.0)
        a: 无量纲自旋
        p: 半通径 (以 M 为单位，即 p/M)
        """
        self.M_phys = M       # 存储物理质量用于记录
        self.M_code = 1.0     # 内部计算强制使用几何单位 M=1
        self.a = a
        self.p = p
        self.e = e
        self.iota = iota
        
        # 使用 M=1 进行映射
        self.k = get_conserved_quantities(self.M_code, a, p, e, iota)

    def _equations_of_motion(self, t, y):
        psi, chi, phi = y
        
        # 恢复 r (这里 r 是以 M 为单位的)
        r = self.p / (1 + self.e * np.cos(psi))
        z = self.k.z_minus * (np.cos(chi)**2)
        sin2theta = 1 - z
        
        # 这里的 M 必须用 self.M_code (1.0)
        Delta = r**2 - 2*self.M_code*r + self.a**2
        Sigma = r**2 + (self.a**2) * z
        
        E, Lz, Q = self.k.E, self.k.Lz, self.k.Q
        
        term1 = Lz / sin2theta
        term2 = self.a * E
        term3 = (self.a / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
        V_phi = term1 - term2 + term3
        
        term1_t = self.a * (Lz - self.a * E * sin2theta)
        term2_t = ((r**2 + self.a**2) / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
        V_t = term1_t + term2_t
        
        dt_dtau = V_t / Sigma
        
        gamma = E * (((r**2 + self.a**2)**2 / Delta) - self.a**2) - \
                (2 * self.M_code * r * self.a * Lz) / Delta
        
        denominator = gamma + (self.a**2) * E * z
        
        # dchi/dt
        dchi_dt = np.sqrt(self.k.beta * (self.k.z_plus - z)) / denominator
        
        # dpsi/dt (使用 Eq 8 的解析形式或链式法则)
        # 此处演示使用 Babak Eq 8 的完整形式 (需要 r3, r4)
        # V_r = (1-E^2)(r_a - r)(r - r_p)(r - r_3)(r - r_4)
        # sqrt(V_r) = sqrt(1-E^2) * sqrt((r_a-r)(r-r_p)(r-r3)(r-r4))
        # 注意：r3, r4 可能为负或复数？对于束缚轨道通常 r3, r4 > r_a > r_p
        # 我们之前 mapping 保证了 r3 > r4
        
        term_r = (1 - E**2) * (self.k.r_a - r) * (r - self.k.r_p) * (r - self.k.r3) * (r - self.k.r4)
        if term_r < 0: term_r = 0
        sqrt_Vr = np.sqrt(term_r)
        
        # dpsi/dt = +/- sqrt(V_r) / denominator * (dpsi/dr)? 
        # 直接用 Babak Eq 8:
        # dpsi/dt = sqrt(V_r) / denominator / (dr/dpsi) ??? 
        # 不，Eq 8 直接给出了 dpsi/dt。
        # 公式 8 右边是 sqrt(...) / denominator
        # 让我们仔细看公式8: dpsi/dtau = ... / Sigma? No, Eq 8 is dpsi/dt directly.
        # 实际上 Eq 8 的分子对应 sqrt(V_r) 对 psi 的转化。
        # r = p / (1 + e cos psi)
        # dr = p e sin psi / (1 + e cos psi)^2 dpsi
        # ... 
        # 无论如何，为了稳健，我们用数值 dr/dt / (dr/dpsi)
        dr_dt = sqrt_Vr / denominator / dt_dtau # dr/dt = (dr/dtau) / (dt/dtau)
        # Wait, dt_dtau is already V_t/Sigma.
        # dr/dtau = sqrt(V_r)/Sigma.
        # So dr/dt = sqrt(V_r) / V_t.
        
        dr_dpsi = (self.p * self.e * np.sin(psi)) / ((1 + self.e * np.cos(psi))**2)
        
        if np.abs(np.sin(psi)) < 1e-6:
            # 在转折点，使用极限值或者近似
             # 这是一个简化的处理，实际上应该展开
             dpsi_dt = np.sqrt(1-E**2) * (self.k.r_a - self.k.r_p) / denominator # 这是一个非常粗糙的占位
             # 更好的做法是直接积分 tau，但这里为了配合 t 积分
             # 我们可以简单地加上一个小量避免除零
             dpsi_dt = (np.sqrt(V_r + 1e-12) / Sigma) / dt_dtau / (dr_dpsi + 1e-9)
        else:
             dpsi_dt = (np.sqrt(V_r) / Sigma) / dt_dtau / dr_dpsi
        
        # dphi/dt
        dphi_dt = V_phi / V_t
        
        return [dpsi_dt, dchi_dt, dphi_dt]

    def evolve(self, t_duration, dt):
        # t_duration 和 dt 都是以 M 为单位的
        t_eval = np.arange(0, t_duration, dt)
        y0 = [0.0, 0.0, 0.0]
        
        sol = solve_ivp(self._equations_of_motion, [0, t_duration], y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
        
        psi = sol.y[0]
        chi = sol.y[1]
        phi = sol.y[2]
        
        r_over_M = self.p / (1 + self.e * np.cos(psi))
        z = self.k.z_minus * (np.cos(chi)**2)
        theta = np.arccos(np.sqrt(z)) # 简化处理，假定在北半球
        
        return NKOrbitTrajectory(sol.t, r_over_M, theta, phi, psi, chi)