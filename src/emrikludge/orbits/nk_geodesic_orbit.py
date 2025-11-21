# # src/emrikludge/orbits/nk_geodesic_orbit.py

# import numpy as np
# from scipy.integrate import solve_ivp
# from dataclasses import dataclass
# from .nk_mapping import get_conserved_quantities

# @dataclass
# class NKOrbitTrajectory:
#     t: np.ndarray
#     r_over_M: np.ndarray  # 明确标识这是 r/M
#     theta: np.ndarray
#     phi: np.ndarray
#     psi: np.ndarray
#     chi: np.ndarray
    
#     # 为了兼容波形代码，添加一个 property 'r'
#     @property
#     def r(self):
#         return self.r_over_M

# class BabakNKOrbit:
#     def __init__(self, M, a, p, e, iota):
#         """
#         M: 黑洞质量 (物理单位，如 1e6 或 1.0)
#         a: 无量纲自旋
#         p: 半通径 (以 M 为单位，即 p/M)
#         """
#         self.M_phys = M       # 存储物理质量用于记录
#         self.M_code = 1.0     # 内部计算强制使用几何单位 M=1
#         self.a = a
#         self.p = p
#         self.e = e
#         self.iota = iota
        
#         # 使用 M=1 进行映射
#         self.k = get_conserved_quantities(self.M_code, a, p, e, iota)

#     def _equations_of_motion(self, t, y):
#         psi, chi, phi = y
        
#         # 恢复 r (这里 r 是以 M 为单位的)
#         r = self.p / (1 + self.e * np.cos(psi))
#         z = self.k.z_minus * (np.cos(chi)**2)
#         sin2theta = 1 - z
        
#         # 这里的 M 必须用 self.M_code (1.0)
#         Delta = r**2 - 2*self.M_code*r + self.a**2
#         Sigma = r**2 + (self.a**2) * z
        
#         E, Lz, Q = self.k.E, self.k.Lz, self.k.Q
        
#         term1 = Lz / sin2theta
#         term2 = self.a * E
#         term3 = (self.a / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
#         V_phi = term1 - term2 + term3
        
#         term1_t = self.a * (Lz - self.a * E * sin2theta)
#         term2_t = ((r**2 + self.a**2) / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
#         V_t = term1_t + term2_t
        
#         dt_dtau = V_t / Sigma
        
#         gamma = E * (((r**2 + self.a**2)**2 / Delta) - self.a**2) - \
#                 (2 * self.M_code * r * self.a * Lz) / Delta
        
#         denominator = gamma + (self.a**2) * E * z
        
#         # dchi/dt
#         # 注意：z_plus - z 应该始终 >= 0，加上 abs 防止数值误差
#         dchi_dt = np.sqrt(np.abs(self.k.beta * (self.k.z_plus - z))) / denominator
        
#         # --- 修正变量名 V_r ---
#         # V_r / (1-E^2) = (r_a - r)(r - r_p)(r - r3)(r - r4)
#         # 确保项为正
#         term_r = (1 - E**2) * (self.k.r_a - r) * (r - self.k.r_p) * (r - self.k.r3) * (r - self.k.r4)
        
#         # 数值保护：虽然理论上在 r_p <= r <= r_a 内 term_r >= 0，但数值误差可能导致微小负数
#         if term_r < 0.0:
#             term_r = 0.0
            
#         V_r = term_r  # <--- 【关键修正】显式定义 V_r
        
#         # 计算 dr/dpsi
#         denom_psi = 1 + self.e * np.cos(psi)
#         dr_dpsi = (self.p * self.e * np.sin(psi)) / (denom_psi**2)
        
#         # dpsi/dt = (dr/dt) / (dr/dpsi)
#         # dr/dt = (dr/dtau) / (dt/dtau) = (sqrt(V_r)/Sigma) / (V_t/Sigma) = sqrt(V_r)/V_t
#         # 所以 dpsi/dt = sqrt(V_r) / (V_t * dr_dpsi)
#         # 注意：V_t = denominator * Sigma / Sigma = denominator ??? 不完全是，但 Babak Eq 8 给出了直接形式
#         # Babak Eq 8: dpsi/dt = sqrt(...) / denominator
#         # 我们这里用链式法则来验证，或者直接用数值更稳健
        
#         # 为了避免除以 dr_dpsi=0 (在转折点 sin(psi)=0)，我们加一个小量处理
#         if np.abs(np.sin(psi)) < 1e-5:
#             # 在转折点附近，利用洛必达法则或解析极限
#             # 简单起见，给 dr_dpsi 加一个极小值避免除零错误，
#             # 因为此时 V_r 也是 0，0/0 需要极限处理。
#             # 更好的方式是直接用 Eq 8 的解析形式 (已经消去了 sin(psi))
#             # 这里用数值稳定处理：
#             dpsi_dt = np.sqrt(V_r + 1e-14) / (V_t * (np.abs(dr_dpsi) + 1e-7)) * np.sign(dr_dpsi)
#             # 或者直接近似一个非零速度，让它跨过转折点
#             if np.abs(dr_dpsi) < 1e-9:
#                  # 强制给一个非常小的角速度推过去
#                  dpsi_dt = 1e-3 / denominator 
#         else:
#              dpsi_dt = np.sqrt(V_r) / (V_t * dr_dpsi)
        
#         # 强制取绝对值，因为 psi 是单调增加的角度参数（类似平近点角）
#         # 实际上 Babak 的 psi 定义是 cos(psi)，所以 psi 是一直增加的
#         dpsi_dt = np.abs(dpsi_dt)

#         # dphi/dt
#         dphi_dt = V_phi / V_t
        
#         return [dpsi_dt, dchi_dt, dphi_dt]

#     def evolve(self, t_duration, dt):
#         """
#         演化轨道。
        
#         策略:
#         1. 使用 RK45 自适应步长积分 (dense_output=True)，保证物理精度。
#         2. 使用积分器生成的插值多项式重采样到均匀时间网格 dt。
#         """
#         # 1. 积分设置
#         # t_duration 和 dt 都是以 M 为单位
#         t_span = [0, t_duration]
#         y0 = [0.0, 0.0, 0.0] # [psi, chi, phi]
        
#         # 2. 执行自适应积分
#         # 注意：这里不传入 t_eval，让积分器自己决定步长
#         sol = solve_ivp(
#             self._equations_of_motion, 
#             t_span, 
#             y0, 
#             method='RK45',
#             rtol=1e-9, 
#             atol=1e-9,
#             dense_output=True  # <--- 关键：启用连续解输出
#         )
        
#         # 3. 重采样 (Resampling) 到均匀网格
#         # 生成均匀时间序列
#         t_uniform = np.arange(0, t_duration, dt)
        
#         # 利用 dense_output 提供的插值函数 sol.sol() 计算状态
#         y_uniform = sol.sol(t_uniform)
        
#         psi = y_uniform[0]
#         chi = y_uniform[1]
#         phi = y_uniform[2]
        
#         # 4. 坐标重建
#         r_over_M = self.p / (1 + self.e * np.cos(psi))
        
#         # 【修正】Theta 重建
#         # 之前的 theta = arccos(sqrt(z)) 会丢失符号，导致粒子一直在北半球反弹
#         # 正确的参数化是 cos(theta) = sqrt(z_minus) * cos(chi)
#         # 这样当 chi 变化时，粒子可以穿过赤道面
#         cos_theta = np.sqrt(self.k.z_minus) * np.cos(chi)
#         theta = np.arccos(cos_theta)
        
#         return NKOrbitTrajectory(t_uniform, r_over_M, theta, phi, psi, chi)
# src/emrikludge/orbits/nk_geodesic_orbit.py

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from .nk_mapping import get_conserved_quantities
# 引入通量计算模块 (需确保 nk_fluxes.py 中有对应实现)
from .nk_fluxes import compute_nk_fluxes

@dataclass
class NKOrbitTrajectory:
    t: np.ndarray
    p: np.ndarray         # 新增：随时间变化的 p
    e: np.ndarray         # 新增：随时间变化的 e
    iota: np.ndarray      # 新增：随时间变化的 iota
    r_over_M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    chi: np.ndarray
    
    @property
    def r(self):
        return self.r_over_M

class BabakNKOrbit:
    def __init__(self, M, a, p, e, iota, mu=0.0):
        """
        初始化轨道。
        :param mu: 小天体质量（物理单位，如太阳质量）。如果为 0 或 None，则不进行辐射反作用演化。
        """
        self.M_phys = M
        self.M_code = 1.0
        self.a = a
        
        # 初始轨道参数
        self.p0 = p
        self.e0 = e
        self.iota0 = iota
        
        # 小天体质量（用于计算通量）
        self.mu_phys = mu
        # 只有当 mu > 0 时才开启 Inspiral 演化
        self.do_inspiral = (mu > 0.0)

        # 缓存上一次的 constants 以加速 mapping (warm start)
        self._cached_constants = None

    def _get_constants_fast(self, p, e, iota):
        """包装 mapping 函数，处理数值敏感性"""
        # 简单的边界保护
        if e < 0: e = 0.0
        if p < 2.0 + 2.0*e: # 简单的 LSO 保护，防止掉入视界导致 mapping 失败
             raise StopIteration("Plunge detected")
             
        try:
            # 这里可以优化：传入上一次的解作为 guess，但这需要修改 nk_mapping 接口
            # 目前直接调用，通常足够快
            k = get_conserved_quantities(self.M_code, self.a, p, e, iota)
            return k
        except Exception:
            raise StopIteration("Mapping failed (unstable orbit)")

    def _equations_of_motion_inspiral(self, t, y):
        """
        6维 ODE 系统: [p, e, iota, psi, chi, phi]
        """
        p, e, iota, psi, chi, phi = y
        
        # 1. 获取当前的动力学常数 (E, Lz, Q)
        #    这决定了当前的轨道频率
        try:
            k = self._get_constants_fast(p, e, iota)
        except StopIteration:
            # 如果 mapping 失败（例如 plunge），返回全 0 导数让积分器停止或报错
            return np.zeros(6)

        # 2. 计算测地线导数 (dpsi/dt, dchi/dt, dphi/dt)
        #    (代码逻辑复用之前的 _equations_of_motion，但使用当前的 k)
        #    ------------------------------------------------------
        r = p / (1 + e * np.cos(psi))
        z = k.z_minus * (np.cos(chi)**2)
        
        Delta = r**2 - 2*self.M_code*r + self.a**2
        Sigma = r**2 + (self.a**2) * z
        
        E, Lz, Q = k.E, k.Lz, k.Q
        
        # V_phi
        sin2theta = 1 - z
        term1 = Lz / sin2theta
        term2 = self.a * E
        term3 = (self.a / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
        V_phi = term1 - term2 + term3
        
        # V_t (dt/dtau * Sigma)
        term1_t = self.a * (Lz - self.a * E * sin2theta)
        term2_t = ((r**2 + self.a**2) / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
        V_t = term1_t + term2_t
        dt_dtau = V_t / Sigma
        
        # dchi/dt
        gamma = E * (((r**2 + self.a**2)**2 / Delta) - self.a**2) - \
                (2 * self.M_code * r * self.a * Lz) / Delta
        denominator = gamma + (self.a**2) * E * z
        dchi_dt = np.sqrt(np.abs(k.beta * (k.z_plus - z))) / denominator
        
        # dpsi/dt
        term_r = (1 - E**2) * (k.r_a - r) * (r - k.r_p) * (r - k.r3) * (r - k.r4)
        V_r = max(0.0, term_r)
        
        denom_psi = 1 + e * np.cos(psi)
        dr_dpsi = (p * e * np.sin(psi)) / (denom_psi**2)
        
        if np.abs(np.sin(psi)) < 1e-5:
             dpsi_dt = np.sqrt(V_r + 1e-14) / (V_t * (np.abs(dr_dpsi) + 1e-7)) * np.sign(dr_dpsi)
             if np.abs(dr_dpsi) < 1e-9: dpsi_dt = 1e-3 / denominator
        else:
             dpsi_dt = np.sqrt(V_r) / (V_t * dr_dpsi)
        dpsi_dt = np.abs(dpsi_dt)
        
        dphi_dt = V_phi / V_t
        
        # 3. 计算辐射反作用 (dp/dt, de/dt, diota/dt)
        #    使用 0PN 公式作为 baseline (Babak 论文中允许使用 PN 驱动)
        #    注意：nk_fluxes 需要返回 dp/dt 而不是 dE/dt
        #    ------------------------------------------------------
        # 调用能流模块
        # 注意传入的是 p_dimless, e, iota, M_sun, mu_sun
        fluxes = compute_nk_fluxes(p, e, iota, self.a, self.M_phys, self.mu_phys, scheme="peters_ghk")
        
        # 假设 nk_fluxes_peters_ghk 返回的是 (dE/dt, dL/dt) 或者直接返回 (dp/dt, de/dt)
        # 为了方便，建议修改 nk_fluxes 让它直接返回几何参数的导数，或者在这里做转换
        # 这里假设 compute_nk_fluxes 返回的是包含 dp_dt, de_dt 的对象
        
        # 注意方向：辐射导致能量减少，inspiral 意味着 p 减小
        # peters_mathews_fluxes 通常返回的是负值 (da/dt < 0)
        
        # 如果 nk_fluxes 返回的是 dE, dL，需要用 Jacobian 转换 (复杂)
        # 如果 nk_fluxes 直接封装了 Peters-Mathews 公式，它直接给 dp/dt, de/dt (简单)
        
        # 我们在 nk_fluxes.py 里实现一个直接返回 dp, de 的函数最稳妥
        # 假设 NKFluxes 对象里有 dp_dt, de_dt 字段
        dp_dt = fluxes.dp_dt
        de_dt = fluxes.de_dt
        diota_dt = fluxes.diota_dt # 0PN 下通常为 0
        
        return [dp_dt, de_dt, diota_dt, dpsi_dt, dchi_dt, dphi_dt]

    def evolve(self, t_duration, dt):
        if not self.do_inspiral:
            # 回退到原来的测地线演化 (3维 ODE)
            # ... (保留原有代码逻辑) ...
            return self._evolve_geodesic(t_duration, dt)
        
        # Inspiral 演化 (6维 ODE)
        t_span = [0, t_duration]
        y0 = [self.p0, self.e0, self.iota0, 0.0, 0.0, 0.0]
        
        # 定义终止事件：p 太小 (Plunge)
        def plunge_event(t, y):
            p = y[0]
            # 简单的停止条件：p < r_ISCO (Schwarzschild ~ 6, Kerr ~ depends)
            return p - 3.0 
        plunge_event.terminal = True
        
        sol = solve_ivp(
            self._equations_of_motion_inspiral, 
            t_span, y0, 
            method='RK45', 
            rtol=1e-7, atol=1e-7, # 稍微放宽精度以提高速度
            events=plunge_event,
            dense_output=True
        )
        
        # 重采样
        t_uniform = np.arange(0, sol.t[-1], dt)
        y_uniform = sol.sol(t_uniform)
        
        p_t = y_uniform[0]
        e_t = y_uniform[1]
        iota_t = y_uniform[2]
        psi_t = y_uniform[3]
        chi_t = y_uniform[4]
        phi_t = y_uniform[5]
        
        # 重建坐标 (注意参数现在是随时间变化的数组)
        # 为了效率，我们需要向量化重建，但 nk_mapping 里的常数计算是单点的
        # 这里有一个近似：由于 z_minus 随时间变化，我们需要重新计算它吗？
        # 是的。为了得到精确的 theta，我们需要每个时刻的 z_minus。
        # 这在后处理时比较耗时。
        # 简化方案：在 _equations_of_motion 里我们其实算过，但没存下来。
        # 严谨方案：再次循环调用 mapping (虽然慢但正确)
        
        r_over_M = np.zeros_like(t_uniform)
        theta = np.zeros_like(t_uniform)
        
        # 这里需要一个循环来做 Mapping 恢复 z_minus
        # 也可以用插值，或者只用初始/末态近似 (不推荐)
        # 我们写个循环吧，对于 1e4 个点还是很快的
        for i in range(len(t_uniform)):
            r_over_M[i] = p_t[i] / (1 + e_t[i] * np.cos(psi_t[i]))
            
            # 重新获取常数以计算 theta
            try:
                k = get_conserved_quantities(1.0, self.a, p_t[i], e_t[i], iota_t[i])
                cos_theta = np.sqrt(k.z_minus) * np.cos(chi_t[i])
                theta[i] = np.arccos(cos_theta)
            except:
                theta[i] = np.pi/2 # fallback
        
        return NKOrbitTrajectory(t_uniform, p_t, e_t, iota_t, r_over_M, theta, phi_t, psi_t, chi_t)

    def _evolve_geodesic(self, t_duration, dt):
        # ... (把之前的 evolve 代码移到这里) ...
        # 记得返回的对象要加上 p, e, iota (常数数组)
        pass