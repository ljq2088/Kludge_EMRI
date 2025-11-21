# src/emrikludge/orbits/nk_geodesic_orbit.py
import sys  
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from .nk_mapping import get_conserved_quantities
# 引入通量计算模块
from .nk_fluxes import compute_nk_fluxes

@dataclass
class NKOrbitTrajectory:
    t: np.ndarray
    p: np.ndarray         # [Inspiral] 随时间变化的 p
    e: np.ndarray         # [Inspiral] 随时间变化的 e
    iota: np.ndarray      # [Inspiral] 随时间变化的 iota
    r_over_M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    chi: np.ndarray
    
    @property
    def r(self):
        return self.r_over_M

class BabakNKOrbit:
    def __init__(self, M, a, p, e, iota, mu=0.0,flux_scheme="gg06_2pn"):
        """
        初始化轨道。
        :param M: 主黑洞质量 (物理单位)
        :param a: 无量纲自旋
        :param p: 半通径 (几何单位 p/M)
        :param e: 偏心率
        :param iota: 倾角 (rad)
        :param mu: 小天体质量（物理单位）。如果为 0，则为测地线演化；如果 > 0，则开启辐射反作用。
        """
        self.M_phys = M
        self.M_code = 1.0     # 内部计算强制使用几何单位 M=1
        self.a = a
        self.flux_scheme = flux_scheme
        # 初始轨道参数
        self.p0 = p
        self.e0 = e
        self.iota0 = iota
        
        # 小天体质量（用于计算通量）
        self.mu_phys = mu
        # 只有当 mu > 0 时才开启 Inspiral 演化
        self.do_inspiral = (mu > 0.0)

        # 缓存 Mapping 结果以用于测地线模式
        if not self.do_inspiral:
            self.k = get_conserved_quantities(self.M_code, a, p, e, iota)
        else:
            self.k = None

        # --- 进度监测变量 ---
        self._total_duration = 0.0
        self._last_print_t = 0.0
        self._print_interval = 100.0 # 默认每 100M 打印一次，evolve 中会动态调整
        self._last_t_call = 0.0

    def _get_constants_fast(self, p, e, iota):
        """[Inspiral] 包装 mapping 函数，处理数值敏感性"""
        # 简单的边界保护
        if e < 0: e = 0.0
        # 简单的 LSO (Last Stable Orbit) 保护，防止掉入视界
        # Schwarzschild ISCO = 6, Kerr 取决于自旋，这里用一个宽松下限
        if p < 1.1 + 2.0*e: 
             raise StopIteration("Plunge detected")
             
        try:
            # 计算当前的 E, Lz, Q
            k = get_conserved_quantities(self.M_code, self.a, p, e, iota)
            return k
        except Exception:
            raise StopIteration("Mapping failed (unstable orbit)")
    def _print_progress(self, t, dt=None):
        """辅助函数：打印进度条"""
        if t - self._last_print_t > self._print_interval:
            percent = (t / self._total_duration) * 100.0
            dt_info = f" | dt ~ {dt:.2e}" if dt else ""
            # 增加 dt 显示，方便监控是否卡死
            sys.stdout.write(f"\r[Integrating] t = {t:.1f} / {self._total_duration:.1f} M ({percent:.1f}%){dt_info}")
            sys.stdout.flush()
            self._last_print_t = t

    def _equations_of_motion(self, t, y):
        """[Geodesic] 3维 ODE 系统: [psi, chi, phi]"""
        # 打印进度
        self._print_progress(t)
        psi, chi, phi = y
        
        # 恢复 r (这里 r 是以 M 为单位的)
        r = self.p0 / (1 + self.e0 * np.cos(psi))
        z = self.k.z_minus * (np.cos(chi)**2)
        
        # 这里的 M 必须用 self.M_code (1.0)
        Delta = r**2 - 2*self.M_code*r + self.a**2
        Sigma = r**2 + (self.a**2) * z
        
        E, Lz, Q = self.k.E, self.k.Lz, self.k.Q
        
        # V_phi
        sin2theta = 1 - z
        term1 = Lz / sin2theta
        term2 = self.a * E
        term3 = (self.a / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
        V_phi = term1 - term2 + term3
        
        # V_t
        term1_t = self.a * (Lz - self.a * E * sin2theta)
        term2_t = ((r**2 + self.a**2) / Delta) * (E * (r**2 + self.a**2) - Lz * self.a)
        V_t = term1_t + term2_t
        
        dt_dtau = V_t / Sigma
        
        # dchi/dt
        gamma = E * (((r**2 + self.a**2)**2 / Delta) - self.a**2) - \
                (2 * self.M_code * r * self.a * Lz) / Delta
        
        denominator = gamma + (self.a**2) * E * z
        dchi_dt = np.sqrt(np.abs(self.k.beta * (self.k.z_plus - z))) / denominator
        
        # dpsi/dt
        term_r = (1 - E**2) * (self.k.r_a - r) * (r - self.k.r_p) * (r - self.k.r3) * (r - self.k.r4)
        if term_r < 0.0: term_r = 0.0
        V_r = term_r
        
        denom_psi = 1 + self.e0 * np.cos(psi)
        dr_dpsi = (self.p0 * self.e0 * np.sin(psi)) / (denom_psi**2)
        
        if np.abs(np.sin(psi)) < 1e-5:
            dpsi_dt = np.sqrt(V_r + 1e-14) / (V_t * (np.abs(dr_dpsi) + 1e-7)) * np.sign(dr_dpsi)
            if np.abs(dr_dpsi) < 1e-9:
                 dpsi_dt = 1e-3 / denominator 
        else:
             dpsi_dt = np.sqrt(V_r) / (V_t * dr_dpsi)
        dpsi_dt = np.abs(dpsi_dt)

        dphi_dt = V_phi / V_t
        
        return [dpsi_dt, dchi_dt, dphi_dt]

    def _equations_of_motion_inspiral(self, t, y):
        """
        [Inspiral] 6维 ODE 系统: [p, e, iota, psi, chi, phi]
        """
        current_dt = t - self._last_t_call
        self._last_t_call = t
        
        # 只有当 t 向前推进了才打印 (避免试探步的回退造成困惑)
        if current_dt > 0:
             self._print_progress(t) # 需要修改 _print_progress 接收 dt
        p, e, iota, psi, chi, phi = y
        try:
            k = self._get_constants_fast(p, e, iota)
            
            # 🚨 检查 C++ 是否返回了失败信号 (E=0)
            if k.E == 0.0:
                # 主动抛出异常，打断积分器
                raise StopIteration(f"Mapping failed (E=0) at p={p:.4f}, e={e:.4f}")
                
        except StopIteration:
            # 返回全 0 导数会让积分器停滞，不如直接报错退出或者返回极大值让它缩小步长
            # 但如果是 mapping 失败，通常意味着轨道崩了，直接停比较好
            # 为了让 solve_ivp 优雅退出，通常比较麻烦
            # 这里我们返回一个全 0，但希望外层的 event 能捕捉到
            return np.zeros(6)
        # 1. 获取当前的动力学常数 (E, Lz, Q)
        try:
            k = self._get_constants_fast(p, e, iota)
        except StopIteration:
            return np.zeros(6) # 停止演化

        # 2. 计算测地线导数 (dpsi/dt, dchi/dt, dphi/dt)
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
        
        # V_t
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
        #    使用 0PN 公式作为 baseline
        fluxes = compute_nk_fluxes(p, e, iota, self.a, self.M_phys, self.mu_phys, 
                                   scheme=self.flux_scheme)
        
        # nk_fluxes 返回的是几何单位下的导数 dp/dt (M单位时间)
        dp_dt = fluxes.dp_dt
        de_dt = fluxes.de_dt
        diota_dt = fluxes.diota_dt
        
        return [dp_dt, de_dt, diota_dt, dpsi_dt, dchi_dt, dphi_dt]

    def evolve(self, t_duration, dt):
        """
        统一演化接口。根据 self.do_inspiral 决定调用哪个内核。
        """
        # 初始化进度监测参数
        self._total_duration = t_duration
        self._last_print_t = 0.0
        # 设置打印间隔：总时长的 1%，且不小于 10M
        self._print_interval = max(10.0, t_duration / 100.0)
        
        if not self.do_inspiral:
            return self._evolve_geodesic(t_duration, dt)
        if not self.do_inspiral:
            return self._evolve_geodesic(t_duration, dt)
        
        # Inspiral 演化 (6维 ODE)
        t_span = [0, t_duration]
        y0 = [self.p0, self.e0, self.iota0, 0.0, 0.0, 0.0]
        
        # 定义终止事件：p 太小 (Plunge)
        def plunge_event(t, y):
            p = y[0]
            # 简单的 ISCO 停止条件
            return p - 3.0 
        plunge_event.terminal = True
        def unbound_event_e(t, y):
            e = y[1]
            return 0.999 - e  # 当 e > 0.999 时停止 (即值 < 0 时触发)
        unbound_event_e.terminal = True
        # 稍微放宽精度以提高 Inspiral 速度
        sol = solve_ivp(
            self._equations_of_motion_inspiral, 
            t_span, y0, 
            method='RK45', 
            rtol=1e-7, atol=1e-7,
            events=[plunge_event, unbound_event_e],
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
        
        # 重建坐标 (需要逐点 Mapping 以获取准确的 z_minus 用于 theta 重建)
        r_over_M = np.zeros_like(t_uniform)
        theta = np.zeros_like(t_uniform)
        
        for i in range(len(t_uniform)):
            r_over_M[i] = p_t[i] / (1 + e_t[i] * np.cos(psi_t[i]))
            try:
                k = get_conserved_quantities(1.0, self.a, p_t[i], e_t[i], iota_t[i])
                cos_theta = np.sqrt(k.z_minus) * np.cos(chi_t[i])
                theta[i] = np.arccos(cos_theta)
            except:
                theta[i] = np.pi/2
        
        return NKOrbitTrajectory(t_uniform, p_t, e_t, iota_t, r_over_M, theta, phi_t, psi_t, chi_t)

    def _evolve_geodesic(self, t_duration, dt):
        """[Geodesic] 仅演化相位，p,e,iota 保持不变"""
        t_span = [0, t_duration]
        y0 = [0.0, 0.0, 0.0] # [psi, chi, phi]
        
        sol = solve_ivp(
            self._equations_of_motion, 
            t_span, y0, 
            method='RK45', 
            rtol=1e-9, atol=1e-9,
            dense_output=True
        )
        
        t_uniform = np.arange(0, t_duration, dt)
        y_uniform = sol.sol(t_uniform)
        
        psi = y_uniform[0]
        chi = y_uniform[1]
        phi = y_uniform[2]
        
        r_over_M = self.p0 / (1 + self.e0 * np.cos(psi))
        cos_theta = np.sqrt(self.k.z_minus) * np.cos(chi)
        theta = np.arccos(cos_theta)
        
        # 构造常数数组
        p_arr = np.full_like(t_uniform, self.p0)
        e_arr = np.full_like(t_uniform, self.e0)
        iota_arr = np.full_like(t_uniform, self.iota0)
        
        return NKOrbitTrajectory(t_uniform, p_arr, e_arr, iota_arr, r_over_M, theta, phi, psi, chi)