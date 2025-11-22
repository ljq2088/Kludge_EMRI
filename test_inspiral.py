import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

# 引入 Python 原版模块
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo

# ------------------------------------------------------------------------------
# C++ 扩展模块集成与适配
# ------------------------------------------------------------------------------
try:
    # 尝试导入 C++ 类和结构体
    from src.emrikludge._emrikludge import BabakNKOrbit_CPP, OrbitState
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("⚠️ Warning: C++ extension (_emrikludge) not found. Acceleration unavailable.")

@dataclass
class CppTrajectoryAdapter:
    """
    适配器：将 C++ 返回的数据结构转换为 Python 代码期望的格式。
    完全模仿 NKOrbitTrajectory 的属性。
    """
    t: np.ndarray
    p: np.ndarray
    e: np.ndarray
    iota: np.ndarray
    r_over_M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    chi: np.ndarray
    
    @property
    def r(self):
        return self.r_over_M

def convert_cpp_results(cpp_states):
    """
    将 C++ 的 vector<OrbitState> 高效转换为 numpy array 封装对象。
    """
    # 获取点数
    n = len(cpp_states)
    if n == 0:
        raise ValueError("C++ evolution returned empty trajectory!")

    # 预分配 numpy 数组 (比列表解析更快)
    t = np.zeros(n)
    p = np.zeros(n)
    e = np.zeros(n)
    iota = np.zeros(n)
    r_over_M = np.zeros(n)
    theta = np.zeros(n)
    phi = np.zeros(n)
    psi = np.zeros(n)
    chi = np.zeros(n)

    # 填充数据
    # 注意：这里假设 OrbitState 绑定了这些字段
    for i, s in enumerate(cpp_states):
        t[i] = s.t
        p[i] = s.p
        e[i] = s.e
        iota[i] = s.iota
        r_over_M[i] = s.r
        theta[i] = s.theta
        phi[i] = s.phi
        psi[i] = s.psi
        chi[i] = s.chi

    return CppTrajectoryAdapter(t, p, e, iota, r_over_M, theta, phi, psi, chi)

# ==========================================
# 0. 物理常数定义 (SI Units)
# ==========================================
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30

def run_inspiral_demo():
    print("=== EMRI NK Simulation Demo ===")

    # ==========================================
    # 1. 初始参数设置 (Initial Parameters)
    # ==========================================
    # 主黑洞参数
    M_phys = 1e6      # 主黑洞质量 [太阳质量]
    a_spin = 0.7      # 无量纲自旋 a/M

    # 小天体参数
    mu_phys = 10.0    # 小天体质量 [太阳质量]
    
    # 初始轨道几何参数
    p_init = 10.0     # 初始半通径 p/M
    e_init = 0.6      # 初始偏心率
    iota_init = np.radians(30.0)  # 初始倾角 (30度)

    # 演化控制
    # 演化时间 (以 M 为单位)。
    # 1 M_sun 的 M 约为 4.92e-6 秒。
    # 1e6 M_sun 的 1 M 约为 4.92 秒。
    T=0.5 #年
    duration_M=T*365.0*24.0*3600.0 / (G_SI * (M_phys * M_SUN_SI) / (C_SI**3))
    dt=2.0 #秒
    dt_M = dt* (C_SI**3) /(G_SI * (M_phys * M_SUN_SI))

    # 观测者设置
    dist_Gpc = 1.0
    dist_meters = dist_Gpc * 1e9 * 3.086e16 
    observer = ObserverInfo(R=dist_meters, theta=np.pi/4, phi=0.0)

    print(f"System: M={M_phys:.1e} M_sun, mu={mu_phys:.1f} M_sun, a={a_spin}")
    print(f"Orbit: p0={p_init}, e0={e_init}, iota0={np.degrees(iota_init):.1f} deg")
    print(f"Evolution: T={duration_M} M, dt={dt_M} M")

    # ==========================================
    # 2. 执行轨道演化 (Inspiral Evolution)
    # ==========================================
    print("\n[1/3] Starting adiabatic inspiral...")
    
    # ⏱️ 计时开始
    start_time = time.time()
    
    traj = None

    # --- 分支逻辑：优先使用 C++ ---
    if CPP_AVAILABLE:
        print(f"      🚀 Using C++ Kernel (BabakNKOrbit_CPP)")
        print(f"      (Progress bar will be printed by C++ stdout below)")
        
        # 初始化 C++ 对象
        # 参数顺序需与 bindings_aak.cpp 中一致: M, a, p, e, iota, mu
        orbiter_cpp = BabakNKOrbit_CPP(M_phys, a_spin, p_init, e_init, iota_init, mu_phys)
        
        # 执行演化 (C++ 内部循环)
        # 注意：这是一个阻塞调用，直到算完才会返回 Python
        cpp_results = orbiter_cpp.evolve(duration_M, dt_M)
        
        # 格式转换
        print(f"      Converting C++ results to Python format...")
        traj = convert_cpp_results(cpp_results)
        
    else:
        print(f"      🐢 Using Python Kernel (BabakNKOrbit)")
        # 初始化 Python 对象
        orbiter = BabakNKOrbit(M_phys, a_spin, p_init, e_init, iota_init, mu=mu_phys)
        # 执行演化
        traj = orbiter.evolve(duration_M, dt_M)
    
    # ⏱️ 计时结束
    elapsed = time.time() - start_time
    
    print(f"      Evolution finished in {elapsed:.2f} seconds.")
    print(f"      Generated {len(traj.t)} steps.")
    print(f"      Final state: p={traj.p[-1]:.4f}, e={traj.e[-1]:.4f}, iota={np.degrees(traj.iota[-1]):.2f} deg")

    # ==========================================
    # 3. 计算波形 (Waveform Generation)
    # ==========================================
    print("\n[2/3] Computing gravitational waveform...")
    # 此时 traj 无论是来自 C++ 还是 Python，结构都是一样的，直接传给波形函数
    h_plus, h_cross = compute_nk_waveform(traj, mu_phys, M_phys, observer, dt_M)
    
    max_h = np.max(np.abs(h_plus))
    print(f"      Max strain amplitude: {max_h:.2e}")

    # ==========================================
    # 4. 绘图与可视化 (Visualization)
    # ==========================================
    print("\n[3/3] Plotting results...")
    
    T_geom_sec = G_SI * (M_phys * M_SUN_SI) / (C_SI**3)
    t_sec = traj.t * T_geom_sec

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 子图 1: 轨道参数演化
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_ylabel('Semi-latus rectum $p/M$', color=color)
    ax1.plot(t_sec, traj.p, color=color, label=r'$p(t)$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(rf"EMRI Inspiral Evolution ($M=10^6 M_\odot, \mu=10 M_\odot, a={a_spin}$)")

    ax1_r = ax1.twinx() 
    color = 'tab:orange'
    ax1_r.set_ylabel('Eccentricity $e$', color=color)
    ax1_r.plot(t_sec, traj.e, color=color, linestyle='--', label=r'$e(t)$')
    ax1_r.tick_params(axis='y', labelcolor=color)

    # 子图 2: 径向运动
    ax2 = axes[1]
    ax2.plot(t_sec, traj.r, 'k-', lw=0.8)
    ax2.set_ylabel('Radial coord $r/M$')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Radial Motion (Zoom-Whirl features)")

    # 子图 3: 波形
    ax3 = axes[2]
    ax3.plot(t_sec, h_plus, 'r-', lw=0.8)
    ax3.set_ylabel('Strain $h_+$')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"Gravitational Waveform (Distance = {dist_Gpc} Gpc)")

    plt.tight_layout()
    filename = "Inspiral_Complete_Test.png"
    plt.savefig(filename, dpi=150)
    print(f"\n[Done] Plot saved to {filename}")
    plt.close()

if __name__ == "__main__":
    run_inspiral_demo()