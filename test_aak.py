import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from src.emrikludge.lisa.response_approx import project_to_lisa_channels
from src.emrikludge.parameters import EMRIParameters
# ==========================================
# 0. 环境检查与导入
# ==========================================
try:
    from src.emrikludge._emrikludge import (
        BabakAAKOrbit, 
        generate_aak_waveform_cpp, 
        compute_kerr_freqs,
        AAKState,
        KerrFreqs
    )
    print("[Init] C++ extension loaded successfully. 🚀")
except ImportError as e:
    print(f"\n[Error] Failed to import C++ module: {e}")
    print("Did you forget to run 'pip install .' ?")
    sys.exit(1)

# ==========================================
# 1. 物理常数与单位转换
# ==========================================
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
PC_SI = 3.085677581e16

def get_length_unit_meters(M_phys_solar):
    """将质量转换为几何长度单位 L = GM/c^2 (meters)"""
    return M_phys_solar * M_SUN_SI * G_SI / (C_SI**2)

def get_time_unit_seconds(M_phys_solar):
    """将质量转换为几何时间单位 T = GM/c^3 (seconds)"""
    return M_phys_solar * M_SUN_SI * G_SI / (C_SI**3)

# ==========================================
# 2. 主测试流程
# ==========================================
def run_true_aak_test():
    print(f"=== AAK Update 9 Verification Pipeline ===")
    
    # --- A. 系统参数 (典型 LISA 源) ---
    M = 1e6      # 主黑洞质量 (Solar Mass)
    mu = 10.0    # 小天体质量 (Solar Mass)
    a = 0.9      # 高自旋
    p0 = 10.0    # 强场区边缘
    e0 = 0.7     # 中等偏心率 (为了看到丰富的谐波)
    iota0 = np.radians(60.0) # 倾角
    
    dist_gpc = 1.0 # 距离 (Gpc)
    
    # --- B. 单位换算 ---
    # 关键：波形振幅公式是 h ~ (mu/D) * ...
    # C++ 代码里 amp_scale = mu / dist
    # 如果 mu 是 10.0 (Solar Mass)，那么 dist 也必须转换为 Solar Mass 单位
    # 这样 mu/dist 才是无量纲的应变
    
    L_M_sun = G_SI * M_SUN_SI / (C_SI**2) # 1 Solar Mass in meters (~1477 m)
    dist_meters = dist_gpc * 1e9 * PC_SI
    dist_in_solar_masses = dist_meters / L_M_sun
    
    # 时间单位 (用于绘图 x 轴)
    T_unit = get_time_unit_seconds(M)
    
    print(f"[Params] M={M:.1e}, mu={mu}, a={a}")
    print(f"[Params] p={p0:.2f}, e={e0:.2f}, i={np.degrees(iota0):.1f} deg")
    print(f"[Units] Distance = {dist_gpc} Gpc = {dist_in_solar_masses:.2e} M_sun units")

    # --- C. 验证频率计算 (Step 1 Check) ---
    print(f"\n[Step 1] Verifying Fundamental Frequencies (Schmidt/GSL)...")
    kf = compute_kerr_freqs(1.0, a, p0, e0, iota0)
    
    if kf.Omega_r == 0.0:
        print("❌ Error: Frequency calculation returned 0. Check Mapping or Parameters.")
        return

    print(f"  Omega_phi   = {kf.Omega_phi:.6f} (rad/M)")
    print(f"  Omega_theta = {kf.Omega_theta:.6f} (rad/M)")
    print(f"  Omega_r     = {kf.Omega_r:.6f} (rad/M)")
    print(f"  Gamma (dt)  = {kf.Gamma:.6f}")
    
    # 物理检查: 进动频率
    perihelion_precess = kf.Omega_theta - kf.Omega_r
    nodal_precess      = kf.Omega_phi - kf.Omega_theta
    print(f"  Perihelion Precession = {perihelion_precess:.6f}")
    print(f"  Lense-Thirring Prec.  = {nodal_precess:.6f}")
    
    if kf.Omega_phi > kf.Omega_theta > kf.Omega_r:
        print("✅ Frequency hierarchy is correct (Phi > Theta > R).")
    else:
        print("⚠️ Warning: Frequency hierarchy unexpected (Check Schmidt formulas).")

    # --- D. 轨道演化 (Step 2 Check) ---
    T=3.0#年
    duration_M = T*365.0*24.0*3600.0 / get_time_unit_seconds(M) # 演化时长 (M)
    dt=5.0 #秒
    dt_M = dt / get_time_unit_seconds(M)        # 采样步长 (AAK 不需要太密)
    
    print(f"\n[Step 2] Evolving AAK Trajectory ({duration_M} M)...")
    start_t = time.time()
    
    # 初始化轨道器 (mu/M 是质量比，但这里构造函数参数名是 mu，需确认 C++ 定义)
    # 查看 bindings_aak.cpp: init(M, a, p, e, iota, mu) -> C++ BabakAAKOrbit
    # C++ 内部 compute_fluxes 需要 mu (Solar) 和 M (Solar) 来计算 ratio?
    # 让我们传入物理质量，让 C++ 处理
    orbiter = BabakAAKOrbit(M, a, p0, e0, iota0, mu)
    
    traj = orbiter.evolve(duration_M, dt_M)
    
    print(f"  Evolution done in {time.time() - start_t:.4f} s. Steps: {len(traj)}")
    
    if len(traj) == 0:
        print("❌ Error: Trajectory is empty!")
        return
        
    # 解包
    t_vec = np.array([s.t for s in traj])
    p_map = np.array([s.p_map for s in traj]) 
    M_map = np.array([s.M_map for s in traj]) # [NEW]
    a_map = np.array([s.a_map for s in traj]) # [NEW]
    e_phys = np.array([s.e for s in traj])
    iota_phys = np.array([s.iota for s in traj])
    
    phir  = np.array([s.Phi_r for s in traj])
    phith = np.array([s.Phi_theta for s in traj])
    phiphi= np.array([s.Phi_phi for s in traj])
    omphi = np.array([s.Omega_phi for s in traj])
    
    print(f"  Final p: {p_map[-1]:.4f} (Delta p = {p_map[0]-p_map[-1]:.4e})")

    # --- E. 波形生成 (Step 3 Check) ---
    print(f"\n[Step 3] Generating Waveform (Peters-Mathews Summation)...")
    start_t = time.time()
    
    # 调用 C++ 波形生成器
    # 注意：传入 dist_in_solar_masses
    h_plus, h_cross = generate_aak_waveform_cpp(
        t_vec, p_map, e_phys, iota_phys, 
        M_map, a_map, # [NEW]
        phir, phith, phiphi,
        omphi,
        M, mu, dist_in_solar_masses,
        np.pi/3, 0.0 
    )
    print(f"  Waveform done in {time.time() - start_t:.4f} s.")
    
    # 检查数值
    max_h = np.max(np.abs(h_plus))
    print(f"  Max Strain: {max_h:.2e}")
    
    if np.isnan(max_h):
        print("❌ Error: Waveform contains NaN! (Check Bessel arguments or Map)")
        return
    if max_h == 0.0:
        print("❌ Error: Waveform is all zeros!")
        return
    print(f"\n[Step 3.5] Applying LISA Response (Python)...")
    class SimpleParams:
        pass
    p_lisa = SimpleParams()
    p_lisa.lambda_S = np.radians(45.0)  # 黄道经度
    p_lisa.beta_S = np.radians(30.0)    # 黄道纬度
    p_lisa.psi_S = 0.5                  # 极化角
    
    # 2. 转换时间为秒 (用于计算轨道位置)
    t_sec = t_vec * T_unit
    
    # 3. 投影
    h_I, h_II = project_to_lisa_channels(t_sec, h_plus, h_cross, params=p_lisa)
    
    print("  LISA response applied.")
    # --- F. 绘图 ---
    print(f"\n[Step 4] Plotting results...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 轨道参数演化
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t_vec, p_map, label=r'$p_{AK}$', color='blue')
    ax1.set_ylabel('Semi-latus rectum $p$ (M)')
    ax1.set_title(f'AAK Evolution ($M=10^6, \\mu=10, a={a}, e_0={e0}$)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 波形全览
    ax2 = fig.add_subplot(3, 1, 2)
    t_sec = t_vec * T_unit
    ax2.plot(t_sec, h_plus, color='black', lw=0.5, alpha=0.8)
    ax2.set_ylabel('Strain $h_+$')
    ax2.set_title('Full Waveform (Amplitude Modulation due to Precession)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 波形细节 (Zoom In) - 验证平滑度和特征
    ax3 = fig.add_subplot(3, 1, 3)
    # 截取中间一段展示“拍”频
    mid = len(t_vec) // 2
    zoom = 200 # 点数
    if len(t_vec) > zoom:
        idx_start = 0 # 从头开始看比较清晰
        idx_end = min(len(t_vec), 500)
        
        ax3.plot(t_vec[idx_start:idx_end], h_plus[idx_start:idx_end], 'r-', lw=1.5, label='$h_+$')
        # 叠加 Cross 验证相位差
        ax3.plot(t_vec[idx_start:idx_end], h_cross[idx_start:idx_end], 'b--', lw=1.0, alpha=0.5, label=r'$h_\times$')
        
        ax3.set_xlabel('Time (M)')
        ax3.set_ylabel('Strain')
        ax3.set_title('Waveform Detail (Smoothness Check)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("AAK_Final_Check.png", dpi=150)
    print("✅ Plot saved to AAK_Final_Check.png")
    #看看h_I,h_II波形
    if len(t_vec) > zoom:
        plt.figure(figsize=(10,4))
        plt.plot(t_vec[idx_start:idx_end], h_I[idx_start:idx_end], 'g-', lw=1.5, label='h_I')
        plt.plot(t_vec[idx_start:idx_end], h_II[idx_start:idx_end], 'm--', lw=1.0, alpha=0.5, label='h_II')
        plt.xlabel('Time (M)')
        plt.ylabel('Strain')
        plt.title('LISA Channels Detail (h_I and h_II)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("AAK_LISA_Channels_Check.png", dpi=150)
        print("✅ Plot saved to AAK_LISA_Channels_Check.png")
    # plt.show()

if __name__ == "__main__":
    run_true_aak_test()