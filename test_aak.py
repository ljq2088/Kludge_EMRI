import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 导入核心模块
# 注意：如果之前安装在 site-packages，直接 import
# 如果是在本地 build 目录，确保 setup.py build_ext --inplace 后当前目录可见
try:
    from src.emrikludge._emrikludge import (
        BabakAAKOrbit, 
        generate_aak_waveform_cpp, 
        compute_kerr_freqs,
        KerrFreqs,
        AAKState
    )
    print("[Import] C++ extension loaded successfully.")
except ImportError as e:
    print(f"[Error] Failed to import C++ extension: {e}")
    print("Please run: pip install .")
    exit(1)

# ==========================================
# 0. 物理常数与单位转换
# ==========================================
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
PC_SI = 3.085677581e16

def test_true_aak():
    print("\n=== True AAK Pipeline Verification ===")

    # 1. 系统参数设置
    # ------------------------------------------
    M = 1e6       # SMBH mass (Solar masses)
    mu = 10.0     # CO mass (Solar masses)
    a = 0.9       # Spin parameter (dimensionless)
    p0 = 10.0     # Initial semi-latus rectum (M)
    e0 = 0.5      # Initial eccentricity
    iota0 = np.radians(45.0) # Initial inclination (rad)
    
    dist_Gpc = 1.0 # Distance in Gpc

    print(f"[Params] M={M:.1e}, mu={mu}, a={a}")
    print(f"[Orbit] p0={p0}, e0={e0}, iota0={np.degrees(iota0):.1f} deg")

    # 2. 关键检查 I: 精确克尔频率 (Elliptic Integrals)
    # ------------------------------------------
    print("\n[Step 1] Verifying Kerr Fundamental Frequencies...")
    t0 = time.time()
    # 注意：C++ 函数签名 compute_kerr_freqs(M_geom, a, p, e, iota)
    # 这里 M 传入 1.0 (几何单位)，因为 p 也是以 M 为单位
    freqs = compute_kerr_freqs(1.0, a, p0, e0, iota0)
    
    print(f"  Compute time: {(time.time()-t0)*1e6:.2f} us")
    print(f"  Omega_r     = {freqs.Omega_r:.6e} (rad/M)")
    print(f"  Omega_theta = {freqs.Omega_theta:.6e} (rad/M)")
    print(f"  Omega_phi   = {freqs.Omega_phi:.6e} (rad/M)")
    print(f"  Gamma       = {freqs.Gamma:.6f} (dt/dlambda)")
    
    if freqs.Omega_r == 0.0 or np.isnan(freqs.Omega_r):
        print("❌ Error: Frequency calculation returned 0 or NaN. Check Mapping/Elliptic Integrals.")
        return

    # 3. 关键检查 II: AAK 轨道演化 (Flux + Map + Phase)
    # ------------------------------------------
    duration = 50000.0 # M (几何单位时长，约 5000 秒 / 1.4 小时)
    dt_sec=2.0
    M_kg = M * M_SUN_SI
    T_unit_sec = G_SI * M_kg / (C_SI**3)
    dt = dt_sec/T_unit_sec # M (步长可以比 NK 大，因为 AAK 是解析的，只要能捕捉参数变化即可)
    print(f"\n[Info] Time unit: {T_unit_sec:.4f} s (for M={M:.1e} M_sun)")
    print(f"[Info] Evolving for {duration} M (~{duration*T_unit_sec/3600:.2f} hours) with dt={dt:.4f} M (~{dt_sec} s)") 
    print(f"[Info] Total steps: {int(duration/dt)}")
    print(f"[Info] Sampling dt: {dt} M (~{dt_sec} s)")
    t0 = time.time()
    
    # 初始化轨道器：传入物理参数
    # 注意最后一个参数 mu，在 C++ 里它是 mu_phys，但在 flux 计算时需要 mass ratio
    # 我们的 aak_orbit.cpp 实现里 flux 用的 mu/M。
    # 检查 BabakAAKOrbit 构造函数：它接收 mu_phys。
    # 在 evolve 内部调用 compute_gg06_fluxes 时，传入了 mu_phys / M_phys。
    # 所以这里我们传入真实的 mu (10.0)
    
    orbiter = BabakAAKOrbit(M, a, p0, e0, iota0, mu) 
    traj = orbiter.evolve(duration, dt)
    
    print(f"  Evolution finished in {time.time()-t0:.4f} s. Steps: {len(traj)}")
    
    if len(traj) == 0:
        print("❌ Error: Trajectory empty! Evolution failed.")
        return

    # 提取数据
    t_vec = np.array([s.t for s in traj])
    p_vec = np.array([s.p_ak for s in traj])   # 代理参数 p
    e_vec = np.array([s.e_ak for s in traj])   # 代理参数 e
    phir_vec = np.array([s.Phi_r for s in traj])
    phith_vec = np.array([s.Phi_theta for s in traj])
    phiphi_vec = np.array([s.Phi_phi for s in traj])
    
    print(f"  Final p_ak: {p_vec[-1]:.4f} (Delta p: {p_vec[0]-p_vec[-1]:.4e})")
    print(f"  Final Phase Phi_phi: {phiphi_vec[-1]:.2f} rad ({phiphi_vec[-1]/(2*np.pi):.1f} cycles)")

    # 4. 关键检查 III: 波形生成 (Harmonic Summation)
    # ------------------------------------------
    print("\n[Step 3] Generating AAK Waveform...")
    
    # [单位制处理]
    # 我们的 C++ 波形代码期望振幅 H ~ mu/D * (...)
    # 为了得到正确的无量纲应变 h，我们需要确保 mu/D 是无量纲的。
    # 方法 1：mu 和 D 都用几何单位 (M_sun, M_sun) ? 不推荐，D 太大。
    # 方法 2：mu 和 D 都用 SI 单位 (m, m) 或 (kg, kg)。
    # 让我们在 Python 端把 D 转为 "M_sun 质量单位对应的长度"
    
    # 1 M_sun 的几何长度 L_sun = G * M_sun / c^2
    L_sun_meters = G_SI * M_SUN_SI / (C_SI**2) # ~ 1477 meters
    
    # 物理质量 mu (单位: Solar Mass)
    # 物理距离 D (单位: meters)
    D_meters = dist_Gpc * 1e9 * PC_SI
    
    # 传入 C++ 的 mu 和 dist
    # 如果 C++ 内部公式是 H = (mu/dist) * ...
    # 只要 mu 和 dist 的比值是正确的量纲 (Length/Length) 即可。
    # 我们可以传入：
    # mu_in = mu (Solar Masses)
    # dist_in = D_meters / L_sun_meters (Solar Masses)
    
    dist_geom_units = D_meters / L_sun_meters
    
    t0 = time.time()
    h_plus, h_cross = generate_aak_waveform_cpp(
        t_vec, p_vec, e_vec, np.full_like(t_vec, iota0), # 假设 iota 变化很慢，用初始值近似，或去 AAKState 加 iota
        phir_vec, phith_vec, phiphi_vec,
        M, mu, dist_geom_units,
        np.pi/2, 0.0 # Viewing angle (Theta, Phi)
    )
    print(f"  Waveform generated in {time.time()-t0:.4f} s")
    
    # 振幅检查
    max_amp = np.max(np.abs(h_plus))
    print(f"  Max Strain Amplitude: {max_amp:.2e}")
    if max_amp < 1e-30:
        print("⚠️ Warning: Amplitude is suspiciously small. Check units.")
    elif max_amp > 1e-15:
        print("⚠️ Warning: Amplitude is suspiciously large. Check units.")
    else:
        print("✅ Amplitude looks physical (1e-20 ~ 1e-23 range).")

    # 5. 绘图验证
    # ------------------------------------------
    print("\n[Step 4] Plotting results...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 轨道参数演化
    axes[0].plot(t_vec, p_vec, label=r'$p_{AK}$')
    axes[0].set_ylabel('Semi-latus Rectum $p$ (M)')
    axes[0].set_title(f'AAK Trajectory Evolution (M={M:.0e}, a={a})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 频率/相位检查 (画一下 cos(phase) 看是否平滑)
    # Zoom in to first few cycles
    zoom_idx = 500
    if len(t_vec) < zoom_idx: zoom_idx = len(t_vec)
    
    axes[1].plot(t_vec[:zoom_idx], np.cos(phiphi_vec[:zoom_idx]), label=r'$\cos(\Phi_\phi)$')
    axes[1].plot(t_vec[:zoom_idx], np.cos(phir_vec[:zoom_idx]), label=r'$\cos(\Phi_r)$', alpha=0.7)
    axes[1].set_ylabel('Phase Evolution')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Fundamental Phase Evolution (Zoom)')
    axes[1].grid(True, alpha=0.3)
    
    # 波形
    time_arr = t_vec * T_unit_sec
    axes[2].plot(time_arr[:zoom_idx], h_plus[:zoom_idx], label=r'$h_+$', color='black', lw=1)
    axes[2].plot(time_arr[:zoom_idx], h_cross[:zoom_idx], label=r'$h_\times$', color='red', alpha=0.5, lw=1)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Strain')
    axes[2].set_title(f'AAK Waveform (First {zoom_idx} steps)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = "AAK_Test_Result_Final.png"
    plt.savefig(out_file, dpi=150)
    print(f"[Success] Plot saved to {out_file}")
    # plt.show()

if __name__ == "__main__":
    test_true_aak()