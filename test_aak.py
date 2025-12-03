import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

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
    from src.emrikludge.lisa.response_approx import project_to_lisa_channels
    print("[Init] C++ extension loaded successfully. 🚀")
except ImportError as e:
    print(f"\n[Error] Failed to import modules: {e}")
    print("Did you forget to run 'pip install .' ?")
    sys.exit(1)

# ==========================================
# 1. 物理常数与单位转换
# ==========================================
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
PC_SI = 3.085677581e16

def get_time_unit_seconds(M_phys_solar):
    """T = GM/c^3 (seconds)"""
    return M_phys_solar * M_SUN_SI * G_SI / (C_SI**3)

def get_length_unit_meters(M_phys_solar):
    """L = GM/c^2 (meters)"""
    return M_phys_solar * M_SUN_SI * G_SI / (C_SI**2)

# ==========================================
# 2. 主测试流程
# ==========================================
def run_true_aak_test():
    print("\n=== AAK End-to-End Test (Update 1.12) ===")
    
    # --- A. 系统参数 ---
    M = 1.0e6       # MBH (Solar Mass)
    mu = 10.0       # CO (Solar Mass)
    a = 0.9         # Spin
    p0 = 10.0       # Initial p
    e0 = 0.7        # Initial e
    iota0 = np.radians(60.0)
    
    dist_gpc = 1.0
    L_M_sun = G_SI * M_SUN_SI / (C_SI**2)
    dist_in_solar = (dist_gpc * 1e9 * PC_SI) / L_M_sun
    
    T_unit = get_time_unit_seconds(M)
    
    print(f"[Params] M={M:.1e}, mu={mu}, a={a}")
    print(f"[Params] p0={p0:.2f}, e0={e0:.2f}, iota={np.degrees(iota0):.1f} deg")
    
    # --- B. 轨道演化 ---
    # 演化时长 1年
    T_years = 1.0 
    duration_M = T_years * 31536000.0 / T_unit 
    # 采样步长: 10秒 (适当降低采样率以加快测试，且不影响长波波形)
    dt_sample_sec = 5.0 
    dt_M = dt_sample_sec / T_unit 
    
    print(f"\n[Step 1] Evolving AAK Trajectory...")
    print(f"  Duration: {T_years} yr ({duration_M:.1e} M)")
    print(f"  Sampling: {dt_sample_sec} s ({dt_M:.1e} M)")
    
    t0 = time.time()
    
    orbiter = BabakAAKOrbit(M, a, p0, e0, iota0, mu)
    traj = orbiter.evolve(duration_M, dt_M)
    
    print(f"  Done in {time.time()-t0:.2f} s. Steps: {len(traj)}")
    
    if not traj:
        print("❌ Error: Trajectory empty.")
        return

    # 数据解包
    t_vec = np.array([s.t for s in traj])
    p_map = np.array([s.p_map for s in traj])
    M_map = np.array([s.M_map for s in traj])
    a_map = np.array([s.a_map for s in traj])
    e_phys = np.array([s.e for s in traj])
    iota_phys = np.array([s.iota for s in traj])
    
    Phi_r = np.array([s.Phi_r for s in traj])
    Phi_th = np.array([s.Phi_theta for s in traj])
    Phi_phi = np.array([s.Phi_phi for s in traj])
    Om_phi = np.array([s.Omega_phi for s in traj])

    print(f"  Final p: {p_map[-1]:.4f}, Final e: {e_phys[-1]:.4f}")

    # --- C. 波形生成 ---
    print(f"\n[Step 2] Generating Waveform...")
    t0 = time.time()
    
    h_plus, h_cross = generate_aak_waveform_cpp(
        t_vec, p_map, e_phys, iota_phys,
        M_map, a_map,
        Phi_r, Phi_th, Phi_phi, Om_phi,
        M, mu, dist_in_solar,
        np.pi/3, 0.0 
    )
    
    h_plus = np.array(h_plus)
    h_cross = np.array(h_cross)
    print(f"  Done in {time.time()-t0:.2f} s.")

    # --- D. LISA 响应 ---
    print(f"\n[Step 3] Applying LISA Response...")
    class LISAConfig:
        lambda_S = np.radians(45.0)
        beta_S = np.radians(30.0)
        psi_S = 0.5
        
    t_sec = t_vec * T_unit
    h_I, h_II = project_to_lisa_channels(t_sec, h_plus, h_cross, params=LISAConfig())
    print("  Done.")

    # --- E. 数据保存 (New Requirement) ---
    print(f"\n[Step 4] Saving Data...")
    save_path = "aak_data_output.npz"
    np.savez(save_path, 
             t_M=t_vec, t_sec=t_sec,
             p_map=p_map, e_phys=e_phys, i_phys=iota_phys,
             M_map=M_map, a_map=a_map,
             h_plus=h_plus, h_cross=h_cross,
             h_I=h_I, h_II=h_II)
    print(f"✅ Data saved to {save_path}")

    # --- F. 绘图 ---
    print(f"\n[Step 5] Plotting...")
    #绘制完整的波形
    plt.plot(t_sec,h_cross,label='h_cross')
    
    plt.xlabel('Time (M)')
    plt.ylabel('Strain')
    plt.title('AAK Waveform Polarizations')
    plt.legend()
    plt.grid()
    plt.savefig("AAK_complete_waveform.png", dpi=150)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. 轨道演化 (检查抖动)
    axes[0].plot(t_sec, p_map, label=r'$p_{map}$ (Smooth)', color='C0')
    # axes[0].plot(t_vec, M_map/M, label=r'$M_{map}/M_{phys}$', color='C1', alpha=0.6)
    axes[0].set_ylabel('Parameters')
    axes[0].set_title(f'AAK Evolution ($e_0={e0}, a={a}$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. LISA 通量 (Zoom In)
    # 看最后阶段的 Chirp
    idx_start = max(0, len(t_vec) - 2000)
    axes[1].plot(t_sec[idx_start:], h_I[idx_start:], label='h_I', color='g', lw=1)
    axes[1].plot(t_sec[idx_start:], h_II[idx_start:], label='h_II', color='m', lw=1, alpha=0.7)
    axes[1].set_ylabel('Strain (LISA)')
    axes[1].set_title('LISA Channels (Late Inspiral)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 极化波形细节 (Extreme Zoom)
    zoom_slice = slice(idx_start, idx_start + 200)
    axes[2].plot(t_sec[zoom_slice], h_plus[zoom_slice], 'k-', lw=2, label='$h_+$')
    axes[2].plot(t_sec[zoom_slice], h_cross[zoom_slice], 'r--', lw=1.5, label='$h_\\times$')
    axes[2].set_xlabel('Time (M)')
    axes[2].set_ylabel('Source Strain')
    axes[2].set_title('Waveform Detail (Check for Smoothness)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("AAK.png", dpi=150)
    print("✅ Plot saved to AAK.png")

if __name__ == "__main__":
    run_true_aak_test()