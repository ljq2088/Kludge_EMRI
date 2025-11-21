import numpy as np
import matplotlib.pyplot as plt
import time
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo
from src.emrikludge.orbits.nk_mapping import get_conserved_quantities

# ==============================================================================
# 用户配置区域 (User Configuration)
# ==============================================================================
# 物理常数 (SI)
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
SEC_PER_YEAR = 31536000.0
# 1. 物理系统参数
M_BH_SOLAR = 1e6        # 主黑洞质量 [太阳质量]
mu_OBJ_SOLAR = 10.0     # 小天体质量 [太阳质量]
SPIN_a = 0.9            # 主黑洞无量纲自旋 (0 < a < 1)

# 2. 初始轨道几何参数
p0_M = 10.0             # 初始半通径 p (单位: M)
e0 = 0.5                # 初始偏心率
iota0_deg = 30.0        # 初始倾角 [度]

# 3. 观测者位置
DIST_GPC = 1.0          # 距离 [Gpc]
THETA_OBS_DEG = 45.0    # 观测者极角 [度]
PHI_OBS_DEG = 0.0       # 观测者方位角 [度]

# 4. 模拟控制
# 模式选择: 'short' (用于调试波形) 或 'long' (用于看长期演化)
SIMULATION_MODE = 'long' 

# 'short' 模式设置
SHORT_DURATION_SEC = 20000.0  # 物理时间 [秒] (约 5.5 小时)
SHORT_DT=1.0                  # 输出采样步长 [秒]
#转换成单位为M
SHORT_DT_M=SHORT_DT * C_SI**3 /(G_SI * M_BH_SOLAR * M_SUN_SI)  # 将秒转换为 M 单位

# 'long' 模式设置
LONG_DURATION_YEARS = 1.0     # 物理时间 [年]
LONG_DT=10.0                  # 输出采样步长 [秒]
#转换成单位为M
LONG_DT_M=SHORT_DT * C_SI**3 /(G_SI * M_BH_SOLAR * M_SUN_SI)  # 将秒转换为 M 单位
# ==============================================================================
# 内部计算逻辑 (无需频繁修改)
# ==============================================================================



def run_simulation():
    print(f"=== EMRI NK Simulation (Mode: {SIMULATION_MODE}) ===")
    
    # -------------------------------------------------------
    # A. 单位换算
    # -------------------------------------------------------
    M_kg = M_BH_SOLAR * M_SUN_SI
    # 时间单位转换因子: T_geom = GM/c^3
    T_unit_sec = G_SI * M_kg / (C_SI**3)
    # 长度单位转换因子: L_unit_m = GM/c^2
    L_unit_m = G_SI * M_kg / (C_SI**2)
    
    print(f"[Units] 1 M (Time) = {T_unit_sec:.4f} s")
    print(f"[Units] 1 M (Length) = {L_unit_m/1000:.4f} km")

    # 确定积分时长 (M)
    if SIMULATION_MODE == 'short':
        duration_M = SHORT_DURATION_SEC / T_unit_sec
        dt_M = SHORT_DT_M
        plot_waveform = True
    else:
        duration_sec = LONG_DURATION_YEARS * SEC_PER_YEAR
        duration_M = duration_sec / T_unit_sec
        dt_M = LONG_DT_M
        plot_waveform = True # 长时间跑波形文件太大，通常只看轨道参数
        
    print(f"[Setup] Duration: {duration_M:.1f} M ({duration_M*T_unit_sec/3600:.2f} hours)")
    print(f"[Setup] Initial Orbit: p={p0_M}, e={e0}, iota={iota0_deg} deg")

    # -------------------------------------------------------
    # B. 执行演化
    # -------------------------------------------------------
    start_time = time.time()
    
    # 初始化轨道器 (默认使用 gg06_2pn flux scheme)
    orbiter = BabakNKOrbit(
        M=M_BH_SOLAR, 
        a=SPIN_a, 
        p=p0_M, 
        e=e0, 
        iota=np.radians(iota0_deg), 
        mu=mu_OBJ_SOLAR,
        flux_scheme="gg06_2pn" 
    )
    
    print("[Compute] Integrating orbit (this may take a while)...")
    # 演化
    traj = orbiter.evolve(duration_M, dt_M)
    
    elapsed = time.time() - start_time
    print(f"[Compute] Done. Steps: {len(traj.t)}. Time elapsed: {elapsed:.2f} s")

    # -------------------------------------------------------
    # C. 结果分析与验证
    # -------------------------------------------------------
    # 转换回物理时间轴
    t_sec = traj.t * T_unit_sec
    t_hours = t_sec / 3600.0
    t_days = t_hours / 24.0
    
    # 验证 Q 的演化
    # 我们需要重新计算首尾的 Q 值
    # 注意：traj.iota 是 rad
    try:
        k_init = get_conserved_quantities(1.0, SPIN_a, traj.p[0], traj.e[0], traj.iota[0])
        k_final = get_conserved_quantities(1.0, SPIN_a, traj.p[-1], traj.e[-1], traj.iota[-1])
        
        print("\n=== Evolution Statistics ===")
        print(f"  p: {traj.p[0]:.4f} -> {traj.p[-1]:.4f} (Delta: {traj.p[-1]-traj.p[0]:.4e})")
        print(f"  e: {traj.e[0]:.4f} -> {traj.e[-1]:.4f} (Delta: {traj.e[-1]-traj.e[0]:.4e})")
        
        iota_deg_i = np.degrees(traj.iota[0])
        iota_deg_f = np.degrees(traj.iota[-1])
        print(f"  iota: {iota_deg_i:.4f} -> {iota_deg_f:.4f} deg (Delta: {iota_deg_f-iota_deg_i:.4e} deg)")
        
        print(f"  Q (spec): {k_init.Q:.4f} -> {k_final.Q:.4f} (Delta: {k_final.Q-k_init.Q:.4e})")
        
        if abs(k_final.Q - k_init.Q) > 1e-8:
            print("  ✅ CHECK: Carter constant Q has evolved.")
        else:
            print("  ⚠️ CHECK: Carter constant Q shows negligible change (expected only for extremely short runs or equatorial orbits).")
            
    except Exception as e:
        print(f"  ⚠️ Warning: Could not validate Q evolution: {e}")

    # -------------------------------------------------------
    # D. 绘图
    # -------------------------------------------------------
    if plot_waveform:
        # 计算波形
        print("[Compute] Generating waveform...")
        dist_m = DIST_GPC * 1e9 * 3.086e16
        obs = ObserverInfo(R=dist_m, theta=np.radians(THETA_OBS_DEG), phi=np.radians(PHI_OBS_DEG))
        h_plus, h_cross = compute_nk_waveform(traj, mu_OBJ_SOLAR, M_BH_SOLAR, obs, dt_M)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # 1. 轨道参数
        axes[0].plot(t_hours, traj.p, label='p(t)')
        ax0r = axes[0].twinx()
        ax0r.plot(t_hours, traj.e, color='orange', label='e(t)', linestyle='--')
        axes[0].set_ylabel('Semi-latus rectum p/M')
        ax0r.set_ylabel('Eccentricity e')
        axes[0].set_title(f"Inspiral Evolution (M={M_BH_SOLAR:.0e}, mu={mu_OBJ_SOLAR})")
        axes[0].grid(True, alpha=0.3)
        
        # 2. 倾角变化 (证明 Q 演化)
        axes[1].plot(t_hours, np.degrees(traj.iota), color='green')
        axes[1].set_ylabel('Inclination (deg)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Inclination Evolution (driven by Q flux)")
        
        # 3. 波形
        axes[2].plot(t_sec, h_plus, lw=0.8) # 波形通常看秒级细节
        axes[2].set_ylabel('Strain h+')
        axes[2].set_xlabel('Time (seconds) [Note: Top plots are in hours]')
        # 为了方便对比，这里把 x 轴标签改一下，或者你可以统一用 hours
        
        plt.tight_layout()
        plt.savefig(f"NK_ShortRun_Test.png", dpi=150)
        print(f"[Output] Saved plot to NK_ShortRun_Test.png")
        
    else:
        # Long run mode: 只画轨道参数演化
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        time_axis = t_days if duration_M * T_unit_sec > 86400 else t_hours
        time_label = 'Days' if duration_M * T_unit_sec > 86400 else 'Hours'
        
        axes[0].plot(time_axis, traj.p)
        axes[0].set_ylabel('p/M')
        axes[0].set_title(f"Long Term Evolution ({LONG_DURATION_YEARS} years)")
        axes[0].grid(True)
        
        axes[1].plot(time_axis, traj.e, color='orange')
        axes[1].set_ylabel('Eccentricity')
        axes[1].grid(True)
        
        axes[2].plot(time_axis, np.degrees(traj.iota), color='green')
        axes[2].set_ylabel('Inclination (deg)')
        axes[2].set_xlabel(f'Time ({time_label})')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"NK_LongRun_Test.png", dpi=150)
        print(f"[Output] Saved plot to NK_LongRun_Test.png")

    plt.close()

if __name__ == "__main__":
    run_simulation()