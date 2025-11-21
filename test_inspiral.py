import numpy as np
import matplotlib.pyplot as plt
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo

# ==========================================
# 0. 物理常数定义 (SI Units)
# ==========================================
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30

def run_inspiral_demo():
    print("=== Running Full EMRI Inspiral Demo ===")

    # ==========================================
    # 1. 初始参数设置 (Initial Parameters)
    # ==========================================
    # 主黑洞参数
    M_phys = 1e6      # 主黑洞质量 [太阳质量]
    a_spin = 0.9      # 无量纲自旋 a/M

    # 小天体参数
    mu_phys = 10.0    # 小天体质量 [太阳质量]
    # 注意：必须设置 mu > 0 才能开启辐射反作用 (Radiation Reaction)

    # 初始轨道几何参数
    p_init = 12.0     # 初始半通径 p/M (几何单位)
                      # 选择 12.0 稍微远一点，以便能看到明显的演化过程
    e_init = 0.5      # 初始偏心率
    iota_init = np.pi / 3  # 初始倾角 (60度)

    # 演化控制
    # 演化时间 (以 M 为单位)。
    # 1e6 M_sun 的 M 对应时间单位约为 5 秒。
    # 20000M 大约是 10^5 秒 (约 1 天的物理时间)。
    duration_M = 20000.0*365 
    dt_M = 2.0        # 采样步长 (M)。对于波形生成，2.0M 足够捕捉低频部分。
                      # 积分器内部会自动使用更小的自适应步长。

    # 观测者设置
    dist_Gpc = 1.0    # 距离 [Gpc]
    dist_meters = dist_Gpc * 1e9 * 3.086e16 # 转换为米
    observer = ObserverInfo(R=dist_meters, theta=np.pi/4, phi=0.0)

    print(f"System: M={M_phys:.1e} M_sun, mu={mu_phys:.1f} M_sun, a={a_spin}")
    print(f"Orbit: p0={p_init}, e0={e_init}, iota0={np.degrees(iota_init):.1f} deg")
    print(f"Evolution: T={duration_M} M, dt={dt_M} M")

    # ==========================================
    # 2. 执行轨道演化 (Inspiral Evolution)
    # ==========================================
    print("\n[1/3] Starting adiabatic inspiral...")
    # 初始化时传入 mu_phys，这会激活 Inspiral 模式 (do_inspiral=True)
    orbiter = BabakNKOrbit(M_phys, a_spin, p_init, e_init, iota_init, mu=mu_phys)
    
    # 开始演化
    # 底层会自动计算 Flux 并更新 p, e
    traj = orbiter.evolve(duration_M, dt_M)
    
    print(f"      Evolution finished. Generated {len(traj.t)} steps.")
    print(f"      Final state: p={traj.p[-1]:.4f}, e={traj.e[-1]:.4f}")

    # ==========================================
    # 3. 计算波形 (Waveform Generation)
    # ==========================================
    print("\n[2/3] Computing gravitational waveform...")
    # compute_nk_waveform 会自动处理物理单位转换
    h_plus, h_cross = compute_nk_waveform(traj, mu_phys, M_phys, observer, dt_M)
    
    # 检查波形量级
    max_h = np.max(np.abs(h_plus))
    print(f"      Max strain amplitude: {max_h:.2e}")

    # ==========================================
    # 4. 绘图与可视化 (Visualization)
    # ==========================================
    print("\n[3/3] Plotting results...")
    
    # 时间单位转换：M -> 秒
    T_geom_sec = G_SI * (M_phys * M_SUN_SI) / (C_SI**3)
    t_sec = traj.t * T_geom_sec

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 子图 1: 轨道参数演化 (p 和 e)
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_ylabel('Semi-latus rectum $p/M$', color=color)
    ax1.plot(t_sec, traj.p, color=color, label=r'$p(t)$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"EMRI Inspiral Evolution ($M=10^6 M_\odot, \mu=10 M_\odot, a={a_spin}$)")

    # 双轴显示偏心率 e
    ax1_r = ax1.twinx() 
    color = 'tab:orange'
    ax1_r.set_ylabel('Eccentricity $e$', color=color)
    ax1_r.plot(t_sec, traj.e, color=color, linestyle='--', label=r'$e(t)$')
    ax1_r.tick_params(axis='y', labelcolor=color)

    # 子图 2: 径向运动 r(t) (Zoom-Whirl 特征)
    ax2 = axes[1]
    # 只画最后一段，看能不能看到 plunge 前的剧烈变化
    ax2.plot(t_sec, traj.r, 'k-', lw=0.8)
    ax2.set_ylabel('Radial coord $r/M$')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Radial Motion (Zoom-Whirl features)")

    # 子图 3: 引力波形 h+(t)
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