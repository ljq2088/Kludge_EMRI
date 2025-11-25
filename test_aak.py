import numpy as np
import matplotlib.pyplot as plt
import time
from src.emrikludge._emrikludge import BabakAAKOrbit, generate_aak_waveform_cpp, compute_kerr_freqs

def test_aak_pipeline():
    # ===========================
    # 1. 参数设置
    # ===========================
    M = 1e6      # 太阳质量
    mu = 10.0    # 太阳质量
    a = 0.9      # 自旋
    p0 = 10.0    # 半通径
    e0 = 0.5     # 偏心率
    iota0 = np.radians(45.0) # 倾角
    
    # 几何单位制转换
    G_SI = 6.67430e-11
    C_SI = 299792458.0
    M_SUN_SI = 1.989e30
    M_sec = M * G_SI * M_SUN_SI / C_SI**3
    
    print(f"=== AAK Pipeline Test ===")
    print(f"Params: M={M:.1e}, mu={mu}, a={a}, p={p0}, e={e0}, i={np.degrees(iota0):.1f}")

    # ===========================
    # 2. 频率计算测试 (Step 1 Check)
    # ===========================
    print("\n[Test] Checking Kerr Frequencies...")
    freqs = compute_kerr_freqs(1.0, a, p0, e0, iota0)
    print(f"  Omega_r     = {freqs.Omega_r:.5e} (rad/M)")
    print(f"  Omega_theta = {freqs.Omega_theta:.5e} (rad/M)")
    print(f"  Omega_phi   = {freqs.Omega_phi:.5e} (rad/M)")
    print(f"  Gamma       = {freqs.Gamma:.5f}")
    
    if freqs.Omega_r == 0.0:
        print("❌ Error: Frequency calculation failed (Mapping error?)")
        return

    # ===========================
    # 3. 轨道演化 (Step 2+4 Check)
    # ===========================
    duration = 20000.0 # M (约 2 天)
    dt = 10.0 # M
    
    print(f"\n[Test] Evolving AAK orbit for {duration} M...")
    start_t = time.time()
    
    orbiter = BabakAAKOrbit(M, a, p0, e0, iota0, mu/M)
    traj = orbiter.evolve(duration, dt)
    
    end_t = time.time()
    print(f"  Evolution finished in {end_t - start_t:.4f} s. Steps: {len(traj)}")
    
    if len(traj) == 0:
        print("❌ Error: Trajectory is empty!")
        return

    # 提取数据用于绘图
    t_vec = np.array([s.t for s in traj])
    p_vec = np.array([s.p_ak for s in traj])
    e_vec = np.array([s.e_ak for s in traj])
    phir_vec = np.array([s.Phi_r for s in traj])
    phith_vec = np.array([s.Phi_theta for s in traj])
    phiphi_vec = np.array([s.Phi_phi for s in traj])

    # ===========================
    # 4. 波形生成 (Step 3 Check)
    # ===========================
    print("\n[Test] Generating AAK Waveform...")
    dist_gpc = 1.0
    dist_meters = dist_gpc * 1e9 * 3.086e16
    # 注意：C++ 接收的距离单位需要和内部公式匹配
    # 我们的 aak_waveform.cpp 假设输入就是物理距离(米)或几何距离，内部做了转换
    # 这里传入几何单位距离: dist_meters / (G M_sun / c^2) 
    # 或者简化：aak_waveform.cpp 里 amp_scale = mu / dist. 如果 mu 是 M_sun, dist 应该是 M_sun 单位?
    # 让我们假设 dist 传入的是 Gpc，内部处理。
    # 查看 C++ 代码: double dist_sec = dist; double amp_scale = mu / dist;
    # 这是一个潜在的单位坑。为了简单，我们传入几何单位距离。
    L_unit = G_SI * M_SUN_SI / C_SI**2 # ~1477 m
    dist_geom = dist_meters / L_unit
    
    start_t = time.time()
    # 注意：Python list 需要转为 vector，pybind11 会自动处理 numpy array -> vector
    # 但我们需要把 struct array 拆开 (上面已经拆成 _vec 了)
    
    # C++ 签名: (t, p, e, iota, Phi_r, Phi_th, Phi_phi, M, mu, dist, ...)
    h_plus, h_cross = generate_aak_waveform_cpp(
        t_vec, p_vec, e_vec, np.full_like(t_vec, iota0), # 假设 iota 变化不大或已存
        phir_vec, phith_vec, phiphi_vec,
        M, mu, dist_geom,
        np.pi/2, 0.0 # viewing angles
    )
    print(f"  Waveform generated in {time.time() - start_t:.4f} s")

    # ===========================
    # 5. 绘图验证
    # ===========================
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_vec, p_vec, label='p (AK map)')
    plt.ylabel('p')
    plt.legend()
    plt.title('AAK Parameter Evolution')

    plt.subplot(3, 1, 2)
    plt.plot(t_vec, np.cos(phir_vec), label='cos(Phi_r)')
    plt.plot(t_vec, np.cos(phiphi_vec), label='cos(Phi_phi)', alpha=0.5)
    plt.ylabel('Phase (cos)')
    plt.legend(loc='upper right')
    plt.xlim(0, 2000) # Zoom in

    plt.subplot(3, 1, 3)
    plt.plot(t_vec, h_plus, label='h+')
    # plt.plot(t_vec, h_cross, label='hx')
    plt.ylabel('Strain')
    plt.xlabel('Time (M)')
    plt.legend()
    plt.xlim(0, 2000) # Zoom in to see cycles

    plt.tight_layout()
    plt.savefig("AAK_Test_Result.png")
    print("\n[Success] Plot saved to AAK_Test_Result.png")

if __name__ == "__main__":
    test_aak_pipeline()