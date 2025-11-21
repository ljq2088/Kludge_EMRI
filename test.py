import numpy as np
import matplotlib.pyplot as plt
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo

# 常数
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30

def run_test_case(case_name, M, a, p, e, iota, duration, dt):
    print(f"--- Running {case_name} ---")
    print(f"Input: M={M:.1e} M_sun, p={p} M")
    
    # 1. 轨道积分
    # 这里直接传 M=1e6，底层会自动处理成 M=1 进行积分
    orbiter = BabakNKOrbit(M, a, p, e, iota)
    traj = orbiter.evolve(duration, dt)
    print(f"Orbit steps: {len(traj.t)}")
    
    # 2. 波形计算
    # 传入物理质量，底层负责转换质量比和距离
    dist_meters = 1.0e9 * 3.086e22 # 1 Gpc
    observer = ObserverInfo(R=dist_meters, theta=np.pi/4, phi=0.0)
    mu = 10.0 # 10 M_sun
    
    h_plus, h_cross = compute_nk_waveform(traj, mu, M, observer, dt)
    
    # 3. 绘图 (转换时间轴到秒)
    M_sec = G_SI * (M * M_SUN_SI) / C_SI**3
    t_sec = traj.t * M_sec
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_sec, h_plus)
    plt.title(f"{case_name}: h+ (M={M:.0e})")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{case_name}.png")
    plt.close()

if __name__ == "__main__":
    # 即使 M 输入 1e6，代码现在也能自洽运行
    run_test_case("Quasi-Circular", 1e6, 0.9, 10.0, 1e-5, np.pi/3, 2000.0, 1.0)
    run_test_case("High-Eccentricity", 1e6, 0.9, 10.0, 0.7, np.pi/3, 4000.0, 0.5)