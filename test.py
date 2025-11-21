import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 导入我们需要测试的新模块
# 确保你的目录结构正确，并且 __init__.py 允许这些导入
from src.emrikludge.orbits.nk_mapping import get_conserved_quantities
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo

def run_test_case(case_name, M, a, p, e, iota, duration, dt):
    print(f"--- Running Test Case: {case_name} ---")
    print(f"Params: M={M:.1e}, a={a}, p={p}, e={e}, iota={iota}")

    # 1. Mapping: (p, e, iota) -> (E, Lz, Q)
    # 注意：对于 e=0 的情况，数值求解可能会不稳定
    # 工程上的 trick 是给 e 一个极小值，例如 1e-5，或者在 mapping 中做特殊处理
    safe_e = e if e > 1e-5 else 1e-5
    
    try:
        constants = get_conserved_quantities(M, a, p, safe_e, iota)
        print(f"  [Mapping Success] E={constants.E:.6f}, Lz={constants.Lz:.6f}, Q={constants.Q:.6f}")
    except Exception as err:
        print(f"  [Mapping Failed] {err}")
        return

    # 2. Orbit: 积分测地线
    # 注意：我们需要把 constants 传递给 Orbit 对象
    # 之前设计的 BabakNKOrbit 构造函数可能需要调整以接收 constants，
    # 或者我们在外部计算好 constants 后传入。
    # 这里假设 BabakNKOrbit 内部逻辑已调整为接收 constants 或 (p,e,iota) 并在内部调用 mapping
    # 为了演示清晰，假设我们直接传几何参数，并在类内部(或通过工厂方法)处理 mapping
    
    orbiter = BabakNKOrbit(M, a, p, safe_e, iota) 
    # 强行注入计算好的 constants (如果你还没修改 BabakNKOrbit 的 __init__ 来自动调用 mapping)
    # orbiter.m_constants = constants 
    
    # 演化
    print("  [Orbit] Integrating geodesic equations...")
    traj = orbiter.evolve(duration, dt)
    print(f"  [Orbit] Generated {len(traj.t)} steps.")

    # 3. Waveform: 计算波形
    # 设置观测者：距离 D=1Gpc (随便设一个大数), theta=45度, phi=0
    dist = 1.0e9 * 3.086e16 # unit consistency might need check, assuming geometric M? 
    # NK 方法通常输出无量纲应变 h * (D/mu)，我们这里设 D=1 便于看形状
    observer = ObserverInfo(R=1.0, theta=np.pi/4, phi=0.0)
    
    mu = 10.0 # 小天体质量
    
    print("  [Waveform] Computing Quadrupole-Octupole waveform...")
    h_plus, h_cross = compute_nk_waveform(traj, mu, M, observer, dt)

    # 4. Plotting
    plt.figure(figsize=(12, 4))
    # 只画前几百个点或者几个周期，避免太密
    plot_len = min(2000, len(traj.t))
    
    plt.subplot(1, 2, 1)
    plt.plot(traj.t[:plot_len], traj.r[:plot_len])
    plt.title(f"{case_name}: Radial Motion r(t)")
    plt.xlabel("t/M")
    plt.ylabel("r/M")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(traj.t[:plot_len], h_plus[:plot_len], label='h+')
    # plt.plot(traj.t[:plot_len], h_cross[:plot_len], label='hx', alpha=0.5)
    plt.title(f"{case_name}: Waveform h+(t)")
    plt.xlabel("t/M")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{case_name}.png", bbox_inches='tight')

if __name__ == "__main__":
    # 设置黑洞参数
    M_bh = 1e6
    spin = 0.9
    
    # 测试用例 1: 准圆轨道 (e ~ 0)
    # 应该看到径向 r 几乎不变，波形是正弦
    run_test_case(
        case_name="Quasi-Circular",
        M=M_bh, a=spin,
        p=10.0, e=0.0, iota=np.pi/3, # 60度倾角
        duration=2000.0, dt=5.0 # 采样率要根据周期调整
    )

    # 测试用例 2: 偏心轨道 (e = 0.7)
    # 应该看到 r 在 r_p 和 r_a 之间震荡，波形有尖峰
    run_test_case(
        case_name="High-Eccentricity",
        M=M_bh, a=spin,
        p=10.0, e=0.7, iota=np.pi/3,
        duration=4000.0, dt=2.0 # 需要更细的 dt 来捕捉近星点
    )