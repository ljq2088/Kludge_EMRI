import numpy as np
import h5py
import time
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt

# 引入核心模块
from src.emrikludge.orbits.nk_geodesic_orbit import BabakNKOrbit
from src.emrikludge.waveforms.nk_waveform import compute_nk_waveform, ObserverInfo

# 尝试导入 C++ 加速模块
try:
    from src.emrikludge._emrikludge import BabakNKOrbit_CPP
    CPP_AVAILABLE = True
    print("[Setup] C++ Acceleration Kernel Detected. 🚀")
except ImportError:
    CPP_AVAILABLE = False
    print("[Setup] Using Python Kernel (Slow). 🐢")

# -----------------------------------------------------------------------------
# 1. 适配器 (Adapter)
# -----------------------------------------------------------------------------
@dataclass
class TrajChunkAdapter:
    """
    轻量级适配器：只包装当前 Chunk 的数据传给波形函数。
    """
    t: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    # 波形计算其实只需要 r, theta, phi (在 Minkowski 转换中用到)
    # 如果 nk_waveform.py 需要其他字段，可以补上，但在你的实现里似乎只需要坐标
    
    # 为了兼容性，补充其他字段 (可以是空或伪造，如果 compute_nk_waveform 不用它们)
    # 根据 nk_waveform.py 的 get_minkowski_trajectory，只需要 r, theta, phi
    # 但为了安全，我们把所有字段都填上
    p: np.ndarray = None
    e: np.ndarray = None
    iota: np.ndarray = None
    psi: np.ndarray = None
    chi: np.ndarray = None
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
# -----------------------------------------------------------------------------
# 2. 主程序
# -----------------------------------------------------------------------------
def run_chunked_wave_generation():
    print("=== EMRI Chunked Waveform Generation ===")

    # --- A. 参数设置 ---
    M_BH = 1e6        # M_sun
    mu_Obj = 10.0     # M_sun
    a_spin = 0.7
    
    p0, e0, iota0_deg = 10.0, 0.5, 60.0
    iota0 = np.radians(iota0_deg)
    
    # 演化设置
    total_duration_M = 0.5*3600.0*24.0*365.0*C_SI / (G_SI * (M_BH * M_SUN_SI))  # 总时长 (M) -> 根据需要设为 6.4e6 (1年)
    chunk_size_M = 500000.0       # 每次计算 1万 M (内存友好)
    dt_M = 1.0                   # 采样步长 (M)

    # 观测者
    dist_Gpc = 1.0
    dist_m = dist_Gpc * 1e9 * 3.086e16
    obs = ObserverInfo(R=dist_m, theta=np.pi/4, phi=0.0)
    
    # --- B. 初始化 ---
    if CPP_AVAILABLE:
        orbiter = BabakNKOrbit_CPP(M_BH, a_spin, p0, e0, iota0, mu_Obj)
    else:
        orbiter = BabakNKOrbit(M_BH, a_spin, p0, e0, iota0, mu=mu_Obj)

    output_file = "emri_waveform_complete.h5"
    print(f"[Output] Data will be streamed to: {output_file}")
    
    current_t = 0.0
    chunk_idx = 0
    total_steps = 0
    
    start_time = time.time()

    # --- C. 流式计算循环 ---
    with h5py.File(output_file, "w") as f:
        # 1. 创建可扩展数据集 (Resizable Datasets)
        # chunk=True 允许数据分块存储，maxshape=(None,) 允许无限扩展
        dset_t = f.create_dataset("t", (0,), maxshape=(None,), dtype='f8', chunks=True)
        dset_h_plus = f.create_dataset("h_plus", (0,), maxshape=(None,), dtype='f8', chunks=True)
        dset_h_cross = f.create_dataset("h_cross", (0,), maxshape=(None,), dtype='f8', chunks=True)
        
        # 可选：也存轨道参数
        dset_p = f.create_dataset("p", (0,), maxshape=(None,), dtype='f8', chunks=True)
        dset_e = f.create_dataset("e", (0,), maxshape=(None,), dtype='f8', chunks=True)

        print(f"\n[Start] Integrating {total_duration_M:.1e} M in chunks of {chunk_size_M:.1e} M...")

        while current_t < total_duration_M:
            # 决定本次演化时长 (最后一段可能不满 chunk_size)
            this_duration = min(chunk_size_M, total_duration_M - current_t)
            if this_duration <= 1e-5: break

            # --- Step 1: C++ 演化 ---
            # orbiter 内部保持状态，这里只需调用 evolve 继续跑
            # 注意：Python端不打印进度条，以免刷屏，C++端会有输出
            cpp_states = orbiter.evolve(this_duration, dt_M)
            
            n_points = len(cpp_states)
            if n_points == 0:
                print("\n[Stop] Evolution terminated early (Plunge or Error).")
                break
            
            # --- Step 2: 提取数据 (Zero-Copy if possible) ---
            # 只需要波形计算用到的列
            t_chunk = np.array([s.t for s in cpp_states])
            p_chunk = np.array([s.p for s in cpp_states]) # 用于监测
            e_chunk = np.array([s.e for s in cpp_states])
            
            # 构造 Adapter (包含 r, theta, phi 用于波形)
            traj_chunk = TrajChunkAdapter(
                t=t_chunk,
                r=np.array([s.r for s in cpp_states]),
                theta=np.array([s.theta for s in cpp_states]),
                phi=np.array([s.phi for s in cpp_states]),
                # 补充其他字段以防万一
                p=p_chunk, e=e_chunk, iota=np.array([s.iota for s in cpp_states]),
                psi=np.array([s.psi for s in cpp_states]), chi=np.array([s.chi for s in cpp_states])
            )
            
            # --- Step 3: 计算波形 ---
            # 这一步是在 Python 中做的，会消耗内存，但仅限于这一个 Chunk
            if len(traj_chunk.t) < 10:
                print(f"      [Warning] Skipping tiny tail batch (size {len(traj_chunk.t)}).")
                continue
            hp_chunk, hc_chunk = compute_nk_waveform(traj_chunk, mu_Obj, M_BH, obs, dt_M)
            
            # --- Step 4: 写入磁盘 ---
            # 扩展数据集大小
            old_size = dset_t.shape[0]
            new_size = old_size + n_points
            
            dset_t.resize((new_size,))
            dset_h_plus.resize((new_size,))
            dset_h_cross.resize((new_size,))
            dset_p.resize((new_size,))
            dset_e.resize((new_size,))
            
            # 写入数据
            dset_t[old_size:] = t_chunk
            dset_h_plus[old_size:] = hp_chunk
            dset_h_cross[old_size:] = hc_chunk
            dset_p[old_size:] = p_chunk
            dset_e[old_size:] = e_chunk
            
            # --- Step 5: 状态更新 ---
            current_t = t_chunk[-1]
            total_steps += n_points
            chunk_idx += 1
            
            # 打印简报 (覆盖 C++ 的最后一行输出)
            sys.stdout.write(f"\r[Python Chunk {chunk_idx}] Saved {n_points} pts. p={p_chunk[-1]:.4f}, e={e_chunk[-1]:.4f}   ")
            sys.stdout.flush()
            
            # 释放大数组内存 (Python 引用计数会自动回收)
            del cpp_states, traj_chunk, hp_chunk, hc_chunk

    # --- D. 结束 ---
    elapsed = time.time() - start_time
    print(f"\n\n[Done] Simulation finished in {elapsed:.2f} s.")
    print(f"       Total steps: {total_steps}")
    print(f"       File size: ~{total_steps * 5 * 8 / 1024 / 1024:.1f} MB")
    print(f"       Saved to: {output_file}")

    # --- E. 简单验证绘图 (只读取最后一点点) ---
    print("[Plot] Plotting last 1000 points check...")
    with h5py.File(output_file, "r") as f:
        t_last = f["t"][-10000:]
        hp_last = f["h_plus"][-10000:]
        
        plt.figure(figsize=(10, 4))
        plt.plot(t_last, hp_last)
        plt.title("Waveform Tail (Last 1000 points)")
        plt.xlabel("Time (M)")
        plt.ylabel("h+")
        plt.grid(True, alpha=0.3)
        plt.savefig("chunk_wave_check.png")
        print("[Plot] Saved chunk_wave_check.png")

if __name__ == "__main__":
    run_chunked_wave_generation()