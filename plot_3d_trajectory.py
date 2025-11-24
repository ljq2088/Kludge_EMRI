import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(h5_file, num_cycles=5):
    """
    读取 HDF5 文件并绘制前 num_cycles 个径向周期的 3D 轨道。
    """
    print(f"Loading trajectory from {h5_file}...")
    
    with h5py.File(h5_file, 'r') as f:
        # 读取必要的坐标数据
        # 注意：根据你的 nk_orbit.cpp 输出，HDF5里应该有这些字段
        # 如果是 test_chunked_wave.py 生成的，通常键名为 't', 'r', 'theta', 'phi', 'psi' 等
        
        if 'r' not in f or 'theta' not in f or 'phi' not in f:
            print("Error: HDF5 file does not contain r/theta/phi coordinates.")
            return

        # 获取时间步长，用于估算点数
        t = f['t'][:]
        psi = f['psi'][:] # 径向相位，用于判断周期
        
        # --- 截取数据 ---
        # 我们希望截取 num_cycles 个径向周期 (psi 从 0 到 num_cycles * 2*pi)
        # 找到结束的索引
        target_psi = num_cycles * 2 * np.pi
        end_idx = np.searchsorted(psi, target_psi)
        
        # 如果数据不够长，就画全部
        if end_idx >= len(psi):
            end_idx = len(psi) - 1
            print(f"Warning: Data shorter than {num_cycles} cycles. Plotting all available data.")
        else:
            print(f"Extracting first {num_cycles} radial cycles (approx {end_idx} points)...")

        # 提取片段
        r_seg = f['r'][:end_idx]
        theta_seg = f['theta'][:end_idx]
        phi_seg = f['phi'][:end_idx]
        
        # 提取中心黑洞参数用于画视界 (可选)
        # 这里假设 M=1 (几何单位)
        M = 1.0 
        # 尝试从文件名或属性读取 a，如果没有则默认 0.9
        a = f.attrs.get('a', 0.9) 

    # --- 坐标转换 (Boyer-Lindquist -> Pseudo-Flat Cartesian) ---
    # NK 方法使用的是 "Bead on a wire" 近似，直接将 BL 坐标映射到平直空间
    x = r_seg * np.sin(theta_seg) * np.cos(phi_seg)
    y = r_seg * np.sin(theta_seg) * np.sin(phi_seg)
    z = r_seg * np.cos(theta_seg)

    # --- 3D 绘图 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制轨道
    # 使用颜色渐变表示时间演化 (从浅蓝到深蓝)
    # ax.plot 只能画单色，可以用 scatter 或 line collection，这里简单起见用 plot
    ax.plot(x, y, z, lw=1.5, color='blue', label='Test Particle Trajectory')
    
    # 标记起点
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
    # 标记终点
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')

    # 2. 绘制中心黑洞 (示意)
    # 画一个半径为 r_plus (视界半径) 的黑色球体
    r_plus = M + np.sqrt(M**2 - a**2)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    bh_x = r_plus * np.cos(u) * np.sin(v)
    bh_y = r_plus * np.sin(u) * np.sin(v)
    bh_z = r_plus * np.cos(v)
    ax.plot_surface(bh_x, bh_y, bh_z, color='black', alpha=0.3)

    # 设置轴标签和比例
    ax.set_xlabel('x (M)')
    ax.set_ylabel('y (M)')
    ax.set_zlabel('z (M)')
    ax.set_title(f'NK Trajectory ({num_cycles} Radial Cycles)')
    
    # 强制坐标轴比例一致 (这样球才是圆的)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.tight_layout()
    
    output_filename = 'orbit_3d_plot.png'
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    # 确保这里的文件名与 test_chunked_wave.py 生成的一致
    file_name = "emri_waveform_complete.h5" 
    
    # 绘制前 5 个径向周期
    # 你可以修改 num_cycles 来看更多或更少
    plot_trajectory(file_name, num_cycles=5)