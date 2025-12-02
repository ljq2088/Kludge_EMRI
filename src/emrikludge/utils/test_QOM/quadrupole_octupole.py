import numpy as np
from scipy.interpolate import make_interp_spline
from dataclasses import dataclass

# --- 常数定义 (SI 单位) ---
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30
PC_SI = 3.085677581e16
@dataclass
class ObserverInfo:
    """
    观测者位置信息
    R: 物理距离 (米)
    theta: 极角 (rad)
    phi: 方位角 (rad)
    """
    R: float      
    theta: float
    phi: float

def get_cartesian_trajectory(r, theta, phi):
    """
    将球坐标轨迹转换为笛卡尔坐标
    输入 r, theta, phi 均为 numpy 数组
    """
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=0)

def compute_derivative_spline(t, y, order=1):
    """
    使用五次样条 (k=5) 计算导数，确保高阶导数光滑。
    这是 NK 方法避免数值噪声的关键。
    """
    if len(t) < 6:
        raise ValueError("数据点太少，无法进行五次样条插值 (至少需要6个点)")
    spl = make_interp_spline(t, y, k=5)
    return spl.derivative(order)(t)

def project_to_tt(h_tensor, observer):
    """
    将空间张量 h_ij 投影到 Transverse-Traceless (TT) 规范，得到 h_plus 和 h_cross
    """
    theta = observer.theta
    phi = observer.phi
    ct = np.cos(theta); st = np.sin(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    s2p = np.sin(2*phi); c2p = np.cos(2*phi)
    
    hxx = h_tensor[0,0]; hyy = h_tensor[1,1]; hzz = h_tensor[2,2]
    hxy = h_tensor[0,1]; hxz = h_tensor[0,2]; hyz = h_tensor[1,2]
    
    # 公式源自 Babak et al. 或类似 Kludge 文献
    h_plus = (ct**2)*(hxx*cp**2 + hxy*s2p + hyy*sp**2) + \
             hzz*(st**2) - np.sin(2*theta)*(hxz*cp + hyz*sp) - \
             (hxx*sp**2 - hxy*s2p + hyy*cp**2)
             
    h_cross = 2 * ( ct*(-0.5*hxx*s2p + hxy*c2p + 0.5*hyy*s2p) + \
                    st*(hxz*sp - hyz*cp) )
                    
    return h_plus, h_cross

def calculate_nk_waveform_from_arrays(t_M, r_M, theta, phi, M_phys, mu_phys, observer):
    """
    核心计算函数
    
    输入:
      t_M: 时间数组 (单位: M)
      r_M: 径向距离数组 (单位: M)
      theta: 极角数组 (rad)
      phi: 方位角数组 (rad)
      M_phys: 主黑洞质量 (太阳质量)
      mu_phys: 小天体质量 (太阳质量)
      observer: ObserverInfo 对象 (包含物理距离 R)
      
    输出:
      h_plus, h_cross
    """
    
    # 1. 单位换算与预处理
    # 质量转换为 kg
    M_kg = M_phys * M_SUN_SI
    mu_kg = mu_phys * M_SUN_SI
    
    # 几何长度单位 L_geom = GM/c^2 (米)
    L_geom = G_SI * M_kg / (C_SI**2)
    
    # 观测距离转换为无量纲距离 D_code
    # D_code = R_physical / (GM/c^2)
    D_code = observer.R / L_geom
    
    # 质量比 q = mu / M
    q = mu_kg / M_kg
    
    # 2. 获取笛卡尔坐标 (x, y, z 均为 M 单位)
    x_vec = get_cartesian_trajectory(r_M, theta, phi)
    t_grid = t_M # 时间也是 M 单位
    
    # 3. 构造多极矩张量 (Tensors)
    # 注意：这些张量也是无量纲的
    
    # --- I_ij (Quadrupole) ---
    # I_ij = q * x_i * x_j
    I_tensor = q * np.einsum('it,jt->ijt', x_vec, x_vec)
    
    # --- M_ijk (Mass Octupole) ---
    # M_ijk = q * x_i * x_j * x_k 
    M_tensor = q * np.einsum('it,jt,kt->ijkt', x_vec, x_vec, x_vec)

    # --- S_ijk (Current Octupole) ---
    # S_ijk = q * v_i * x_j * x_k
    # 需要先计算速度 v (对无量纲时间 t_M 求导)
    v_vec = np.zeros_like(x_vec)
    for i in range(3):
        v_vec[i] = compute_derivative_spline(t_grid, x_vec[i], order=1)
        
    S_tensor = q * np.einsum('it,jt,kt->ijkt', v_vec, x_vec, x_vec)
    
    # 4. 对多极矩求导 (使用样条求导)
    # 根据公式: h ~ d2_I + d2_S + d3_M
    
    # I_ij 需要 2 阶导
    d2_I = np.zeros_like(I_tensor)
    for i in range(3):
        for j in range(3):
            d2_I[i, j] = compute_derivative_spline(t_grid, I_tensor[i, j], order=2)

    # S_ijk 需要 2 阶导
    d2_S = np.zeros_like(S_tensor)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d2_S[i, j, k] = compute_derivative_spline(t_grid, S_tensor[i, j, k], order=2)

    # M_ijk 需要 3 阶导
    d3_M = np.zeros_like(M_tensor)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d3_M[i, j, k] = compute_derivative_spline(t_grid, M_tensor[i, j, k], order=3)
    
    # 5. 组装 h_ij 张量
    # 观测方向向量 n
    sin_T = np.sin(observer.theta)
    n_vec = np.array([
        sin_T * np.cos(observer.phi),
        sin_T * np.sin(observer.phi),
        np.cos(observer.theta)
    ])
    
    # 张量缩并: n_i * S_ijk -> S_jk
    n_d2S = np.einsum('i,ijkt->jkt', n_vec, d2_S)
    n_d3M = np.einsum('i,ijkt->jkt', n_vec, d3_M)
    
    # Kludge Waveform Formula (approximate)
    # h = (2/D) * [ I'' - 2 n.S'' + n.M''' ]
    h_tensor = (2.0 / D_code) * (d2_I - 2.0 * n_d2S + n_d3M)
    
    # 6. 投影到观测平面 (h+, hx)
    return project_to_tt(h_tensor, observer)