# src/emrikludge/waveforms/nk_waveform.py

import numpy as np
from dataclasses import dataclass

# 常数定义
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_SI = 1.989e30

@dataclass
class ObserverInfo:
    R: float      # 距离 (物理单位，米)
    theta: float
    phi: float

def get_minkowski_trajectory(trajectory):
    # 使用 trajectory.r (即 r/M)
    r = trajectory.r 
    theta = trajectory.theta
    phi = trajectory.phi

    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=0)

def compute_nk_waveform(trajectory, mu_phys, M_phys, observer: ObserverInfo, dt_M):
    """
    计算波形。
    输入:
      trajectory: 轨道 (r, t 均为 M 单位)
      mu_phys: 小天体质量 (太阳质量)
      M_phys: 主黑洞质量 (太阳质量)
      observer.R: 观测距离 (米)
      dt_M: 轨道的时间步长 (M 单位)
    """
    # 1. 单位转换
    # 质量 (kg)
    M_kg = M_phys * M_SUN_SI
    mu_kg = mu_phys * M_SUN_SI
    
    # 长度单位 L_geom = GM/c^2 (米)
    L_geom = G_SI * M_kg / (C_SI**2)
    # 时间单位 T_geom = GM/c^3 (秒)
    T_geom = G_SI * M_kg / (C_SI**3)
    
    # 将观测距离转换为无量纲距离 D_code = R_meters / L_geom
    D_code = observer.R / L_geom
    
    # 将小天体质量转换为质量比 q = mu / M
    q = mu_kg / M_kg
    
    # 2. 准备数据 (全是无量纲的)
    x_vec = get_minkowski_trajectory(trajectory)
    
    # 3. 数值微分 (对无量纲时间 t_code 求导)
    # dt_code = dt_M
    v_vec = np.gradient(x_vec, dt_M, axis=1, edge_order=2)
    
    # 4. 多极矩 (无量纲)
    # I_ij = q * x_i * x_j (注意这里用质量比 q)
    I_tensor = q * np.einsum('it,jt->ijt', x_vec, x_vec)
    
    # S_ijk = v_i * I_jk
    S_tensor = np.einsum('it,jkt->ijkt', v_vec, I_tensor)
    
    # M_ijk = x_i * I_jk
    M_tensor = np.einsum('it,jkt->ijkt', x_vec, I_tensor)
    
    # 5. 高阶导数 (对 t_code)
    d1_I = np.gradient(I_tensor, dt_M, axis=2, edge_order=2)
    d2_I = np.gradient(d1_I, dt_M, axis=2, edge_order=2)
    
    d1_S = np.gradient(S_tensor, dt_M, axis=3, edge_order=2)
    d2_S = np.gradient(d1_S, dt_M, axis=3, edge_order=2)
    
    d1_M = np.gradient(M_tensor, dt_M, axis=3, edge_order=2)
    d2_M = np.gradient(d1_M, dt_M, axis=3, edge_order=2)
    d3_M = np.gradient(d2_M, dt_M, axis=3, edge_order=2)
    
    # 6. 组装 h_ij (无量纲应变)
    # 公式: h = (2/D) * d2I/dt2 ...
    # 这里的 D 是 D_code，t 是 t_code，I 包含 q
    # 结果 h 就是物理应变 (因为 h 本身无量纲)
    
    sin_T = np.sin(observer.theta)
    n_vec = np.array([
        sin_T * np.cos(observer.phi),
        sin_T * np.sin(observer.phi),
        np.cos(observer.theta)
    ])
    
    n_d2S = np.einsum('i,ijkt->jkt', n_vec, d2_S)
    n_d3M = np.einsum('i,ijkt->jkt', n_vec, d3_M)
    
    h_tensor = (2.0 / D_code) * (d2_I - 2.0 * n_d2S + n_d3M)
    
    # 7. 投影
    return _project_to_tt(h_tensor, observer)

def _project_to_tt(h_tensor, observer):
    # ... (保持原有的投影逻辑) ...
    theta = observer.theta
    phi = observer.phi
    ct = np.cos(theta); st = np.sin(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    s2p = np.sin(2*phi); c2p = np.cos(2*phi)
    
    hxx = h_tensor[0,0]; hyy = h_tensor[1,1]; hzz = h_tensor[2,2]
    hxy = h_tensor[0,1]; hxz = h_tensor[0,2]; hyz = h_tensor[1,2]
    
    h_plus = (ct**2)*(hxx*cp**2 + hxy*s2p + hyy*sp**2) + \
             hzz*(st**2) - np.sin(2*theta)*(hxz*cp + hyz*sp) - \
             (hxx*sp**2 - hxy*s2p + hyy*cp**2)
             
    h_cross = 2 * ( ct*(-0.5*hxx*s2p + hxy*c2p + 0.5*hyy*s2p) + \
                    st*(hxz*sp - hyz*cp) )
                    
    return h_plus, h_cross