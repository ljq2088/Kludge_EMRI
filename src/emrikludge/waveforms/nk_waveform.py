# # src/emrikludge/waveforms/nk_waveform.py
# """
# Numerical Kludge (NK) 波形模块 —— 目前实现：纯质量四极矩版本。

# 输入：NKOrbitTrajectory（近似 Kerr 轨道，多频相位）
# 输出：源坐标系下的 h_+(t), h_×(t)。

# 思路：
# - 在“伪平直”笛卡尔坐标系中构造粒子轨道 x(t), y(t), z(t)
# - 质量四极矩：I_ij = μ x_i x_j
# - 数值二阶导数：d²I_ij/dt²
# - 四极公式（几何单位）：h_ij = (2 / D_L) d²I_ij/dt²
# - 对给定观测方向 n，构造极化基 (p,q,n)，再投影得到 h_+, h_×.
# """

# from dataclasses import dataclass
# from typing import Optional, Tuple

# import numpy as np

# from ..parameters import EMRIParameters, WaveformConfig
# from ..orbits.nk_geodesic_orbit import NKOrbitTrajectory
# from ..parameters import NKParameters

# @dataclass
# class NKPolarizations:
#     """NK 波形的极化结果容器。"""
#     t: np.ndarray
#     h_plus: np.ndarray
#     h_cross: np.ndarray


# # ========= 工具函数：数值二阶导 =========

# def _second_time_derivative(y: np.ndarray, dt: float) -> np.ndarray:
#     """
#     对一维时间序列 y(t) 做二阶中心差分，返回 d²y/dt²。
#     """
#     d2y = np.zeros_like(y)
#     d2y[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dt * dt)
#     if y.size > 1:
#         d2y[0] = d2y[1]
#         d2y[-1] = d2y[-2]
#     return d2y


# # ========= 工具函数：由观测方向 n 构造极化基 (p, q, n) =========

# def _orthonormal_basis_from_n(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     给定观测方向 n（3 维向量），构造一组正交归一基 (p, q, n)。
#     """
#     n = np.asarray(n, dtype=float)
#     n = n / np.linalg.norm(n)

#     if abs(n[2]) < 0.9:
#         tmp = np.array([0.0, 0.0, 1.0])
#     else:
#         tmp = np.array([1.0, 0.0, 0.0])

#     p = np.cross(tmp, n)
#     p_norm = np.linalg.norm(p)
#     if p_norm < 1e-14:
#         p = np.array([1.0, 0.0, 0.0])
#     else:
#         p = p / p_norm

#     q = np.cross(n, p)

#     return p, q, n


# # ========= 核心：从 NK 轨道生成 h_+, h_× =========

# def generate_nk_polarizations(
#     traj: NKOrbitTrajectory,
#     params: NKParameters,
#     config: WaveformConfig,
#     D_L: float = 1.0,
#     n_obs: Optional[np.ndarray] = None,
# ) -> NKPolarizations:
#     """
#     生成 NK 波形 (h_+, h_×)。

#     - multipole_order = 'quad' 时：只计算质量四极矩
#     - multipole_order = 'quad+oct' 时：额外加入质量八极 + 电流四极
#       （具体公式见 Barack & Cutler 2004, Appendix; Babak et al. 2007）
#     """
#     if params.multipole_order == "quad":
#         t = traj.t
#         dt = config.dt

#         # 1. 粒子轨道的笛卡尔坐标 (x, y, z)
#         r = traj.r_over_M
#         theta = traj.theta
#         phi = traj.phi

#         sin_th = np.sin(theta)
#         cos_th = np.cos(theta)
#         cos_ph = np.cos(phi)
#         sin_ph = np.sin(phi)

#         x = r * sin_th * cos_ph
#         y = r * sin_th * sin_ph
#         z = r * cos_th

#         # 2. 质量四极矩 I_ij = μ x_i x_j（几何单位）
#         mu_geom = params.mu

#         I_xx = mu_geom * x * x
#         I_yy = mu_geom * y * y
#         I_zz = mu_geom * z * z
#         I_xy = mu_geom * x * y
#         I_xz = mu_geom * x * z
#         I_yz = mu_geom * y * z

#         # 3. 二阶时间导数
#         I_xx_ddot = _second_time_derivative(I_xx, dt)
#         I_yy_ddot = _second_time_derivative(I_yy, dt)
#         I_zz_ddot = _second_time_derivative(I_zz, dt)
#         I_xy_ddot = _second_time_derivative(I_xy, dt)
#         I_xz_ddot = _second_time_derivative(I_xz, dt)
#         I_yz_ddot = _second_time_derivative(I_yz, dt)

#         # 4. 四极公式：h_ij = (2 / D_L) * d²I_ij/dt²
#         prefactor = 2.0 / D_L

#         h_xx = prefactor * I_xx_ddot
#         h_yy = prefactor * I_yy_ddot
#         h_zz = prefactor * I_zz_ddot
#         h_xy = prefactor * I_xy_ddot
#         h_xz = prefactor * I_xz_ddot
#         h_yz = prefactor * I_yz_ddot

#         # 5. 极化投影
#         if n_obs is None:
#             n_obs = np.array([0.0, 0.0, 1.0])  # face-on

#         p, q, n = _orthonormal_basis_from_n(n_obs)

#         e_plus = np.outer(p, p) - np.outer(q, q)
#         e_cross = np.outer(p, q) - np.outer(q, p)

#         N = t.size
#         h_plus = np.zeros(N, dtype=float)
#         h_cross = np.zeros(N, dtype=float)

#         for k in range(N):
#             Hk = np.array(
#                 [
#                     [h_xx[k], h_xy[k], h_xz[k]],
#                     [h_xy[k], h_yy[k], h_yz[k]],
#                     [h_xz[k], h_yz[k], h_zz[k]],
#                 ],
#                 dtype=float,
#             )
#             h_plus[k] = np.sum(Hk * e_plus)
#             h_cross[k] = np.sum(Hk * e_cross)
#         return NKPolarizations(t=t, h_plus=h_plus, h_cross=h_cross)

#     elif params.multipole_order == "quad+oct":
#         h_plus, h_cross = _nk_quadrupole_plus_octupole(...)
#     else:
#         raise ValueError(f"Unknown multipole_order: {params.multipole_order}")
#     return h_plus, h_cross
    
    
import numpy as np
from dataclasses import dataclass

# 假设你有一个包含 t, r, theta, phi 的 Trajectory 对象
# from ..orbits.nk_geodesic_orbit import NKOrbitTrajectory

@dataclass
class ObserverInfo:
    """观测者位置信息 (球坐标)"""
    R: float      # 距离 D
    theta: float  # 极角 Theta
    phi: float    # 方位角 Phi

def get_minkowski_trajectory(trajectory):
    """
    将 BL 坐标 (r, theta, phi) 转换为平直空间笛卡尔坐标 (x, y, z)。
    对应 Babak Eq. 11a - 11c
    """
    r = trajectory.r
    theta = trajectory.theta
    phi = trajectory.phi

    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.stack([x, y, z], axis=0) # shape (3, N_steps)

def compute_nk_waveform(trajectory, mu, M, observer: ObserverInfo, dt):
    """
    计算 Quadrupole-Octupole 波形 (Babak Eq. 18)。
    使用数值微分计算导数。
    """
    # 1. 获取位置 x^i(t) 和 速度 v^i(t)
    # x_vec shape: (3, N)
    x_vec = get_minkowski_trajectory(trajectory)
    
    # 计算速度 v = dx/dt
    # 使用 gradient 进行数值微分，axis=1 表示对时间维操作
    v_vec = np.gradient(x_vec, dt, axis=1, edge_order=2)
    
    # 2. 构造源的多极矩 (Source Moments)
    # mu: 小天体质量
    
    # Mass Quadrupole I_jk = mu * x_j * x_k (Babak Eq 25)
    # 使用 einsum 可以方便地构造张量积
    # I_tensor shape: (3, 3, N)
    I_tensor = mu * np.einsum('it,jt->ijt', x_vec, x_vec)
    
    # Current Quadrupole S_ijk = v_i * I_jk (Babak Eq 26) -> mu * v_i * x_j * x_k
    S_tensor = np.einsum('it,jkt->ijkt', v_vec, I_tensor)
    
    # Mass Octupole M_ijk = x_i * I_jk (Babak Eq 27) -> mu * x_i * x_j * x_k
    M_tensor = np.einsum('it,jkt->ijkt', x_vec, I_tensor)
    
    # 3. 数值微分求导
    # d2I/dt2
    d1_I = np.gradient(I_tensor, dt, axis=2, edge_order=2)
    d2_I = np.gradient(d1_I, dt, axis=2, edge_order=2)
    
    # d2S/dt2
    d1_S = np.gradient(S_tensor, dt, axis=3, edge_order=2)
    d2_S = np.gradient(d1_S, dt, axis=3, edge_order=2)
    
    # d3M/dt3
    d1_M = np.gradient(M_tensor, dt, axis=3, edge_order=2)
    d2_M = np.gradient(d1_M, dt, axis=3, edge_order=2)
    d3_M = np.gradient(d2_M, dt, axis=3, edge_order=2)
    
    # 4. 组装 h_jk (Babak Eq 18)
    # h_jk = (2/D) * [ d2I_jk - 2 n_i d2S_ijk + n_i d3M_ijk ]
    # 这里需要观测者方向向量 n_i
    
    # 观测者方向 n = (sinT cosP, sinT sinP, cosT)
    sin_T = np.sin(observer.theta)
    n_vec = np.array([
        sin_T * np.cos(observer.phi),
        sin_T * np.sin(observer.phi),
        np.cos(observer.theta)
    ]) # shape (3,)
    
    # 缩项 n_i S_ijk -> S_njk
    n_d2S = np.einsum('i,ijkt->jkt', n_vec, d2_S)
    
    # 缩项 n_i M_ijk -> M_njk
    n_d3M = np.einsum('i,ijkt->jkt', n_vec, d3_M)
    
    # 组合
    # term_quad = d2_I
    # term_oct  = -2 * n_d2S + n_d3M
    h_tensor = (2.0 / observer.R) * (d2_I - 2.0 * n_d2S + n_d3M)
    
    # 5. 投影到观测者坐标系 (TT gauge)
    # 这一步通常使用 Babak Eq 22-23 实现
    # 返回 h_plus, h_cross
    return project_to_tt(h_tensor, observer)

def project_to_tt(h_tensor, observer):
    """
    将 h_jk 投影得到 h_plus 和 h_cross (Babak Eq 22, 23)。
    """
    # 我们需要先把 h_tensor 从 (x,y,z) 基底转到球坐标基底 (r, theta, phi)
    # 或者直接使用 Eq 23 的显式公式，那样更简单。
    
    theta = observer.theta
    phi = observer.phi
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    c2p = np.cos(2*phi)
    s2p = np.sin(2*phi)
    
    hxx = h_tensor[0, 0, :]
    hyy = h_tensor[1, 1, :]
    hzz = h_tensor[2, 2, :]
    hxy = h_tensor[0, 1, :]
    hxz = h_tensor[0, 2, :]
    hyz = h_tensor[1, 2, :]
    
    # Babak Eq 23a
    # h_theta_theta
    h_TT = (ct**2) * (hxx * cp**2 + hxy * s2p + hyy * sp**2) + \
           hzz * (st**2) - \
           np.sin(2*theta) * (hxz * cp + hyz * sp)
           
    # Babak Eq 23c (注意：这里应该是 h_phi_phi，公式里写的是 h^PhiPhi)
    h_PP = hxx * sp**2 - hxy * s2p + hyy * cp**2
    
    # Babak Eq 23b
    h_TP = ct * (-0.5 * hxx * s2p + hxy * c2p + 0.5 * hyy * s2p) + \
           st * (hxz * sp - hyz * cp)
           
    # 极化模式
    h_plus = h_TT - h_PP
    h_cross = 2 * h_TP
    
    return h_plus, h_cross