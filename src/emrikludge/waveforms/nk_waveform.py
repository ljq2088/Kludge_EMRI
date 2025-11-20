# src/emrikludge/waveforms/nk_waveform.py
"""
Numerical Kludge (NK) 波形模块 —— 目前实现：纯质量四极矩版本。

输入：NKOrbitTrajectory（近似 Kerr 轨道，多频相位）
输出：源坐标系下的 h_+(t), h_×(t)。

思路：
- 在“伪平直”笛卡尔坐标系中构造粒子轨道 x(t), y(t), z(t)
- 质量四极矩：I_ij = μ x_i x_j
- 数值二阶导数：d²I_ij/dt²
- 四极公式（几何单位）：h_ij = (2 / D_L) d²I_ij/dt²
- 对给定观测方向 n，构造极化基 (p,q,n)，再投影得到 h_+, h_×.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..parameters import EMRIParameters, WaveformConfig
from ..orbits.nk_geodesic_orbit import NKOrbitTrajectory
from ..parameters import NKParameters

@dataclass
class NKPolarizations:
    """NK 波形的极化结果容器。"""
    t: np.ndarray
    h_plus: np.ndarray
    h_cross: np.ndarray


# ========= 工具函数：数值二阶导 =========

def _second_time_derivative(y: np.ndarray, dt: float) -> np.ndarray:
    """
    对一维时间序列 y(t) 做二阶中心差分，返回 d²y/dt²。
    """
    d2y = np.zeros_like(y)
    d2y[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dt * dt)
    if y.size > 1:
        d2y[0] = d2y[1]
        d2y[-1] = d2y[-2]
    return d2y


# ========= 工具函数：由观测方向 n 构造极化基 (p, q, n) =========

def _orthonormal_basis_from_n(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定观测方向 n（3 维向量），构造一组正交归一基 (p, q, n)。
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)

    if abs(n[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])

    p = np.cross(tmp, n)
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-14:
        p = np.array([1.0, 0.0, 0.0])
    else:
        p = p / p_norm

    q = np.cross(n, p)

    return p, q, n


# ========= 核心：从 NK 轨道生成 h_+, h_× =========

def generate_nk_polarizations(
    traj: NKOrbitTrajectory,
    params: NKParameters,
    config: WaveformConfig,
    D_L: float = 1.0,
    n_obs: Optional[np.ndarray] = None,
) -> NKPolarizations:
    """
    生成 NK 波形 (h_+, h_×)。

    - multipole_order = 'quad' 时：只计算质量四极矩
    - multipole_order = 'quad+oct' 时：额外加入质量八极 + 电流四极
      （具体公式见 Barack & Cutler 2004, Appendix; Babak et al. 2007）
    """
    if params.multipole_order == "quad":
        t = traj.t
        dt = config.dt

        # 1. 粒子轨道的笛卡尔坐标 (x, y, z)
        r = traj.r_over_M
        theta = traj.theta
        phi = traj.phi

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        cos_ph = np.cos(phi)
        sin_ph = np.sin(phi)

        x = r * sin_th * cos_ph
        y = r * sin_th * sin_ph
        z = r * cos_th

        # 2. 质量四极矩 I_ij = μ x_i x_j（几何单位）
        mu_geom = params.mu

        I_xx = mu_geom * x * x
        I_yy = mu_geom * y * y
        I_zz = mu_geom * z * z
        I_xy = mu_geom * x * y
        I_xz = mu_geom * x * z
        I_yz = mu_geom * y * z

        # 3. 二阶时间导数
        I_xx_ddot = _second_time_derivative(I_xx, dt)
        I_yy_ddot = _second_time_derivative(I_yy, dt)
        I_zz_ddot = _second_time_derivative(I_zz, dt)
        I_xy_ddot = _second_time_derivative(I_xy, dt)
        I_xz_ddot = _second_time_derivative(I_xz, dt)
        I_yz_ddot = _second_time_derivative(I_yz, dt)

        # 4. 四极公式：h_ij = (2 / D_L) * d²I_ij/dt²
        prefactor = 2.0 / D_L

        h_xx = prefactor * I_xx_ddot
        h_yy = prefactor * I_yy_ddot
        h_zz = prefactor * I_zz_ddot
        h_xy = prefactor * I_xy_ddot
        h_xz = prefactor * I_xz_ddot
        h_yz = prefactor * I_yz_ddot

        # 5. 极化投影
        if n_obs is None:
            n_obs = np.array([0.0, 0.0, 1.0])  # face-on

        p, q, n = _orthonormal_basis_from_n(n_obs)

        e_plus = np.outer(p, p) - np.outer(q, q)
        e_cross = np.outer(p, q) - np.outer(q, p)

        N = t.size
        h_plus = np.zeros(N, dtype=float)
        h_cross = np.zeros(N, dtype=float)

        for k in range(N):
            Hk = np.array(
                [
                    [h_xx[k], h_xy[k], h_xz[k]],
                    [h_xy[k], h_yy[k], h_yz[k]],
                    [h_xz[k], h_yz[k], h_zz[k]],
                ],
                dtype=float,
            )
            h_plus[k] = np.sum(Hk * e_plus)
            h_cross[k] = np.sum(Hk * e_cross)
        return NKPolarizations(t=t, h_plus=h_plus, h_cross=h_cross)

    elif params.multipole_order == "quad+oct":
        h_plus, h_cross = _nk_quadrupole_plus_octupole(...)
    else:
        raise ValueError(f"Unknown multipole_order: {params.multipole_order}")
    return h_plus, h_cross
    
    
