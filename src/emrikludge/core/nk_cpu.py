# src/emrikludge/core/nk_cpu.py
"""
NK (Numerical Kludge) 波形 —— CPU 端到端接口。

流程：
------
1. 调用 orbits.nk_geodesic_orbit.evolve_nk_orbit 生成 NK 轨道；
2. 调用 waveforms.nk_waveform.generate_nk_polarizations 生成 (h_+, h_×)；
3. 若需要，再投影到 LISA TDI 通道 (h_I, h_II)。

注意：
------
- 当前版本的轨道仍然是“后开普勒多频 + 通量”的近似，而非真 Kerr 测地线；
- 波形层目前只实现了质量四极矩，后续可加入八极矩与电流四极矩。
"""

from typing import Tuple, Optional

import numpy as np

from ..parameters import EMRIParameters, WaveformConfig
from ..orbits.nk_geodesic_orbit import evolve_nk_orbit
from ..waveforms.nk_waveform import generate_nk_polarizations
from ..lisa.response_approx import project_to_lisa_channels

def generate_nk_waveform_cpu(
    params: EMRIParameters,
    config: WaveformConfig,
    D_L: float = 1.0,
    n_obs: Optional[np.ndarray] = None,
):
    """
    端到端生成 NK 波形（当前版本：四极矩 + 近似 NK 轨道）。

    返回
    ----
    t : ndarray
        时间数组（与轨道一致）。
    hI, hII : ndarray or None
        LISA 伪 TDI 通道。若未开启 LISA 响应，则返回 None。
    h_plus, h_cross : ndarray
        源坐标系下的两种极化。

    调用链
    ------
    - evolve_nk_orbit -> NKOrbitTrajectory
    - generate_nk_polarizations -> h_+(t), h_×(t)
    - （可选）LISA 响应：project_to_lisa_channels(t, h_+, h_×, ...)
    """
    # 1. 轨道演化
    traj = evolve_nk_orbit(params, config)

    # 2. 源坐标系波形 (h_+, h_×)
    nk_pol = generate_nk_polarizations(
        traj,
        params,
        config,
        D_L=D_L,
        n_obs=n_obs,
    )
    t = nk_pol.t
    h_plus = nk_pol.h_plus
    h_cross = nk_pol.h_cross

    # 3. LISA 响应（与 AAK 保持同样接口）
    if config.include_lisa_response:
        
        # 若函数位置/签名不同，请按实际项目调整 import。
        

        hI, hII = project_to_lisa_channels(
            t,
            h_plus,
            h_cross,
            params,
            config,
        )
    else:
        hI = None
        hII = None

    return t, hI, hII, h_plus, h_cross
