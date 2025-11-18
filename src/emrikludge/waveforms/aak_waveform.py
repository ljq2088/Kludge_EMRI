# src/emrikludge/waveforms/aak_waveform.py
"""
AAK 波形生成（时间域）——基于 AK/AAK 结构 + Peters–Mathews 振幅。

功能：
- 给定 AAKOrbitTrajectory 和 EMRIParameters, WaveformConfig；
- 使用 Peters–Mathews 的多谐波公式，计算 h_+, h_x；
- 可选：使用简化 LISA 响应计算两个独立通道 h_I, h_II
  （这里的响应只是占位，将来可以替换为你已有的 tdi_response 模块）。

与其他模块的关系：
- orbits.aak_osculating_orbit.evolve_aak_orbit: 提供轨道 (t,p,e,...)；
- core.aak_cpu: 顶层函数使用本模块生成最终波形；
- lisa.response_approx: 将来可以替换本文件里的简单投影。
"""

from __future__ import annotations
import numpy as np
from scipy.special import jv  # Bessel J_n
from typing import Tuple

from ..constants import G_SI, C_SI, M_SUN_SI
from ..parameters import EMRIParameters, WaveformConfig
from ..orbits.aak_osculating_orbit import AAKOrbitTrajectory
from typing import Tuple

import numpy as np
from scipy.special import jv as BesselJ

from ..parameters import EMRIParameters, WaveformConfig
from ..constants import G_SI, C_SI, M_SUN_SI
from ..utils.math_utils import safe_divide
def peters_mathews_harmonics(n: int,
                             e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Peters–Mathews 模型中，第 n 个谐波的系数 a_n(e), b_n(e)。

    支持 e 是标量或 ndarray，返回的 a_n, b_n 与 e 形状一致，便于
    在整个轨道时间序列上矢量化计算。

    标准形式（见 Peters & Mathews 1963；Favata 2011 Sec.II A）\cite{Peters1963,Favata2011}：
        a_n(e) = -n [ J_{n-2}(ne) - 2 e J_{n-1}(ne)
                      + 2/n J_n(ne) + 2 e J_{n+1}(ne) - J_{n+2}(ne) ]
        b_n(e) = -sqrt(1-e^2) [ J_{n-2}(ne) - 2 J_n(ne) + J_{n+2}(ne) ]

    其中 J_k 是第一类 Bessel 函数 J_k(x)。
    """
    e = np.asarray(e)

    # 允许非常小的偏心率，避免 sqrt(1-e^2) 的数值问题
    # 这里简单裁剪一下，e->0 的极限自然会落到 n=2 主谐波
    e_clipped = np.clip(e, 0.0, 0.999999999)

    ne = n * e_clipped

    Jm2 = jv(n - 2, ne)
    Jm1 = jv(n - 1, ne)
    J0  = jv(n,     ne)
    Jp1 = jv(n + 1, ne)
    Jp2 = jv(n + 2, ne)

    a_n = -n * (Jm2 - 2.0 * e_clipped * Jm1
                + 2.0 / n * J0
                + 2.0 * e_clipped * Jp1 - Jp2)
    b_n = -np.sqrt(1.0 - e_clipped**2) * (Jm2 - 2.0 * J0 + Jp2)

    return a_n, b_n


def _peters_mathews_harmonic_amplitudes(n: int,
                                        e: float,
                                        M_solar: float,
                                        mu_solar: float,
                                        distance_m: float,
                                        f_orb: float,
                                        iota: float) -> Tuple[float, float]:
    """
    计算 Peters–Mathews 公式中第 n 阶谐波对应的 (h_+, h_x) 振幅贡献。

    具体公式可见：
    - Peters & Mathews 1963, Phys. Rev. 131, 435;
    - 以及很多教材/综述对该公式的再现。

    这里不逐字抄写原文，而是根据标准结果用 Bessel 函数组合给出。

    注意：
    - 这里仍然是假设在“源坐标系”中观测的 (h_+, h_x)，
      没有包含 LISA 轨道和臂长投影效应；
    - LISA 投影应在后一步使用 response 函数实现。
    """
    m1 = M_solar * M_SUN_SI
    m2 = mu_solar * M_SUN_SI
    Mtot = m1 + m2
    eta = m1 * m2 / Mtot**2

    # 轨道角频率
    omega_orb = 2.0 * np.pi * f_orb  # rad/s
    omega_n = n * omega_orb

    # 标准化的“频率因子”，见多处教材
    x = (G_SI * Mtot * omega_orb / C_SI**3) ** (2.0 / 3.0)

    # Peters–Mathews 中的 Bessel 组合：
    # 这里使用常见形式中的系数，简写为 c_plus, c_cross。
    # 详细系数请对照 Peters & Mathews 原文或任何 GR 教材。
    ne = n * e
    Jn_2 = BesselJ(n - 2, ne)
    Jn_1 = BesselJ(n - 1, ne)
    Jn_0 = BesselJ(n, ne)
    Jn1 = BesselJ(n + 1, ne)
    Jn2 = BesselJ(n + 2, ne)

    # 下面这些组合因子是 Peters–Mathews 标准结果的重新写法；
    # 为避免一次性复制长公式，这里写出“结构”：
    #
    # c_plus ∝ [ -(1 - e^2)(Jn_2 - 2e Jn_1 + 2/n Jn + 2e Jn1 - Jn2) +
    #            (2 cos^2 iota - sin^2 iota) * (Jn_2 - 2 Jn + Jn2)
    #          ]
    #
    # c_cross ∝ cos(iota) * [ (1 - e^2)(Jn_2 - 2e Jn_1 + 2/n Jn + 2e Jn1 - Jn2) +
    #                         (Jn_2 - 2 Jn + Jn2)
    #                       ]
    #
    # 具体数值系数（如 1/n 等）可在需要时对照 Peters 论文修正；
    # 这里的重点是体现出完整 Bessel 结构，而非截断至少数谐波。

    # 一个合理的实现是引入辅助量：
    A_n = Jn_2 - 2.0 * e * Jn_1 + 2.0 * Jn_0 / safe_divide(n, 1.0) \
        + 2.0 * e * Jn1 - Jn2
    B_n = Jn_2 - 2.0 * Jn_0 + Jn2

    # 常规前因子：
    # h ∼ (G M_tot η / (c^2 R)) x * [组合系数]
    pref = 2.0 * G_SI * Mtot * eta / (C_SI**2 * distance_m)
    pref *= x

    cosi = np.cos(iota)
    cos2i = cosi**2
    sin2i = 1.0 - cos2i

    c_plus = -(1.0 + cos2i) * A_n + sin2i * B_n
    c_cross = 2.0 * cosi * (A_n + B_n)

    h_plus_n = pref * c_plus
    h_cross_n = pref * c_cross

    return h_plus_n, h_cross_n


def generate_aak_polarizations(
    traj: AAKOrbitTrajectory,
    params: EMRIParameters,
    config: WaveformConfig,
    D_L: float = 1.0e9 * 3.085677581e16,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    在源坐标系下生成 AAK 波形的两个极化态 h_+(t), h_×(t)。

    - 使用 Peters–Mathews 多谐波展开；
    - 每个谐波的相位中显式包含 AK/AAK 的 (Phi, gamma, alpha)；
    - 幅度使用圆轨道 0PN 标度 + e 依赖的 Bessel 系数。

    参考：
    - Peters & Mathews 1963, Phys.Rev. 131, 435. 
    - Favata 2011, Class.Quant.Grav. 28 103001 (Sec.II A). 
    - Barack & Cutler 2004, Phys.Rev.D 69, 082005 (AK 波形整体结构). 
    """
    t = traj.t
    e = traj.e
    iota = traj.iota
    Phi = traj.Phi
    gamma = traj.gamma
    alpha = traj.alpha

    # 谐波截断：统一处理
        # 谐波截断：统一使用 WaveformConfig.n_max
    if config.include_only_quadrupole:
        n_max = 2
    else:
        n_max = config.n_max

    # 质量 + 距离标度
    M_SI = params.M * M_SUN_SI
    mu_SI = params.mu * M_SUN_SI
    M_tot = M_SI + mu_SI

    # 对 EMRI 来说，chirp mass 近似为 (mu^(3/5) M^(2/5))
    Mc = (mu_SI**3 * M_SI**2)**(1.0 / 5.0)

    # 用 dPhi/dt 估计轨道频率
    dPhi_dt = np.gradient(Phi, t, edge_order=2)  # [rad/s]
    f_orb = np.abs(dPhi_dt) / (2.0 * np.pi)      # [Hz]

    # 0PN 圆轨道幅度标度：h0 ~ (G^{5/3}/c^4 D) Mc^{5/3} (π f)^{2/3}
    amp_prefac = (G_SI**(5.0 / 3.0) / (C_SI**4 * D_L)) \
        * Mc**(5.0 / 3.0) * (np.pi * f_orb)**(2.0 / 3.0)

    cosi = np.cos(iota)
    cos2i = cosi**2

    h_plus = np.zeros_like(t, dtype=float)
    h_cross = np.zeros_like(t, dtype=float)

    for n in range(1, n_max + 1):
        a_n, b_n = peters_mathews_harmonics(n, e)

        if config.use_precession_in_phase:
            # AK/AAK 相位组合：n Φ + 2γ + 2α
            phi_n = n * Phi + 2.0 * gamma + 2.0 * alpha
        else:
            phi_n = n * Phi

        cos_phi = np.cos(phi_n)
        sin_phi = np.sin(phi_n)

        # Peters–Mathews 结构：
        #   h_+ = Σ [ (1+cos^2ι) a_n cos(nΦ) + 2 cosι b_n sin(nΦ) ]
        #   h_× = Σ [ 2 cosι a_n sin(nΦ) - 2 b_n cos(nΦ) ]
        h_plus += amp_prefac * (
            (1.0 + cos2i) * a_n * cos_phi + 2.0 * cosi * b_n * sin_phi
        )
        h_cross += amp_prefac * (
            2.0 * cosi * a_n * sin_phi - 2.0 * b_n * cos_phi
        )

    return h_plus, h_cross

from typing import Optional

def project_to_lisa_channels(t,h_plus: np.ndarray,
                             h_cross: np.ndarray,
                             params: Optional[EMRIParameters] = None,
    config: Optional[WaveformConfig] = None,) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 (h_+, h_x) 投影到两个独立的 LISA-like 通道 (h_I, h_II)。

    这里先使用极简的“定向天线 pattern”：
        h_I  = F_I^+ h_+ + F_I^× h_×
        h_II = F_{II}^+ h_+ + F_{II}^× h_×

    F^+, F^× 只依赖 source angles 和 polarization angle，
    具体表达式参考 LISA 低频极限近似。

    将来你可以在 lisa/response_approx.py 或 tdi_response.py 里
    实现更精确的 TDI 响应，并在这里调用。
    params, config : 可选
        EMRIParameters 和 WaveformConfig，用于未来扩展。
    """
    # 简单的天线 pattern（取 LISA 低频极限的形式）：
    theta = params.theta_S
    phi = params.phi_S
    psi = params.psi

    # 下面是标准地面 interferometer 的 F^+, F^× 形式，
    # 作为占位用；LISA 实际 pattern 会随时间变化：
    cos2psi = np.cos(2.0 * psi)
    sin2psi = np.sin(2.0 * psi)
    cos2phi = np.cos(2.0 * phi)
    sin2phi = np.sin(2.0 * phi)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    F_plus = 0.5 * (1.0 + costheta**2) * cos2phi * cos2psi \
        - costheta * sin2phi * sin2psi
    F_cross = 0.5 * (1.0 + costheta**2) * cos2phi * sin2psi \
        + costheta * sin2phi * cos2psi

    # 两个“虚拟”通道，可以简单地取旋转 45°
    F_plus_I = F_plus
    F_cross_I = F_cross
    F_plus_II = F_plus * np.cos(np.pi / 4.0) - F_cross * np.sin(np.pi / 4.0)
    F_cross_II = F_plus * np.sin(np.pi / 4.0) + F_cross * np.cos(np.pi / 4.0)

    h_I = F_plus_I * h_plus + F_cross_I * h_cross
    h_II = F_plus_II * h_plus + F_cross_II * h_cross

    return h_I, h_II
# # src/emrikludge/waveforms/aak_waveform.py
# from __future__ import annotations

# from typing import Dict

# import numpy as np

# from ..parameters import EMRIParameters, WaveformConfig
# from ..core.aak_cpu import compute_aak_waveform_cpu
# from ..logging_utils import get_logger

# logger = get_logger(__name__)


# def generate_aak_waveform(
#     params: EMRIParameters,
#     config: WaveformConfig | None = None,
# ) -> Dict[str, np.ndarray]:
#     """
#     端到端生成 AAK 波形（CPU 版本）。

#     参数
#     ----
#     params : EMRIParameters
#         EMRI 系统参数（质量、自旋、轨道、源位置等）。
#     config : WaveformConfig, optional
#         波形配置。如果为 None，则使用默认配置。

#     返回
#     ----
#     dict[str, np.ndarray]
#         包含键:
#           - "t"  : 时间数组
#           - "hI" : 探测器 I 通道波形
#           - "hII": 探测器 II 通道波形
#         如果在 config 中开启相应选项，还可能包含:
#           - "hplus", "hcross"
#           - "r", "theta", "phi"
#     """
#     if config is None:
#         config = WaveformConfig.default_for_aak()

#     logger.info("生成 AAK 波形: %s", params)

#     # 直接调用 core 层的 CPU 核
#     res = compute_aak_waveform_cpu(params, config)

#     # 此处可以做一些简单的后处理，例如单位转换、裁剪时间区间等，
#     # 但建议尽量保持“薄封装”，把复杂处理放到更高层或用户脚本中。

#     return res