# src/emrikludge/lisa/response_approx.py
"""
LISA 响应近似模块（占位版本）

当前目标
--------
- 提供一个统一的接口 project_to_lisa_channels(...)，
  供 AAK / NK 顶层代码调用；
- 先给出一个“最简近似”：忽略 LISA 轨道与天线模式，
  直接令
      h_I(t)  = h_+(t)
      h_II(t) = h_×(t)
  这样整个端到端框架可以跑通，不会因缺 LISA 响应而报错。

后续升级
--------
- 可以用 Cutler 1998、Barack & Cutler 2004、Cornish & Rubbo 2003 中的
  LISA 低频近似/刚体近似，真正引入：
    - 天区位置 (λ, β)；
    - 偏振角 ψ；
    - LISA 轨道调制（年周期、TDI X/Y/Z 或 A/E/T）。  # 参考文献见笔记
- 也可以调用你已有的 lisa/orbit.py, tdi_response.py 做更精细建模。
"""

from typing import Tuple

import numpy as np

from ..parameters import EMRIParameters, WaveformConfig
from typing import Optional

# def project_to_lisa_channels(
#     t: np.ndarray,
#     h_plus: np.ndarray,
#     h_cross: np.ndarray,
#     params: EMRIParameters,
#     config: WaveformConfig,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     将源坐标系下的 (h_+, h_×) 投影到 LISA 的两个伪通道 (h_I, h_II)。

#     当前实现（占位版）：
#     --------------------
#     - 完全忽略 LISA 的天线模式与轨道调制：
#           h_I  = h_+
#           h_II = h_×
#     - 这样做的目的只是保证 AAK / NK 框架可以端到端运行，
#       便于你先调试轨道和波形本身（谐波结构、相位演化等）；
#     - 一旦你准备好使用更真实的 LISA 响应（例如 Cutler 1998 的
#       低频刚体近似），只需要在本函数中用真正的
#       F_+(t), F_×(t) 替换掉下面两行即可。

#     参数
#     ----
#     t : ndarray
#         时间数组（秒）。
#     h_plus, h_cross : ndarray
#         源坐标系下的两种极化。
#     params : EMRIParameters
#         系统参数（目前没有使用，但保留接口，后续可从中读入天区位置等）。
#     config : WaveformConfig
#         波形配置（目前没有使用）。

#     返回
#     ----
#     hI, hII : ndarray
#         LISA 的两个伪通道波形。
#     """
#     # 简单占位：直接当作两个互相正交的“虚拟干涉仪”
#     hI = np.array(h_plus, copy=True)
#     hII = np.array(h_cross, copy=True)
#     return hI, hII
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