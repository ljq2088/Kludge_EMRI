# # src/emrikludge/core/aak_cpu.py
# from __future__ import annotations

# from dataclasses import asdict
# from typing import Tuple, Dict, Any

# import numpy as np

# from ..parameters import EMRIParameters, WaveformConfig as PyWaveformConfig
# from ..logging_utils import get_logger

# # 导入 C++ 扩展模块（名字需与 bindings_aak.cpp 中的 PYBIND11_MODULE 一致）
# from . import _aak_cpu  # 假设在 __init__.py 里 from _aak_cpu import * 或直接安装为同名模块

# logger = get_logger(__name__)


# def _to_cpp_emri(params: EMRIParameters) -> "_aak_cpu.EMRIParams":
#     """
#     将 Python 端的 EMRIParameters dataclass 转换为 C++ EMRIParams 对象。

#     注意：这里要求 Python dataclass 的字段名与 C++ 结构体中一致，
#     即 M, mu, a, p0, e0, iota0, thetaS, phiS, ...。
#     """
#     cpp_emri = _aak_cpu.EMRIParams()
#     d = asdict(params)
#     for key, value in d.items():
#         if hasattr(cpp_emri, key):
#             setattr(cpp_emri, key, float(value) if isinstance(value, (int, float)) else value)
#         else:
#             logger.debug("Python EMRIParameters 字段 %s 在 C++ EMRIParams 中不存在，忽略。", key)
#     return cpp_emri


# def _to_cpp_wfconf(conf: PyWaveformConfig) -> "_aak_cpu.WaveformConfig":
#     """
#     将 Python 端的 WaveformConfig 转换为 C++ WaveformConfig。
#     """
#     cpp_conf = _aak_cpu.WaveformConfig()
#     d = asdict(conf)
#     for key, value in d.items():
#         if hasattr(cpp_conf, key):
#             setattr(cpp_conf, key, value)
#         else:
#             logger.debug("Python WaveformConfig 字段 %s 在 C++ WaveformConfig 中不存在，忽略。", key)
#     return cpp_conf


# def compute_aak_waveform_cpu(
#     params: EMRIParameters,
#     conf: PyWaveformConfig,
# ) -> Dict[str, np.ndarray]:
#     """
#     高层接口：调用 C++ 扩展计算 AAK 波形（CPU 版本）。

#     这里不做复杂逻辑，只负责：
#       1. 参数转换；
#       2. 调用 _aak_cpu.compute_aak_waveform_cpu；
#       3. 将结果转成 numpy 数组，方便 Python 上层使用。
#     """
#     cpp_emri = _to_cpp_emri(params)
#     cpp_conf = _to_cpp_wfconf(conf)

#     logger.info("调用 C++ AAK CPU 核心进行波形计算...")
#     res = _aak_cpu.compute_aak_waveform_cpu(cpp_emri, cpp_conf)

#     # 结果转换为 numpy 数组
#     out: Dict[str, np.ndarray] = {
#         "t": np.asarray(res.t, dtype=float),
#         "hI": np.asarray(res.hI, dtype=float),
#         "hII": np.asarray(res.hII, dtype=float),
#     }

#     # 如果存在可选字段，则也转换
#     if getattr(res, "hplus", None):
#         out["hplus"] = np.asarray(res.hplus, dtype=float)
#     if getattr(res, "hcross", None):
#         out["hcross"] = np.asarray(res.hcross, dtype=float)
#     if getattr(res, "r", None):
#         out["r"] = np.asarray(res.r, dtype=float)
#     if getattr(res, "theta", None):
#         out["theta"] = np.asarray(res.theta, dtype=float)
#     if getattr(res, "phi", None):
#         out["phi"] = np.asarray(res.phi, dtype=float)

#     return out
# src/emrikludge/core/aak_cpu.py
"""
AAK CPU 波形生成主入口（Python 版）。

功能：
- 提供高级接口 `generate_aak_waveform_cpu`：
    输入：EMRIParameters, WaveformConfig
    输出：t, h_I, h_II, h_plus, h_cross

与其他模块的关系：
- orbits.aak_osculating_orbit.evolve_aak_orbit: 轨道演化；
- waveforms.aak_waveform.generate_aak_polarizations:
    从轨道得到 (h_+, h_x)；
- waveforms.aak_waveform.project_to_lisa_channels:
    投影到 LISA 通道。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..parameters import EMRIParameters, WaveformConfig
from ..orbits.aak_osculating_orbit import evolve_aak_orbit
from ..waveforms.aak_waveform import (
    generate_aak_polarizations,
   
)
from ..lisa.response_approx import project_to_lisa_channels


def generate_aak_waveform_cpu(params: EMRIParameters,
                              config: WaveformConfig):
    """
    端到端 AAK (CPU)：
    1. 积分 AK/AAK 轨道 (含 Φ,γ,α 演化)；
    2. 在源坐标系下生成多谐波 h_+, h_×；
    3. 通过 LISA 响应变换成 hI, hII。
    """
    traj = evolve_aak_orbit(params, config)
    h_plus, h_cross = generate_aak_polarizations(traj, params, config)
    t = traj.t

    # 这里先用你已有的 LISA 低频近似；以后可以用 BC04 的
    # 天线模式公式 (20–26) 替换，真正做“轨道调制”。
    hI, hII = project_to_lisa_channels(t, h_plus, h_cross, params, config)

    return t, hI, hII, h_plus, h_cross