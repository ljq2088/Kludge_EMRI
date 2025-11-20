# src/emrikludge/parameters.py
"""
EMRI 与波形配置参数。

- EMRIParameters: 物理系统参数（MBH/CO 质量、自旋、初始轨道等）；
- WaveformConfig: 波形生成相关的数值/控制参数；
- LISAConfig: 探测器相关参数（目前只做简单占位，可逐步扩展）。

与其他模块的关系：
- core.aak_cpu: 顶层生成 AAK 波形的入口，接收 (EMRIParameters, WaveformConfig)；
- orbits.aak_osculating_orbit: 使用 EMRIParameters 的几何单位质量和初始
  (p,e,iota) 来进行 PN 演化；
- waveforms.aak_waveform: 使用参数 + 轨道轨迹生成 (h_+, h_x) 并投影到 LISA。
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from dataclasses import dataclass
from typing import Optional

from .constants import (
    mass_solar_to_seconds,
    distance_gpc_to_meters,
)


@dataclass
class EMRIParameters:
    """
    EMRI 内禀 & 部分外禀参数。

    单位约定：
    - M, mu: 以太阳质量 M_sun 为单位输入；
    - distance: 以 Gpc 为单位输入（便于和 LISA 文献对应）；
    - p0: 半通径，单位 M (几何单位)，即 p0 是除以 MBH 质量后的 p/M；
    - angles （inclination, sky angles 等）用弧度。

    注意：
    - M_default = 1.0e6 M_sun
    - mu_default = 10.0 M_sun
    （按照你的要求）
    """

    # --- 质量与自旋 ---
    M: float = 1.0e6   # MBH mass in M_sun
    mu: float = 10.0   # CO mass in M_sun
    a: float = 0.7     # dimensionless spin, a = S / M^2

    # --- 初始轨道参数 (AK/AAK 习惯) ---
    p0: float = 10.0   # semi-latus rectum in units of M (dimensionless)
    e0: float = 0.2    # eccentricity
    iota0: float = 0.3 # orbital inclination to BH spin axis (radians)

    # --- 初始相位 ---
    Phi0: float = 0.0      # orbital phase at t=0
    gamma0: float = 0.0    # periastron precession phase
    alpha0: float = 0.0    # Lense-Thirring precession phase

    # --- 距离 & 天区位置 ---
    distance: float = 1.0  # luminosity distance in Gpc
    theta_S: float = 1.0   # source polar angle in SSB frame
    phi_S: float = 2.0     # source azimuthal angle in SSB frame
    theta_K: float = 0.2   # BH spin polar angle in SSB frame
    phi_K: float = 1.3     # BH spin azimuthal angle in SSB frame
    psi: float = 0.0       # polarization angle

    def M_geom(self) -> float:
        """MBH 质量（几何单位，单位：秒）。"""
        return mass_solar_to_seconds(self.M)

    def mu_geom(self) -> float:
        """CO 质量（几何单位，单位：秒）。"""
        return mass_solar_to_seconds(self.mu)

    def eta(self) -> float:
        """质量比 η = mu / M。"""
        return self.mu / self.M

    def distance_SI(self) -> float:
        """返回物理单位下的距离（米）。"""
        return distance_gpc_to_meters(self.distance)
@dataclass
class AAKParameters(EMRIParameters):
    """
    AAK 专用 EMRI 参数封装。

    说明：
    - 继承自 EMRIParameters，包含：
      * M, mu, a               : MBH / 小天体质量、自旋
      * p0, e0, iota0          : 初始轨道参数（半通径、偏心率、倾角）
      * Phi0, gamma0, alpha0   : 初始相位（轨道 / 近日点 / 平面进动）
      * D_L, theta_S, phi_S,   : 外禀参数（天区位置、极化角等）
        theta_K, phi_K, psi0, varphi0, chi0

    作用：
    - 在类型上把「AAK 算法」的参数和 NK 参数区分开；
    - 以后如果只对 AAK 生效的开关（例如 use_aak_mapping、
      是否做相位重采样）需要新增字段，可以直接加在这里，
      不会影响 NK 侧代码。
    """
    # 目前不新增字段，仅做语义区分
    pass


@dataclass
class NKParameters(EMRIParameters):
    """
    NK（Numerical Kludge）专用 EMRI 参数封装。

    在 EMRIParameters 的物理量基础上，增加若干
    『近似方案 / kludge 方案』相关的开关：

    geodesic_parametrization :
        geodesic 常数的参数化方式：
        - 'p-e-iota' : 用 (p, e, iota) 作为主变量，在内部转换为
                       Kerr 不变量 (E, L_z, Q) 来构造测地线；
        - 'ELQ'      : 未来可扩展为直接输入 (E, L_z, Q) 的模式。

    nk_flux_model :
        NK 辐射反作用的能流 / 角动量流模型选择，对应文献：
        - 'PM'    : 仅用 Peters & Mathews 0PN 结果（调试用）；
        - 'Ryan96': F. Ryan 1996 对 Kerr 非赤道轨道的 PN 能流展开 ；
        - 'GG06' : Gair & Glampedakis 2006 的 Teukolsky 数值拟合 / hybrid flux ；
        后续也可以扩展为 Teukolsky 拟合表等。

    multipole_order :
        NK 波形中多极矩截断级别：
        - 'quad'       : 仅质量四极矩（当前已经实现）；
        - 'quad+oct'   : 质量四极 + 电流四极 / 质量八极（Barack & Cutler,
                         Babak et al. 的 kludge 波形多极展开 :contentReference[oaicite:2]{index=2}）。

    include_rr :
        是否开启辐射反作用：
        - True  : 调用 nk_fluxes 计算 (dp/dt, de/dt, diota/dt)，
                  做逐步 inspiral；
        - False : 只演化测地线相位 (psi_r, psi_theta, psi_phi)，
                  轨道参数 (p,e,iota) 固定，对应「冻结轨道」的纯 NK 波形。
    """

    geodesic_parametrization: str = "p-e-iota"
    nk_flux_model: str = "GG06"
    multipole_order: str = "quad"
    include_rr: bool = True

@dataclass
class WaveformConfig:
    """
    波形生成配置。

    与 AAK/NK 算法本身无关，但控制：
    - 采样率、总时长；
    - 轨道积分精度；
    - 谐波截断、是否只保留主谐波等；
    """

    dt: float = 10.0
    T: float = 2.0 * 365.25 * 86400.0

    # 轨道积分精度控制
    atol: float = 1e-10
    rtol: float = 1e-10
    max_step: Optional[float] = None   # 若为 None，则由积分器自行决定

    # 波形内容选择
    include_lisa_response: bool = True
    include_only_quadrupole: bool = False  # 是否只保留主谐波 n=2（调试用）

    # --- Peters–Mathews / AK 的谐波截断 ---
    n_max: int = 20
    """
    在多谐波展开中保留的最高谐波编号 n_max。
    一般 n_max ~ 20 足以覆盖 e≲0.7 的情况。
    """

    use_aak_frequency_mapping: bool = True

    use_precession_in_phase: bool = True
    """
    是否在每个谐波的相位里显式加入 (2*gamma + 2*alpha)。
    """
 
    """
    轨道演化所用的 PN 通量模型：
    - "BC04" : 使用 Barack & Cutler 2004 的 dν/dt, de/dt, dΦ/dt, dγ/dt, dα/dt（推荐）
    - "PM"   : 使用 Peters & Mathews 纯四极近似，只对 (p,e) 演化（调试/对比用）

    默认 "BC04"。
    """
    
    # AAK / NK 共用的能流模型标记，可以和 NKParameters.nk_flux_model 对齐
    flux_model: str = "PM"
    """
    'PM'     : Peters–Mathews 0PN (默认)
    'BC04'   : Barack & Cutler 2004 在 AAK 侧的 3PN flux（AAK 已经接入）
    'Ryan96' : Ryan (NK 侧备选)
    'GG06'   : Gair & Glampedakis (NK 侧备选)
    """
@dataclass
class LISAConfig:
    """
    LISA 轨道与噪声配置（占位）：

    - 目前只保留最基本的参数，后续可以与你已有的 noise_curves / orbit 模块对接；
    - 这里只做 AAK/NK 波形结构示范。
    """
    arm_length: float = 2.5e9  # meters, LISA 经典 2.5 Gm
    orbit_approx: str = "low-frequency"  # 'low-frequency' or 'rigid-adiabatic'
