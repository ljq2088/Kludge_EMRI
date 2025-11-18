# src/emrikludge/constants.py
"""
常数与单位转换模块。

- 提供 SI 单位下的基础常数 (G, c, M_SUN, PC 等)；
- 提供几何单位转换因子：
    - M_sun_to_seconds, M_sun_to_meters
- 约定：整个 Kludge_EMRI 内部“主计算”统一使用几何单位 G=c=1，
  质量用秒（或米）表示，时间用秒表示，距离用秒（或米）表示。

与其他模块的关系：
- parameters.EMRIParameters: 将用户输入的 (M, mu) [单位: M_sun]
  转为几何单位质量；
- orbits.evolution_pn_fluxes: 使用 G,c 把 Peters–Mathews 公式写成
  对几何单位变量的演化；
- waveforms.aak_waveform: 在需要时进行距离/振幅单位转换。
"""

import numpy as np

# --- SI 常数 ---
G_SI = 6.67430e-11          # m^3 / (kg s^2)
C_SI = 2.99792458e8         # m / s
M_SUN_SI = 1.98847e30       # kg
PC_SI = 3.0856775814913673e16  # m
KPC_SI = 1.0e3 * PC_SI
MPC_SI = 1.0e6 * PC_SI
GPC_SI = 1.0e9 * PC_SI
YEAR_SI = 365.25 * 86400.0

# --- 几何单位转换 ---
# 1 M_sun 的几何质量：GM/c^3（单位：秒）
M_SUN_IN_SECONDS = G_SI * M_SUN_SI / C_SI**3
# 同样的，以米为单位：
M_SUN_IN_METERS = G_SI * M_SUN_SI / C_SI**2

def mass_solar_to_seconds(m_solar: float) -> float:
    """将以 M_sun 为单位的质量转换成几何单位（秒）。"""
    return m_solar * M_SUN_IN_SECONDS

def mass_solar_to_meters(m_solar: float) -> float:
    """将以 M_sun 为单位的质量转换成几何单位（米）。"""
    return m_solar * M_SUN_IN_METERS

def distance_mpc_to_meters(d_mpc: float) -> float:
    """将以 Mpc 为单位的距离转换成米。"""
    return d_mpc * MPC_SI

def distance_gpc_to_meters(d_gpc: float) -> float:
    """将以 Gpc 为单位的距离转换成米。"""
    return d_gpc * GPC_SI
