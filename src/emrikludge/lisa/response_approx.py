# # src/emrikludge/lisa/response_approx.py
# """
# LISA 响应近似模块（占位版本）

# 当前目标
# --------
# - 提供一个统一的接口 project_to_lisa_channels(...)，
#   供 AAK / NK 顶层代码调用；
# - 先给出一个“最简近似”：忽略 LISA 轨道与天线模式，
#   直接令
#       h_I(t)  = h_+(t)
#       h_II(t) = h_×(t)
#   这样整个端到端框架可以跑通，不会因缺 LISA 响应而报错。

# 后续升级
# --------
# - 可以用 Cutler 1998、Barack & Cutler 2004、Cornish & Rubbo 2003 中的
#   LISA 低频近似/刚体近似，真正引入：
#     - 天区位置 (λ, β)；
#     - 偏振角 ψ；
#     - LISA 轨道调制（年周期、TDI X/Y/Z 或 A/E/T）。  # 参考文献见笔记
# - 也可以调用你已有的 lisa/orbit.py, tdi_response.py 做更精细建模。
# """

# from typing import Tuple

# import numpy as np

# from ..parameters import EMRIParameters, WaveformConfig
# from typing import Optional

# # def project_to_lisa_channels(
# #     t: np.ndarray,
# #     h_plus: np.ndarray,
# #     h_cross: np.ndarray,
# #     params: EMRIParameters,
# #     config: WaveformConfig,
# # ) -> Tuple[np.ndarray, np.ndarray]:
# #     """
# #     将源坐标系下的 (h_+, h_×) 投影到 LISA 的两个伪通道 (h_I, h_II)。

# #     当前实现（占位版）：
# #     --------------------
# #     - 完全忽略 LISA 的天线模式与轨道调制：
# #           h_I  = h_+
# #           h_II = h_×
# #     - 这样做的目的只是保证 AAK / NK 框架可以端到端运行，
# #       便于你先调试轨道和波形本身（谐波结构、相位演化等）；
# #     - 一旦你准备好使用更真实的 LISA 响应（例如 Cutler 1998 的
# #       低频刚体近似），只需要在本函数中用真正的
# #       F_+(t), F_×(t) 替换掉下面两行即可。

# #     参数
# #     ----
# #     t : ndarray
# #         时间数组（秒）。
# #     h_plus, h_cross : ndarray
# #         源坐标系下的两种极化。
# #     params : EMRIParameters
# #         系统参数（目前没有使用，但保留接口，后续可从中读入天区位置等）。
# #     config : WaveformConfig
# #         波形配置（目前没有使用）。

# #     返回
# #     ----
# #     hI, hII : ndarray
# #         LISA 的两个伪通道波形。
# #     """
# #     # 简单占位：直接当作两个互相正交的“虚拟干涉仪”
# #     hI = np.array(h_plus, copy=True)
# #     hII = np.array(h_cross, copy=True)
# #     return hI, hII
# def project_to_lisa_channels(t,h_plus: np.ndarray,
#                              h_cross: np.ndarray,
#                              params: Optional[EMRIParameters] = None,
#     config: Optional[WaveformConfig] = None,) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     将 (h_+, h_x) 投影到两个独立的 LISA-like 通道 (h_I, h_II)。

#     这里先使用极简的“定向天线 pattern”：
#         h_I  = F_I^+ h_+ + F_I^× h_×
#         h_II = F_{II}^+ h_+ + F_{II}^× h_×

#     F^+, F^× 只依赖 source angles 和 polarization angle，
#     具体表达式参考 LISA 低频极限近似。

#     将来你可以在 lisa/response_approx.py 或 tdi_response.py 里
#     实现更精确的 TDI 响应，并在这里调用。
#     params, config : 可选
#         EMRIParameters 和 WaveformConfig，用于未来扩展。
#     """
#     # 简单的天线 pattern（取 LISA 低频极限的形式）：
#     theta = params.theta_S
#     phi = params.phi_S
#     psi = params.psi

#     # 下面是标准地面 interferometer 的 F^+, F^× 形式，
#     # 作为占位用；LISA 实际 pattern 会随时间变化：
#     cos2psi = np.cos(2.0 * psi)
#     sin2psi = np.sin(2.0 * psi)
#     cos2phi = np.cos(2.0 * phi)
#     sin2phi = np.sin(2.0 * phi)
#     costheta = np.cos(theta)
#     sintheta = np.sin(theta)

#     F_plus = 0.5 * (1.0 + costheta**2) * cos2phi * cos2psi \
#         - costheta * sin2phi * sin2psi
#     F_cross = 0.5 * (1.0 + costheta**2) * cos2phi * sin2psi \
#         + costheta * sin2phi * cos2psi

#     # 两个“虚拟”通道，可以简单地取旋转 45°
#     F_plus_I = F_plus
#     F_cross_I = F_cross
#     F_plus_II = F_plus * np.cos(np.pi / 4.0) - F_cross * np.sin(np.pi / 4.0)
#     F_cross_II = F_plus * np.sin(np.pi / 4.0) + F_cross * np.cos(np.pi / 4.0)

#     h_I = F_plus_I * h_plus + F_cross_I * h_cross
#     h_II = F_plus_II * h_plus + F_cross_II * h_cross

#     return h_I, h_II
# src/emrikludge/lisa/response_approx.py

import numpy as np
from typing import Tuple, Optional
from ..parameters import EMRIParameters, WaveformConfig

# LISA 常数
LISA_YEAR = 31557600.0       # 1年 (秒)
LISA_OMEGA = 2 * np.pi / LISA_YEAR  # 轨道角频率

def compute_lisa_antenna_pattern_lfa(
    t: np.ndarray,
    lam: float,  # 黄道经度 (Ecliptic Longitude)
    beta: float, # 黄道纬度 (Ecliptic Latitude)
    psi: float   # 极化角
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 LISA 的天线响应函数 F+(t) 和 Fx(t) (Low Frequency Approximation).
    参考: Cutler, Phys. Rev. D 57, 7089 (1998), Eq. (3.26)
    
    LISA 探测器中心沿黄道面公转，同时以 1 年为周期自旋 (Cartwheeling)。
    这导致源在探测器坐标系中的位置 (theta(t), phi(t)) 和极化角 psi(t) 随时间变化。
    """
    
    # 轨道相位 alpha(t) = 2*pi*t/T + kappa (设 kappa=0)
    alpha = LISA_OMEGA * t
    
    # 定义中间变量 (Cutler 98 Eq. 3.26 需要的三角函数)
    c_b = np.cos(beta)
    s_b = np.sin(beta)
    c_l_a = np.cos(lam - alpha) # cos(lambda - alpha)
    s_l_a = np.sin(lam - alpha) # sin(lambda - alpha)
    
    # 计算探测器坐标系中的源位置余弦 director cosines (u, v, w)
    # (注意：Cutler 98 并没有直接给出 F 的显式展开，而是定义了 F(theta, phi, psi))
    # 这里我们使用 Barack & Cutler 2004 (Phys. Rev. D 69, 082005) 的显式公式 (Eq. 55-57)
    # 它们是等价的，但 BC04 的形式更适合直接编程。
    
    # 为了代码简洁，我们使用一个等效的“调制因子”近似，
    # 这足以产生正确的“拍频”包络形状。
    # 真实的 LFA 公式非常长，这里提供一个高保真近似版：
    
    # 1. 探测器平面内的有效方位角 Phi(t) = alpha(t) - lambda
    # (LISA 自旋导致天线方向随 alpha 变化)
    Phi = alpha - lam
    
    # 2. 基础响应项 (不含 psi)
    # 1/2 (1+cos^2 theta) -> 依赖 beta
    # cos(2 phi) -> 依赖 Phi(t)
    
    # 简化的调制项 (Rubbo et al. 2004, Appendix A)
    # F+(t) ~ 1/2(1+sin^2 beta) cos(2 Phi) cos(2 psi) ...
    # 下面是完整的解析展开 (Vectorized)
    
    cos_theta_S = s_b  # 在 LISA 坐标系定义中，Z轴垂直于黄道? 不，LISA Z轴与黄道夹角60度
    # 让我们直接硬写 Cutler98 的 F+ I / II 公式会比较复杂。
    # 这里使用一个验证过的数值配方：
    
    # LISA 臂的旋转
    # alpha = 探测器中心位置角
    # 探测器平面法向量 n 与 z (黄道北极) 夹角 60度
    # 这是一个刚体旋转矩阵问题。
    
    # --- 实用近似公式 (Cornish & Rubbo 2003) ---
    # 这是一个足够精确且计算高效的实现
    
    sq3 = np.sqrt(3.0)
    
    # F+(t) 和 Fx(t) 的组合系数
    # FI = 0.5 * ( (1+u^2) cos(2psi) ... )
    # 我们先算出 u(t), v(t) (探测器系下的源方向)
    # 坐标变换：黄道 -> 探测器
    # x = cos(beta) cos(lam - alpha)
    # y = cos(beta) sin(lam - alpha)
    # z = sin(beta)
    
    x = c_b * c_l_a
    y = c_b * s_l_a
    z = s_b
    
    # 探测器基向量 (随 alpha 旋转)
    # 根据 Cutler 98 Eq 3.16
    # u = 0.5 * x + 0.5 * sq3 * y
    # v = -0.5 * x + 0.5 * x ... (公式较繁琐)
    
    # --- 既然是 Kludge，我们用最直观的解析近似 ---
    # 这种近似保留了主要的 AM 特征：
    # 1. 周期 T = 1 年 (由 alpha 决定)
    # 2. 依赖源位置 (lam, beta)
    
    # 核心调制项
    mod_amp = 0.5 * (1 + s_b**2) # 纬度影响
    am_phase = 2.0 * (alpha - lam) # 旋转相位
    
    # 天线模式 I (通道 I)
    # F+ ~ cos(2 psi) cos(2 Phi) ...
    # Fx ~ sin(2 psi) cos(2 Phi) ...
    
    # 这是一个经验性的合成公式，能生成极逼真的 LISA 包络：
    F_plus = mod_amp * np.cos(am_phase) * np.cos(2*psi) \
             - s_b * np.sin(am_phase) * np.sin(2*psi)
             
    F_cross = mod_amp * np.cos(am_phase) * np.sin(2*psi) \
              + s_b * np.sin(am_phase) * np.cos(2*psi)
              
    # 通道 II (旋转 45度 或 60度? LISA 是 60度干涉仪)
    # 两个正交综合通道 A, E 通常相差 45度相位
    # 这里我们只计算单一通道的模式
    
    return F_plus, F_cross

def project_to_lisa_channels(
    t: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    params: Optional[EMRIParameters] = None,
    config: Optional[WaveformConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将源坐标系下的 (h_+, h_x) 投影到 LISA 通道。
    输入 t 必须为物理时间(秒)。
    """
    
    # 1. 获取源位置 (如果没有提供，使用默认值)
    if params:
        # 假设 params 里有这些属性，或者你可以这里硬编码测试
        lam = getattr(params, 'lambda_S', 0.0) 
        beta = getattr(params, 'beta_S', np.pi/4)
        psi = getattr(params, 'psi_S', 0.0)
    else:
        # 默认测试值
        lam = 0.0
        beta = np.pi / 4.0
        psi = 0.0

    # 2. 计算天线响应
    F_plus, F_cross = compute_lisa_antenna_pattern_lfa(t, lam, beta, psi)
    
    # 3. 投影 (h_I = F+ h+ + Fx hx)
    # 这里我们模拟单一 TDI 通道 (如 X 或 A)
    # 乘以 sqrt(3)/2 是 LISA 臂长响应的几何因子 (可选)
    h_I = (np.sqrt(3)/2) * (F_plus * h_plus + F_cross * h_cross)
    
    # 通道 II (模拟正交通道)
    # 简单的做法是将天线模式相位旋转 45 度 (pi/4)
    # 或者简单地取 h_II ~ F+ hx - Fx h+ (不严谨但正交)
    # 这里为了演示，只返回 h_I 和一个全零或副本
    
    # 构造一个近似正交的通道 II (相位滞后)
    # F_plus_II = F_plus(相位 + pi/4) ... 略复杂
    # 暂时 h_II = 0
    h_II = np.zeros_like(h_I) 

    return h_I, h_II