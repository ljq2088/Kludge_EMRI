// cpp/emrikludge/emri_params.hpp
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "constants.hpp"

namespace emrikludge {

/// @brief EMRI 系统基本参数（源物理量 + 轨道参数）
///
/// 这个结构体会从 Python 层的 EMRIParameters dataclass 映射而来，
/// 再被 aak_orbit / nk_orbit / ak_orbit / aak_waveform 等模块使用。
struct EMRIParams {
    // ----------- 源质量与自旋 -----------
    double M;        ///< 中心黑洞质量 (单位: M_sun 或代码统一单位)
    double mu;       ///< 小天体质量
    double a;        ///< Kerr 自旋参数 a = J / M

    // ----------- 轨道参数 -----------
    double p0;       ///< 初始半长轴相关参数 (通常是半长径 p)
    double e0;       ///< 初始偏心率
    double iota0;    ///< 初始轨道倾角 (弧度制)

    // 也可以视情况添加 (E,Lz,Q) 等常量的等效输入，如果 AAK/NK 需要

    // ----------- 源在天球上的位置 -----------
    double thetaS;   ///< 源天球坐标: 极角 (弧度)
    double phiS;     ///< 源天球坐标: 方位角 (弧度)

    // ----------- 探测器参考系参数 -----------
    double thetaK;   ///< BH 自旋方向在天球上的极角
    double phiK;     ///< BH 自旋方向在天球上的方位角
    double dist;     ///< 源到探测器的距离 (Mpc 或代码单位)

    // ----------- 初始相位 -----------
    double Phi_phi0; ///< 方位方向初始相位
    double Phi_r0;   ///< 径向方向初始相位
    double Phi_theta0; ///< 极向方向初始相位

    // ----------- 波形采样相关 -----------
    double T;        ///< 总观测时长 (秒)
    double dt;       ///< 采样时间步长 (秒)

    bool use_eccentric; ///< 是否使用偏心轨道 (AAK/NK 配置相关)
    bool use_equatorial;///< 是否限制为赤道轨道

    // 可以根据原实现添加更多字段，例如: Taper、窗口函数类型等

    EMRIParams();
};

/// @brief 波形输出配置（AAK/NK 都可以共享）
///
/// 与 Python 层 WaveformConfig 对应，用于控制返回哪些通道、
/// 是否返回轨道信息等。
struct WaveformConfig {
    bool return_polarizations; ///< 是否返回 h_plus, h_cross
    bool return_tdi_channels;  ///< 是否直接计算并返回 TDI (X,Y,Z)
    bool return_orbit;         ///< 是否返回轨道 (r(t), theta(t), phi(t))

    std::string tdi_mode;      ///< "none", "X", "XYZ", "AE", "AET" 等

    WaveformConfig();
};

} // namespace emrikludge
