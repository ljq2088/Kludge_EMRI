// cpp/emrikludge/waveform/aak_waveform.hpp
#pragma once

#include <vector>
#include <string>

#include "emri_params.hpp"
#include "lisa/lisa_orbit.hpp"
#include "lisa/lisa_response.hpp"
#include "orbit/aak_orbit.hpp"
#include "phase/aak_phase.hpp"

namespace emrikludge {

/// @brief AAK 波形输出容器
///
/// 之所以单独定义一个结构体，是为了以后方便扩展
/// （例如增加额外的通道、频率域输出等），并统一暴露给 Python。
struct AAKWaveformResult {
    std::vector<double> t;       ///< 时间数组
    std::vector<double> hI;      ///< 探测器通道 I (或 A)
    std::vector<double> hII;     ///< 探测器通道 II (或 E)

    // 可选：极化形式
    std::vector<double> hplus;
    std::vector<double> hcross;

    // 可选：轨道信息
    std::vector<double> r;
    std::vector<double> theta;
    std::vector<double> phi;
};

/// @brief 计算 AAK 波形（CPU 版本）
///
/// 高层接口的核心函数：
///  - 输入: EMRIParams + WaveformConfig
///  - 内部调用:
///      1. aak_orbit: 计算轨道随时间的演化
///      2. aak_phase: 计算相位
///      3. lisa_orbit + lisa_response: 把源波形投影到探测器通道
///  - 输出: AAKWaveformResult，其中至少包含 (t, hI, hII)
///
/// 这个函数会在 bindings_aak.cpp 中被暴露给 Python。
AAKWaveformResult
compute_aak_waveform_cpu(const EMRIParams& emri,
                         const WaveformConfig& wf_conf);

} // namespace emrikludge
