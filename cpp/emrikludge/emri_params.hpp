// cpp/emrikludge/emri_params.hpp
#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <string>
namespace emrikludge {

// 1. 唯一定义 KerrConstants
struct KerrConstants {
    double E;
    double Lz;
    double Q;
    double r3;
    double r4;
    double r_p;
    double r_a;
    double z_minus;
    double z_plus;
    double beta;
};

// 2. EMRIParams 定义
struct EMRIParams {
    double M;
    double mu;
    double a;
    double p0;
    double e0;
    double iota0;
    double thetaS;
    double phiS;
    double thetaK;
    double phiK;
    double dist;
    double Phi_phi0;
    double Phi_r0;
    double Phi_theta0;
    double T;
    double dt;
    bool use_eccentric;
    bool use_equatorial;

    EMRIParams(); // 构造函数声明
};

// 3. WaveformConfig 定义
struct WaveformConfig {
    bool return_polarizations;
    bool return_tdi_channels;
    bool return_orbit;
    std::string tdi_mode;

    WaveformConfig(); // 构造函数声明
};
// =======================================================
// 🛡️ AAK 占位符定义 (Placeholder Definitions)
// 为了让 AAK 模块能通过编译，先定义空结构体。
// 后续实现 AAK 时再完善或移到独立文件中。
// =======================================================

struct AAKOrbitTrajectory {
    // 暂时留空，或者是加一些占位成员防止 unused warning
    int _placeholder; 
};

struct LISAOrbit {
    int _placeholder;
};

struct LISAResponse {
    int _placeholder;
};

// 简单的 AAK 返回结果结构体 (如果之前没定义的话)
struct AAKWaveformResult {
    std::vector<double> t;
    std::vector<double> hplus;
    std::vector<double> hcross;
    // 还有 X, Y, Z 等 TDI 通道
};
} // namespace emrikludge