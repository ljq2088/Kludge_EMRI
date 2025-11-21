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

} // namespace emrikludge