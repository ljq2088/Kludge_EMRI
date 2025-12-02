#pragma once
#include <vector>
#include "nk_orbit.hpp" // 复用基础结构
#include "../emri_params.hpp"

namespace emrikludge {

// AAK 轨迹状态：包含物理参数和平直时空代理参数
struct AAKState {
    double t;
    double p_ak;    // AK Map 后的 p
    double e_ak;    // AK Map 后的 e
    double iota_ak; // AK Map 后的 iota
    double Phi_r;   // 精确相位
    double Phi_theta;
    double Phi_phi;
    double Omega_phi;
};

class BabakAAKOrbit {
public:
    BabakAAKOrbit(double M, double a, double p, double e, double iota, double mu)
        : M_phys(M), a_spin(a), p0(p), e0(e), iota0(iota), mu_phys(mu) {}

    // 演化函数
    std::vector<AAKState> evolve(double duration, double dt);

private:
    double M_phys;
    double a_spin;
    double p0, e0, iota0;
    double mu_phys;

    // 内部相位累积
    double m_Phi_r = 0.0;
    double m_Phi_theta = 0.0;
    double m_Phi_phi = 0.0;
};

} // namespace