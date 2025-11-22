// cpp/emrikludge/orbit/nk_orbit.hpp
#pragma once
#include <vector>
#include <cmath>
#include <array>

// 【关键修复】引入上级目录的 emri_params.hpp 以获取 KerrConstants 定义
#include "../emri_params.hpp"

namespace emrikludge {
struct NKFluxes {
        double dE_dt;
        double dLz_dt;
        double dQ_dt;
        double dp_dt;
        double de_dt;
        double diota_dt;
    };
// 简单的状态结构体 (仅在 Orbit 内部使用，也可以移到 emri_params)
struct OrbitState {
    double t;
    double p;
    double e;
    double iota;
    double psi;
    double chi;
    double phi;
    double r; 
    double theta;
};

class BabakNKOrbit {
public:
    BabakNKOrbit(double M, double a, double p, double e, double iota, double mu);
    // 如果传入 0，则使用内部的 Schwarzschild 猜测
    static KerrConstants get_conserved_quantities(
        double M, double a, double p, double e, double iota,
        double E_g = 0.0, double L_g = 0.0, double Q_g = 0.0
    );

    NKFluxes compute_gg06_fluxes(double p, double e, double iota, double a, double M, double mu);
    std::vector<OrbitState> evolve(double duration, double dt);

    // 获取当前状态 (用于调试或断点保存)
    OrbitState get_current_state() const;

private:
    double M_phys; 
    double mu_phys;
    double a_spin;
    double p0, e0, iota0;
    bool do_inspiral;

    double m_t;
    double m_p, m_e, m_iota;
    double m_psi, m_chi, m_phi;

    KerrConstants k_cached; 

    static double radial_potential(double r, double M, double a, double E, double Lz, double Q);
    static double radial_potential_deriv(double r, double M, double a, double E, double Lz, double Q);
};

} // namespace emrikludge