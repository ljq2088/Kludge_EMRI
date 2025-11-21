#pragma once
#include <vector>
#include <cmath>
#include <array>

// 定义一些常数
const double PI = 3.14159265358979323846;

struct KerrConstants {
    double M;
    double a;
    double E;
    double Lz;
    double Q;
    // 辅助变量
    double beta;
    double z_minus;
    double z_plus;
    double r_p;
    double r_a;
    double r3;
    double r4;
};

struct OrbitState {
    double t;
    double p;
    double e;
    double iota;
    double psi;
    double chi;
    double phi;
};

struct NKFluxes {
    double dE_dt;
    double dLz_dt;
    double dQ_dt;
    double dp_dt;
    double de_dt;
    double diota_dt;
};

class BabakNKOrbit {
public:
    BabakNKOrbit(double M, double a, double p, double e, double iota, double mu);
    
    // 演化函数 (暴露给 Python)
    std::vector<OrbitState> evolve(double duration, double dt);

private:
    double M_phys; // 物理质量
    double mu_phys;
    double a_spin;
    double p0, e0, iota0;
    bool do_inspiral;

    // 核心导数函数 (RHS)
    static void equations_of_motion(const double t, const double y[], double dydt[], void* params);

    // --- 内部核心算法 (C++ 实现) ---
    
    // 1. Mapping: 手写牛顿法求解 (E, L, Q)
    static KerrConstants get_conserved_quantities(double M, double a, double p, double e, double iota);
    
    // 2. Fluxes: 计算 GG06 2PN 通量
    static NKFluxes compute_gg06_fluxes(double p, double e, double iota, double a, double M, double mu);
    
    // 辅助计算函数
    static double radial_potential(double r, double M, double a, double E, double Lz, double Q);
    static void compute_roots_and_z(double M, double a, double p, double e, double iota, KerrConstants& k);
};