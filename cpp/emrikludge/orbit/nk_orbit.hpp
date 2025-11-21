#pragma once
#include <vector>
#include <cmath>
#include <array>

// 定义与 Python dataclass 对应的 C++ 结构体
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

// 简单的状态结构体
struct OrbitState {
    double t;
    double p;
    double e;
    double iota;
    double psi;
    double chi;
    double phi;
    double r; // r_over_M
    double theta;
};

class BabakNKOrbit {
public:
    BabakNKOrbit(double M, double a, double p, double e, double iota, double mu);
    
    // 暴露给 Python 的 Mapping 函数 (静态函数，方便独立调用)
    static KerrConstants get_conserved_quantities(double M, double a, double p, double e, double iota);

    // 演化函数
    std::vector<OrbitState> evolve(double duration, double dt);

private:
    double M_phys; 
    double mu_phys;
    double a_spin;
    double p0, e0, iota0;
    bool do_inspiral;
    KerrConstants k_cached; // 缓存的常数

    // 内部辅助函数
    static double radial_potential(double r, double M, double a, double E, double Lz, double Q);
    static double radial_potential_deriv(double r, double M, double a, double E, double Lz, double Q);
};