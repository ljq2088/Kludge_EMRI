#pragma once
#include <vector>
#include <cmath>

struct KerrConstants {
    double M;
    double a;
    double E;
    double Lz;
    double Q;
    // 预计算的辅助变量
    double beta;
    double z_minus;
    double z_plus;
    double r_p;
    double r_a;
    // r3, r4 等根
};

struct OrbitState {
    double t;
    double psi;
    double chi;
    double phi;
};

class BabakNKOrbit {
public:
    BabakNKOrbit(double M, double a, double p, double e, double iota);
    
    // 核心导数函数，设计为纯函数，方便迁移到 CUDA kernel
    static void derivatives(const double t, const double y[], double dydt[], const KerrConstants& params);
    
    // 演化函数
    std::vector<OrbitState> evolve(double duration, double dt);

private:
    KerrConstants m_params;
};