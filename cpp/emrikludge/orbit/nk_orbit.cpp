#include "nk_orbit.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// 引入 GSL (如果你的环境有) 或者使用简单的手写 RK45
// 为了代码独立性，这里我们假设使用 GSL ODE 库，或者你可以直接把 Python 的 solve_ivp 替换为 C++ 简单的定步长 RK4
// 但通常为了 python binding 方便，我们先只把 "Mapping" 和 "Flux" 逻辑写好，
// 可以在 Python 端调用 C++ 的 mapping/flux 函数，或者完全在 C++ 端积分。

// 这里为了解决你的燃眉之急，我们先把 "Mapping" 和 "Flux" 的 C++ 逻辑写出来。
// 你可以通过 pybind11 把这些函数暴露给 Python，替换掉 Python 里的慢速函数。

using namespace std;

// =========================================================
// 1. Mapping 模块 (替代 Python 的 get_conserved_quantities)
// =========================================================

double BabakNKOrbit::radial_potential(double r, double M, double a, double E, double Lz, double Q) {
    double Delta = r*r - 2*M*r + a*a;
    double term1 = pow(E*(r*r + a*a) - Lz*a, 2);
    double term2 = Delta * (pow(r*r + pow(Lz - a*E, 2) + Q, 1)); // Q definition varies, here matches Babak Eq 1
    // 注意: Babak Eq 1 定义的 R(r) = term1 - term2
    // 但通常 term2 里的 Q 是 Carter constant.
    // Babak Eq 2a: term2 = Delta * [r^2 + (Lz-aE)^2 + Q]
    return term1 - term2;
}

KerrConstants BabakNKOrbit::get_conserved_quantities(double M, double a, double p, double e, double iota) {
    // 初始猜测 (Weak Field)
    double r_p = p / (1.0 + e);
    double r_a = p / (1.0 - e);
    
    double E = 0.95;
    double Lz = sqrt(p*M) * cos(iota);
    double Q = p*M * pow(sin(iota), 2);
    
    // Newton-Raphson 迭代
    const int MAX_ITER = 50;
    const double TOL = 1e-10;
    
    for(int iter=0; iter<MAX_ITER; ++iter) {
        // 计算残差 F(x)
        // F1 = V(rp) = 0
        // F2 = V(ra) = 0
        // F3 = Q*c2 - L^2*s2 = 0
        
        double Vp = radial_potential(r_p, M, a, E, Lz, Q);
        double Va = radial_potential(r_a, M, a, E, Lz, Q);
        double F_inc = Q * pow(cos(iota), 2) - Lz*Lz * pow(sin(iota), 2);
        
        if (abs(Vp) < TOL && abs(Va) < TOL && abs(F_inc) < TOL) break;
        
        // 计算雅可比 J = dF/dx (解析导数)
        // 变量 x = {E, Lz, Q}
        
        // dV/dE, dV/dL, dV/dQ at rp and ra
        auto get_derivs = [&](double r) -> array<double, 3> {
            double Delta = r*r - 2*M*r + a*a;
            double X = E*(r*r + a*a) - Lz*a;
            // double Y = r*r + pow(Lz - a*E, 2) + Q;
            
            double dV_dE = 2*X*(r*r+a*a) - Delta * 2*(Lz - a*E)*(-a);
            double dV_dL = 2*X*(-a)      - Delta * 2*(Lz - a*E);
            double dV_dQ = 0             - Delta * 1.0;
            return {dV_dE, dV_dL, dV_dQ};
        };
        
        auto dVp = get_derivs(r_p);
        auto dVa = get_derivs(r_a);
        
        // Jacobian Matrix
        double J11 = dVp[0], J12 = dVp[1], J13 = dVp[2];
        double J21 = dVa[0], J22 = dVa[1], J23 = dVa[2];
        double J31 = 0.0,    J32 = -2*Lz*pow(sin(iota), 2), J33 = pow(cos(iota), 2);
        
        // 求解线性方程 J * dx = -F
        // 使用 Cramer 法则求解 3x3 系统 (简单且无依赖)
        double det = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31);
        
        if (abs(det) < 1e-14) break; // Singular
        
        double invDet = 1.0 / det;
        
        // 只需要 dx
        double b1 = -Vp, b2 = -Va, b3 = -F_inc;
        
        double dE = invDet * (b1*(J22*J33 - J23*J32) + b2*(J13*J32 - J12*J33) + b3*(J12*J23 - J13*J22));
        double dL = invDet * (b1*(J23*J31 - J21*J33) + b2*(J11*J33 - J13*J31) + b3*(J13*J21 - J11*J23));
        double dQ = invDet * (b1*(J21*J32 - J22*J31) + b2*(J12*J31 - J11*J32) + b3*(J11*J22 - J12*J21));
        
        // 更新
        // 阻尼牛顿法 (Damped Newton) 防止飞出
        double step = 1.0;
        E += step * dE;
        Lz += step * dL;
        Q += step * dQ;
    }
    
    // 填充结果
    KerrConstants k;
    k.M = M; k.a = a; k.E = E; k.Lz = Lz; k.Q = Q;
    k.r_p = r_p; k.r_a = r_a;
    
    // 辅助根 r3, r4, z+, z- 计算 (同 Python 逻辑)
    double E2 = E*E;
    double inv_1_E2 = 1.0 / (1.0 - E2);
    double S_r = 2*M * inv_1_E2 - (r_p + r_a);
    double P_r = (a*a * Q * inv_1_E2) / (r_p * r_a);
    double delta_r = sqrt(max(0.0, S_r*S_r - 4*P_r));
    k.r3 = (S_r + delta_r)/2.0;
    k.r4 = (S_r - delta_r)/2.0;
    
    k.beta = a*a * (1.0 - E2);
    double term_b = Q + Lz*Lz + a*a*(1.0 - E2);
    double delta_z = sqrt(max(0.0, term_b*term_b - 4*k.beta*Q));
    k.z_plus = (term_b + delta_z)/(2*k.beta);
    k.z_minus = (term_b - delta_z)/(2*k.beta);
    
    return k;
}

// =========================================================
// 2. Flux 模块 (替代 Python 的 nk_fluxes_gg06_2pn)
// =========================================================

NKFluxes BabakNKOrbit::compute_gg06_fluxes(double p, double e, double iota, double a, double M, double mu) {
    // 这里需要把 nk_fluxes.py 里的 _calc_gg06_fluxes_raw 和 Jacobian 逻辑搬过来
    // 由于代码较长，这里展示核心结构
    
    NKFluxes flux;
    
    // 1. 钳位 e
    double safe_e = max(0.0, min(e, 0.999));
    
    // 2. 计算 E_dot, L_dot, Q_dot (Specific)
    // ... 实现 _g_coeffs 和 Flux 公式 ...
    // (此处需将 Python 公式逐行翻译为 C++)
    
    // 3. 计算 Jacobian (解析法)
    // 复用上面的 get_derivs 逻辑构建矩阵 A
    // 构建矩阵 B (对 p, e, iota 的导数)
    // 求解 J * dx = dy
    
    // 暂时返回占位符，你需要把 Python 代码中的公式填入这里
    flux.dp_dt = 0.0; // placeholder
    flux.de_dt = 0.0;
    flux.diota_dt = 0.0;
    
    return flux;
}

// =========================================================
// 3. 积分器接口
// =========================================================

BabakNKOrbit::BabakNKOrbit(double M, double a, double p, double e, double iota, double mu)
    : M_phys(M), a_spin(a), p0(p), e0(e), iota0(iota), mu_phys(mu) {
    do_inspiral = (mu > 0.0);
}

// 如果你不想手写 RK45，可以先只把上面的 Mapping 函数暴露给 Python
// 这样 Python 的积分器就能快很多