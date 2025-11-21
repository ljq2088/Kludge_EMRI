#include "nk_orbit.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace std;
namespace emrikludge{
// 构造函数
BabakNKOrbit::BabakNKOrbit(double M, double a, double p, double e, double iota, double mu)
    : M_phys(M), a_spin(a), p0(p), e0(e), iota0(iota), mu_phys(mu) {
    do_inspiral = (mu > 0.0);
}

// 径向势 V_r
double BabakNKOrbit::radial_potential(double r, double M, double a, double E, double Lz, double Q) {
    double Delta = r*r - 2*M*r + a*a;
    double term1 = pow(E*(r*r + a*a) - Lz*a, 2);
    double term2 = Delta * (pow(r*r + pow(Lz - a*E, 2) + Q, 1)); 
    return term1 - term2;
}

// 径向势导数 (用于 e=0 情况)
double BabakNKOrbit::radial_potential_deriv(double r, double M, double a, double E, double Lz, double Q) {
    double eps = 1e-5 * r;
    double v_plus = radial_potential(r + eps, M, a, E, Lz, Q);
    double v_minus = radial_potential(r - eps, M, a, E, Lz, Q);
    return (v_plus - v_minus) / (2 * eps);
}

// --- 核心 Mapping 函数 ---
KerrConstants BabakNKOrbit::get_conserved_quantities(double M, double a, double p, double e, double iota) {
    // 1. 准备变量
    // 强制 M=1 (几何单位)
    double M_code = 1.0; 
    // 输入的 p 假设已经是 p/M
    
    double r_p = p / (1.0 + e);
    double r_a = p / (1.0 - e);
    
    // 初始猜测 (Weak Field Approximation)
    double E = 0.93;
    double Lz = sqrt(p) * cos(iota);
    double Q = p * pow(sin(iota), 2);
    
    // Newton-Raphson 迭代配置
    const int MAX_ITER = 100;
    const double TOL = 1e-10;
    
    bool success = false;

    for(int iter=0; iter<MAX_ITER; ++iter) {
        // 计算残差 F(x)
        double F1, F2;
        
        // 分支：圆轨道 vs 偏心轨道
        if (e < 1e-4) {
            // 圆轨道: V(p)=0, V'(p)=0
            F1 = radial_potential(p, M_code, a, E, Lz, Q);
            F2 = radial_potential_deriv(p, M_code, a, E, Lz, Q);
        } else {
            // 偏心轨道: V(rp)=0, V(ra)=0
            F1 = radial_potential(r_p, M_code, a, E, Lz, Q);
            F2 = radial_potential(r_a, M_code, a, E, Lz, Q);
        }
        // 倾角条件: Q*cos^2(i) - Lz^2*sin^2(i) = 0
        double F3 = Q * pow(cos(iota), 2) - Lz*Lz * pow(sin(iota), 2);
        
        if (abs(F1) < TOL && abs(F2) < TOL && abs(F3) < TOL) {
            success = true;
            break;
        }
        
        // --- 计算 Jacobian J = dF/dx (解析导数) ---
        // 变量 x = {E, Lz, Q}
        // 辅助 lambda: 计算 dV/dE, dV/dL, dV/dQ
        auto get_V_derivs = [&](double r) -> array<double, 3> {
            double Delta = r*r - 2*M_code*r + a*a;
            double X = E*(r*r + a*a) - Lz*a;
            double Y = r*r + pow(Lz - a*E, 2) + Q;
            
            // dV/dE = 2X(r^2+a^2) - Delta * 2(L-aE)(-a)
            double dV_dE = 2*X*(r*r+a*a) + 2*a*Delta*(Lz - a*E);
            // dV/dL = 2X(-a) - Delta * 2(L-aE)
            double dV_dL = -2*a*X - 2*Delta*(Lz - a*E);
            // dV/dQ = -Delta
            double dV_dQ = -Delta;
            return {dV_dE, dV_dL, dV_dQ};
        };
        
        // 简单起见，对于圆轨道导数的雅可比，我们用数值差分，
        // 对于偏心轨道用解析导数。为了代码简洁，这里全部用中心差分计算 Jacobian。
        // 这样代码量最少且不容易写错公式。
        
        double dE = 1e-5, dL = 1e-5, dQ = 1e-5;
        
        auto eval_F = [&](double tE, double tL, double tQ) -> array<double, 3> {
            double tF1, tF2;
            if (e < 1e-4) {
                tF1 = radial_potential(p, M_code, a, tE, tL, tQ);
                tF2 = radial_potential_deriv(p, M_code, a, tE, tL, tQ);
            } else {
                tF1 = radial_potential(r_p, M_code, a, tE, tL, tQ);
                tF2 = radial_potential(r_a, M_code, a, tE, tL, tQ);
            }
            double tF3 = tQ * pow(cos(iota), 2) - tL*tL * pow(sin(iota), 2);
            return {tF1, tF2, tF3};
        };
        
        // Column 1: dF/dE
        auto F_Ep = eval_F(E+dE, Lz, Q);
        auto F_Em = eval_F(E-dE, Lz, Q);
        double J11 = (F_Ep[0]-F_Em[0])/(2*dE), J21 = (F_Ep[1]-F_Em[1])/(2*dE), J31 = (F_Ep[2]-F_Em[2])/(2*dE);
        
        // Column 2: dF/dL
        auto F_Lp = eval_F(E, Lz+dL, Q);
        auto F_Lm = eval_F(E, Lz-dL, Q);
        double J12 = (F_Lp[0]-F_Lm[0])/(2*dL), J22 = (F_Lp[1]-F_Lm[1])/(2*dL), J32 = (F_Lp[2]-F_Lm[2])/(2*dL);
        
        // Column 3: dF/dQ
        auto F_Qp = eval_F(E, Lz, Q+dQ);
        auto F_Qm = eval_F(E, Lz, Q-dQ);
        double J13 = (F_Qp[0]-F_Qm[0])/(2*dQ), J23 = (F_Qp[1]-F_Qm[1])/(2*dQ), J33 = (F_Qp[2]-F_Qm[2])/(2*dQ);
        
        // 求解 J * delta = -F
        double det = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31);
        if (abs(det) < 1e-15) break;
        
        double invDet = 1.0/det;
        double b1 = -F1, b2 = -F2, b3 = -F3;
        
        double step_E = invDet * (b1*(J22*J33 - J23*J32) + b2*(J13*J32 - J12*J33) + b3*(J12*J23 - J13*J22));
        double step_L = invDet * (b1*(J23*J31 - J21*J33) + b2*(J11*J33 - J13*J31) + b3*(J13*J21 - J11*J23));
        double step_Q = invDet * (b1*(J21*J32 - J22*J31) + b2*(J12*J31 - J11*J32) + b3*(J11*J22 - J12*J21));
        
        // 阻尼更新
        double alpha = 1.0;
        E += alpha * step_E;
        Lz += alpha * step_L;
        Q += alpha * step_Q;
    }
    
    if (!success) {
        // 如果失败，返回全0或抛出异常
        // 为了不让 Python 崩溃，返回一个标记值
        return {0,0,0,0,0,0,0,0,0,0};
    }
    
    // 2. 计算辅助变量 (r3, r4, z+, z-)
    // 这里直接复用 Python nk_mapping.py 中的逻辑
    double E2 = E*E;
    double inv_1_E2 = 1.0 / (1.0 - E2);
    double S_r = 2*M_code * inv_1_E2 - (r_p + r_a);
    
    // 保护分母
    double denom = r_p * r_a;
    if (denom < 1e-8) denom = 1e-8;
    double P_r = (a*a * Q * inv_1_E2) / denom;
    
    double delta_r = sqrt(max(0.0, S_r*S_r - 4*P_r));
    double r3 = (S_r + delta_r)/2.0;
    double r4 = (S_r - delta_r)/2.0;
    if (r4 > r3) swap(r3, r4);
    
    double beta = a*a * (1.0 - E2);
    double term_b = Q + Lz*Lz + a*a*(1.0 - E2);
    double delta_z = 0.0;
    double z_plus = 0.0, z_minus = 0.0;
    
    if (abs(beta) > 1e-12) {
        delta_z = sqrt(max(0.0, term_b*term_b - 4*beta*Q));
        z_plus = (term_b + delta_z)/(2*beta);
        z_minus = (term_b - delta_z)/(2*beta);
    }
    
    return {E, Lz, Q, r3, r4, r_p, r_a, z_minus, z_plus, beta};
}

// 占位符：演化函数暂时返回空，我们目前只用 Mapping 加速
std::vector<OrbitState> BabakNKOrbit::evolve(double duration, double dt) {
    return {};
}
}