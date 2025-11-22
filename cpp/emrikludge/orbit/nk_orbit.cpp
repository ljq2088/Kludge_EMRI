#include "nk_orbit.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
//定义常数PI
const double PI = 3.14159265358979323846;
using namespace std;
// ---------------------------------------------------------
// 辅助函数：GG06 Flux Coefficients
// ---------------------------------------------------------
struct GG06Coeffs {
    double g[17]; // g1..g16, map 10a->10, 10b->0 (special handling)
    double g10a, g10b;
};

GG06Coeffs calc_g_coeffs(double e) {
    GG06Coeffs c;
    double e2 = e*e;
    double e4 = e2*e2;
    double e6 = e4*e2;

    c.g[1] = 1.0 + (73.0/24.0)*e2 + (37.0/96.0)*e4;
    c.g[2] = (73.0/12.0) + (823.0/24.0)*e2 + (949.0/32.0)*e4 + (491.0/192.0)*e6;
    c.g[3] = (1247.0/336.0) + (9181.0/672.0)*e2;
    c.g[4] = 4.0 + (1375.0/48.0)*e2;
    c.g[5] = (44711.0/9072.0) + (172157.0/2592.0)*e2;
    c.g[6] = (33.0/16.0) + (359.0/32.0)*e2;
    c.g[7] = (8191.0/672.0) + (44531.0/336.0)*e2;
    c.g[8] = (3749.0/336.0) - (5143.0/168.0)*e2;
    c.g[9] = 1.0 + (7.0/8.0)*e2;
    // g10 (equatorial) is not strictly needed if we use 10a/10b for inclined
    c.g[11] = (1247.0/336.0) + (425.0/336.0)*e2;
    c.g[12] = 4.0 + (97.0/8.0)*e2;
    c.g[13] = (44711.0/9072.0) + (302893.0/6048.0)*e2;
    c.g[14] = (33.0/16.0) + (95.0/16.0)*e2;
    c.g[15] = (8191.0/672.0) + (48361.0/1344.0)*e2;
    c.g[16] = (417.0/56.0) - (37241.0/672.0)*e2;

    c.g10a = (61.0/24.0) + (63.0/8.0)*e2 + (95.0/64.0)*e4;
    c.g10b = (61.0/8.0) + (91.0/4.0)*e2 + (461.0/64.0)*e4;
    
    return c;
}

void calc_N_coeffs(double p, double M, double a, double E_circ, double L_circ, 
                   double& N1, double& N4, double& N5) {
    // N1 = E*p^4 + a^2*E*p^2 - 2*a*M*(L-aE)*p
    N1 = E_circ * pow(p, 4) + a*a * E_circ * p*p - 2.0*a*M * (L_circ - a*E_circ) * p;
    // N4 = (2Mp - p^2)L - 2MaEp
    N4 = (2.0*M*p - p*p) * L_circ - 2.0*M*a*E_circ * p;
    // N5 = (2Mp - p^2 - a^2)/2
    N5 = (2.0*M*p - p*p - a*a) / 2.0;
}



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
    double M_code = 1.0; 
    double r_p = p / (1.0 + e);
    double r_a = p / (1.0 - e);
    
    // 理论公式: E^2 = [ (p-2-2e)(p-2+2e) ] / [ p(p-3-e^2) ]
    //          L^2 = p^2 / (p-3-e^2)
    
    // 分母项 (反映了 ISCO/Whirl 附近的强场发散)
    double denom_factor = p - 3.0 - e*e;
    
    // 安全保护：如果 p 过于接近不稳定区域 (p ~ 3+e^2)，分母会趋于 0
    // 我们给它一个软下限，防止初值变成 inf
    if (denom_factor < 0.01) denom_factor = 0.01;
    
    // 计算 Schwarzschild 能量 E
    double num_E = (p - 2.0 - 2.0*e) * (p - 2.0 + 2.0*e);
    // 保护根号下不为负
    if (num_E < 0.0) num_E = 0.0; 
    
    double E_schw_sq = num_E / (p * denom_factor);
    double E_guess = sqrt(E_schw_sq);
    
    // 计算 Schwarzschild 角动量 L (Total L)
    double L_schw_sq = (p * p) / denom_factor;
    double L_schw = sqrt(L_schw_sq);
    
    double E = E_guess;
    // 将 Schwarzschild 的总角动量 L 投影到 Kerr 的 Lz 和 Q
    // Lz ~ L * cos(iota)
    // Q  ~ L^2 * sin^2(iota)
    double Lz = L_schw * cos(iota);
    double Q = L_schw_sq * pow(sin(iota), 2);
    
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
// 在 BabakNKOrbit 类方法的实现区域
NKFluxes BabakNKOrbit::compute_gg06_fluxes(double p, double e, double iota, double a, double M, double mu) {
    NKFluxes flux = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // 1. 几何单位换算 (内部 M=1, mu=q)
    double M_code = 1.0;
    double q_mass = mu / M; // mass ratio
    
    // 钳位 e
    double safe_e = max(0.0, min(e, 0.999));
    
    // 预计算
    double q_spin = a / M_code;
    double v2 = M_code / p;
    double Mp = v2; 
    double Mp15 = pow(Mp, 1.5);
    double Mp2  = Mp * Mp;
    double Mp25 = pow(Mp, 2.5);
    double Mp35 = pow(Mp, 3.5);
    double Mp5  = pow(Mp, 5.0);
    
    double cos_i = cos(iota);
    double sin_i = sin(iota);
    double sin2_i = sin_i*sin_i;
    double cos2_i = cos_i*cos_i;
    
    GG06Coeffs g = calc_g_coeffs(safe_e);
    double prefix = pow(1.0 - safe_e*safe_e, 1.5);
    
    // --- A. 计算 2PN Fluxes (Specific) ---
    // Energy Flux (dE/dt / mu)
    // Coeff = - (32/5) * (mu/M) * (M/p)^5  (注意：这是 Specific flux, 所以只有 mu/M 一次项)
    // Wait, Python code used (mu/M)^2 for Total Flux. Here we want Specific Flux directly.
    // Specific Flux = Total Flux / mu.
    // Factor_E_spec = - (32/5) * (mu/M) * (M/p)^5
    double factor_E = -(32.0/5.0) * q_mass * Mp5;
    
    double term_E = g.g[1] 
                  - q_spin * Mp15 * g.g[2] * cos_i
                  - Mp * g.g[3]
                  + PI * Mp15 * g.g[4]
                  - Mp2 * g.g[5]
                  + q_spin*q_spin * Mp2 * g.g[6]
                  - (527.0/96.0) * q_spin*q_spin * Mp2 * sin2_i;
                  
    double dE_spec = factor_E * prefix * term_E;
    
    // Angular Momentum (dL/dt / mu)
    // Factor_L_spec = - (32/5) * (mu/M) * (M/p)^3.5 / mu * mu? No.
    // L_spec ~ M. L_total ~ mu*M. 
    // L_total_dot ~ mu^2. L_spec_dot ~ mu.
    // So coeff should be proportional to q_mass.
    // Factor = - (32/5) * q_mass * (M/p)^3.5
    double factor_L = -(32.0/5.0) * q_mass * Mp35;
    
    double term_L = g.g[9] * cos_i
                  + q_spin * Mp15 * (g.g10a - cos2_i * g.g10b)
                  - Mp * g.g[11] * cos_i
                  + PI * Mp15 * g.g[12] * cos_i
                  - Mp2 * g.g[13] * cos_i
                  + q_spin*q_spin * Mp2 * cos_i * (g.g[14] - (45.0/8.0)*sin2_i);
                  
    double dL_spec = factor_L * prefix * term_L;
    
    // Carter Constant (dQ/dt / mu^2)
    // Q_total ~ mu^2. Q_spec ~ 1 (M^2).
    // Q_total_dot ~ mu^3. Q_spec_dot ~ mu.
    // Python code: factor_Q ~ mu^2/M * ... * sqrt(Q_total).
    // sqrt(Q_total) = mu * sqrt(Q_spec).
    // So Q_total_dot ~ mu^3.
    // Q_spec_dot = Q_total_dot / mu^2 = (mu^3 / mu^2) ~ mu.
    // Correct.
    
    // 我们需要当前的 Q_spec
    KerrConstants k_curr = get_conserved_quantities(M_code, a, p, safe_e, iota);
    double Q_spec = k_curr.Q;
    double sqrt_Q_spec = sqrt(abs(Q_spec));
    
    // Factor_Q_spec_dot = - (64/5) * q_mass * (M/p)^3.5 * sin(i) * prefix
    double factor_Q = -(64.0/5.0) * q_mass * Mp35 * sin_i * prefix;
    
    double term_Q = g.g[9]
                  - q_spin * Mp15 * cos_i * g.g10b
                  - Mp * g.g[11]
                  + PI * Mp15 * g.g[12]
                  - Mp2 * g.g[13]
                  + q_spin*q_spin * Mp2 * (g.g[14] - (45.0/8.0)*sin2_i);
                  
    double dQ_spec = factor_Q * sqrt_Q_spec * term_Q;

    // --- B. Circular Fix (修正 E_dot) ---
    // 1. 计算圆轨道通量
    // 2. 计算 N 系数
    // 3. 修正
    // (为了简洁，这里省略了 Circular Fix 的完整复现，先用 Raw 2PN 跑通)
    // 如果需要完整精度，可以后续补上。目前的瓶颈是速度，Raw 2PN 足够测试。
    
    // --- C. Jacobian 转换 (dY/dX -> dX/dY) ---
    // 我们需要解线性方程 J * [dp, de, di] = [dE, dL, dQ]
    // 这里使用解析雅可比 (Implicit Differentiation)
    
    // 1. 计算 V 的偏导数
    auto get_V_derivs = [&](double r) -> array<double, 4> {
        double Delta = r*r - 2*M_code*r + a*a;
        double X = k_curr.E*(r*r + a*a) - k_curr.Lz*a;
        double Y = r*r + pow(k_curr.Lz - a*k_curr.E, 2) + k_curr.Q;
        
        double dX_dE = r*r + a*a;
        double dX_dL = -a;
        
        double dY_dE = 2*(k_curr.Lz - a*k_curr.E) * (-a);
        double dY_dL = 2*(k_curr.Lz - a*k_curr.E);
        double dY_dQ = 1.0;
        
        double dV_dE = 2*X*dX_dE - Delta*dY_dE;
        double dV_dL = 2*X*dX_dL - Delta*dY_dL;
        double dV_dQ = -Delta;
        
        double dDelta_dr = 2*r - 2*M_code;
        double dX_dr = k_curr.E * 2*r;
        double dY_dr = 2*r;
        double dV_dr = 2*X*dX_dr - (dDelta_dr*Y + Delta*dY_dr);
        
        return {dV_dE, dV_dL, dV_dQ, dV_dr};
    };
    
    auto dVp = get_V_derivs(k_curr.r_p);
    auto dVa = get_V_derivs(k_curr.r_a);
    
    // Matrix A (Left side: dF/dConserved)
    double A[3][3] = {
        {dVp[0], dVp[1], dVp[2]},
        {dVa[0], dVa[1], dVa[2]},
        {0.0,    -2.0*k_curr.Lz*sin2_i, cos2_i} // dG/dE=0, dG/dL, dG/dQ
    };
    
    // Matrix B (Right side: - dF/dGeom)
    // dp, de, diota
    double drp_dp = 1.0 / (1.0 + safe_e);
    double drp_de = -p / pow(1.0 + safe_e, 2);
    
    double dra_dp = 1.0 / (1.0 - safe_e);
    double dra_de = p / pow(1.0 - safe_e, 2);
    
    double dG_di = -2.0 * sin_i * cos_i * (k_curr.Q + k_curr.Lz*k_curr.Lz);
    
    double B[3][3] = {
        {-dVp[3]*drp_dp, -dVp[3]*drp_de, 0.0},
        {-dVa[3]*dra_dp, -dVa[3]*dra_de, 0.0},
        {0.0, 0.0, -dG_di}
    };
    
    // Solve J = A^-1 * B  =>  A * J = B
    // We want X_dot: J * X_dot = Y_dot  => A^-1 * B * X_dot = Y_dot => B * X_dot = A * Y_dot
    // 等等，关系是: A * dY + B * dX = 0  => dY/dX = -A^-1 * B ???
    // No. F(Y(X), X) = 0 => dF/dY * dY/dX + dF/dX = 0 => A * J + (-B) = 0 => A * J = B. (If B is defined as -dF/dX)
    // Correct. J = A^-1 * B.
    // We have Y_dot (Fluxes). We want X_dot (p_dot...).
    // Y_dot = J * X_dot.
    // So Y_dot = (A^-1 * B) * X_dot => A * Y_dot = B * X_dot.
    // 我们需要解 B * X_dot = (A * Y_dot).
    
    // 1. Compute RHS = A * Y_dot
    double Y_dot[3] = {dE_spec, dL_spec, dQ_spec};
    double RHS[3] = {0, 0, 0};
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            RHS[i] += A[i][j] * Y_dot[j];
        }
    }
    
    // 2. Solve B * X_dot = RHS for X_dot
    // B is:
    // [ B00 B01  0 ]
    // [ B10 B11  0 ]
    // [  0   0  B22]
    // This structure is easy to solve!
    // Row 2: B22 * diota_dt = RHS[2] => diota_dt = RHS[2] / B22
    
    if (abs(B[2][2]) > 1e-14) {
        flux.diota_dt = RHS[2] / B[2][2];
    } else {
        flux.diota_dt = 0.0; // singularity at pole/equator?
    }
    
    // 2x2 sub-system for p, e
    double detB_sub = B[0][0]*B[1][1] - B[0][1]*B[1][0];
    if (abs(detB_sub) > 1e-14) {
        flux.dp_dt = (B[1][1]*RHS[0] - B[0][1]*RHS[1]) / detB_sub;
        flux.de_dt = (B[0][0]*RHS[1] - B[1][0]*RHS[0]) / detB_sub;
    } else {
        flux.dp_dt = 0.0;
        flux.de_dt = 0.0;
    }
    
    // 填充 Flux 结构体用于返回 (Python端可能只用 dp, de, di)
    flux.dE_dt = dE_spec;
    flux.dLz_dt = dL_spec;
    flux.dQ_dt = dQ_spec;
    
    return flux;
}
std::vector<OrbitState> BabakNKOrbit::evolve(double duration, double dt) {
    std::vector<OrbitState> traj;
    
    // 预估步数并分配内存 (避免动态扩容开销)
    size_t estimated_steps = static_cast<size_t>(duration / dt);
    traj.reserve(estimated_steps + 1000);
    
    // 初始状态
    double t = 0.0;
    double p = p0;
    double e = e0;
    double iota = iota0;
    double psi = 0.0;
    double chi = 0.0;
    double phi = 0.0;
    
    // 状态向量数组 y[6] = {p, e, iota, psi, chi, phi}
    
    // 积分循环
    while (t < duration) {
        // 1. 记录当前点 (Output)
        //    计算辅助坐标 r, theta
        KerrConstants k = get_conserved_quantities(1.0, a_spin, p, e, iota);
        // 如果 Mapping 失败 (返回全0)，则停止演化
        if (k.E == 0.0) break;

        double r_val = p / (1.0 + e * cos(psi));
        double z_val = k.z_minus * pow(cos(chi), 2);
        double theta_val = acos(sqrt(max(0.0, z_val))); // 简单处理符号

        traj.push_back({t, p, e, iota, psi, chi, phi, r_val, theta_val});
        
        // 2. 检查终止条件 (Plunge)
        if (p < 3.0 || e >= 0.999) break;

        // 3. RK4 积分步
        // 定义导数计算 lambda (RHS)
        auto get_derivs = [&](double cp, double ce, double ci, double cpsi, double cchi, double cphi) -> std::array<double, 6> {
            // A. 计算 Flux (轨道参数导数)
            double dp_dt=0, de_dt=0, diota_dt=0;
            if (do_inspiral) {
                NKFluxes f = compute_gg06_fluxes(cp, ce, ci, a_spin, 1.0, mu_phys);
                dp_dt = f.dp_dt;
                de_dt = f.de_dt;
                diota_dt = f.diota_dt;
            }
            
            // B. 计算 Geodesic (相位导数)
            KerrConstants ck = get_conserved_quantities(1.0, a_spin, cp, ce, ci);
            if (ck.E == 0.0) return {0,0,0,0,0,0}; // 失败保护

            // ... (这里复用 Python _equations_of_motion 的公式) ...
            // 为了简洁且避免重复代码，这里写出核心公式：
            
            double r = cp / (1.0 + ce * cos(cpsi));
            double z = ck.z_minus * pow(cos(cchi), 2);
            double Delta = r*r - 2.0*r + a_spin*a_spin; // M=1
            double Sigma = r*r + a_spin*a_spin * z;
            
            // V_phi
            double sin2theta = 1.0 - z;
            double term1 = ck.Lz / sin2theta;
            double term2 = a_spin * ck.E;
            double term3 = (a_spin / Delta) * (ck.E * (r*r + a_spin*a_spin) - ck.Lz * a_spin);
            double V_phi = term1 - term2 + term3;
            
            // V_t
            double term1_t = a_spin * (ck.Lz - a_spin * ck.E * sin2theta);
            double term2_t = ((r*r + a_spin*a_spin) / Delta) * (ck.E * (r*r + a_spin*a_spin) - ck.Lz * a_spin);
            double V_t = term1_t + term2_t;
            
            // dchi/dt
            double gamma = ck.E * (pow(r*r + a_spin*a_spin, 2)/Delta - a_spin*a_spin) 
                           - (2.0 * r * a_spin * ck.Lz) / Delta;
            double denominator = gamma + a_spin*a_spin * ck.E * z;
            double dchi_dt = sqrt(abs(ck.beta * (ck.z_plus - z))) / denominator;
            
            // dpsi/dt
            double term_r = (1.0 - ck.E*ck.E) * (ck.r_a - r) * (r - ck.r_p) * (r - ck.r3) * (r - ck.r4);
            double V_r = max(0.0, term_r);
            double denom_psi = 1.0 + ce * cos(cpsi);
            double dr_dpsi = (cp * ce * sin(cpsi)) / (denom_psi*denom_psi);
            
            double dpsi_dt = 0.0;
            if (abs(sin(cpsi)) < 1e-5) {
                 // 简单处理转折点，避免除零
                 dpsi_dt = sqrt(V_r + 1e-14) / (V_t * (abs(dr_dpsi) + 1e-7)); 
                 if (dr_dpsi < 0) dpsi_dt = -dpsi_dt; // 简单的符号处理，实际应更严谨
                 // 更简单的: 给一个非零的推力
                 if (abs(dr_dpsi) < 1e-9) dpsi_dt = 1e-3; 
            } else {
                 dpsi_dt = sqrt(V_r) / (V_t * abs(dr_dpsi)); // 注意符号
                 // 因为我们在 ODE 里积分 psi，psi 应该是单调增加的 (如平近点角) 
                 // 或者如果是几何角度，它会震荡。
                 // NK 方法通常积分的是一个单调递增的相角 psi，然后 r = p/(1+e cos psi)
                 // 所以 dpsi/dt 应该始终为正
            }
            dpsi_dt = abs(dpsi_dt);
            
            double dphi_dt = V_phi / V_t;
            
            return {dp_dt, de_dt, diota_dt, dpsi_dt, dchi_dt, dphi_dt};
        };
        
        // RK4 Step
        auto k1 = get_derivs(p, e, iota, psi, chi, phi);
        
        auto k2 = get_derivs(p + 0.5*dt*k1[0], e + 0.5*dt*k1[1], iota + 0.5*dt*k1[2], 
                             psi + 0.5*dt*k1[3], chi + 0.5*dt*k1[4], phi + 0.5*dt*k1[5]);
                             
        auto k3 = get_derivs(p + 0.5*dt*k2[0], e + 0.5*dt*k2[1], iota + 0.5*dt*k2[2], 
                             psi + 0.5*dt*k2[3], chi + 0.5*dt*k2[4], phi + 0.5*dt*k2[5]);
                             
        auto k4 = get_derivs(p + dt*k3[0], e + dt*k3[1], iota + dt*k3[2], 
                             psi + dt*k3[3], chi + dt*k3[4], phi + dt*k3[5]);

        // Update
        p    += (dt/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        e    += (dt/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
        iota += (dt/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
        psi  += (dt/6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
        chi  += (dt/6.0) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
        phi  += (dt/6.0) * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]);
        
        t += dt;
    }
    
    return traj;
}
// // 占位符：演化函数暂时返回空，我们目前只用 Mapping 加速
// std::vector<OrbitState> BabakNKOrbit::evolve(double duration, double dt) {
//     return {};
// }
}