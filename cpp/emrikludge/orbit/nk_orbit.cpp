// cpp/emrikludge/orbit/nk_orbit.cpp
#include "nk_orbit.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio> 
#include <array>

const double PI = 3.14159265358979323846;
using namespace std;

// ---------------------------------------------------------
// 辅助函数：GG06 Flux Coefficients
// ---------------------------------------------------------
struct GG06Coeffs {
    double g[17]; 
    double g10a, g10b;
};

GG06Coeffs calc_g_coeffs(double e) {
    GG06Coeffs c;
    double e2 = e*e;
    double e4 = e2*e2;
    double e6 = e4*e2;
    // Gair & Glampedakis (2006) Eqs 39 & 46
    c.g[1] = 1.0 + (73.0/24.0)*e2 + (37.0/96.0)*e4;
    c.g[2] = (73.0/12.0) + (823.0/24.0)*e2 + (949.0/32.0)*e4 + (491.0/192.0)*e6;
    c.g[3] = (1247.0/336.0) + (9181.0/672.0)*e2;
    c.g[4] = 4.0 + (1375.0/48.0)*e2;
    c.g[5] = (44711.0/9072.0) + (172157.0/2592.0)*e2;
    c.g[6] = (33.0/16.0) + (359.0/32.0)*e2;
    c.g[7] = (8191.0/672.0) + (44531.0/336.0)*e2;
    c.g[8] = (3749.0/336.0) - (5143.0/168.0)*e2;
    c.g[9] = 1.0 + (7.0/8.0)*e2;
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

namespace emrikludge {
// 构造函数：初始化状态变量
BabakNKOrbit::BabakNKOrbit(double M, double a, double p, double e, double iota, double mu)
    : M_phys(M), a_spin(a), mu_phys(mu) {
    
    do_inspiral = (mu > 0.0);
    
    // 初始化状态
    m_t = 0.0;
    m_p = p;
    m_e = e;
    m_iota = iota;
    m_psi = 0.0;
    m_chi = 0.0;
    m_phi = 0.0;
}

OrbitState BabakNKOrbit::get_current_state() const {
    // 为了返回 r 和 theta，需要简单算一下
    // 注意：这里不保证 mapping 成功，仅作简单转换
    double r_val = m_p / (1.0 + m_e * cos(m_psi));
    // theta 需要 mapping，这里简化返回 0 (调用者通常只关心 p,e,iota)
    return {m_t, m_p, m_e, m_iota, m_psi, m_chi, m_phi, r_val, 0.0};
}
// BabakNKOrbit::BabakNKOrbit(double M, double a, double p, double e, double iota, double mu)
//     : M_phys(M), a_spin(a), p0(p), e0(e), iota0(iota), mu_phys(mu) {
//     do_inspiral = (mu > 0.0);
// }

double BabakNKOrbit::radial_potential(double r, double M, double a, double E, double Lz, double Q) {
    double Delta = r*r - 2*M*r + a*a;
    double term1 = pow(E*(r*r + a*a) - Lz*a, 2);
    double term2 = Delta * (r*r + pow(Lz - a*E, 2) + Q);
    return term1 - term2;
}

double BabakNKOrbit::radial_potential_deriv(double r, double M, double a, double E, double Lz, double Q) {
    double Delta = r*r - 2*M*r + a*a;
    double dDelta = 2*r - 2*M;
    double X = E*(r*r + a*a) - Lz*a;
    double dX = E * 2*r;
    double Y = r*r + pow(Lz - a*E, 2) + Q;
    double dY = 2*r;
    return 2*X*dX - (dDelta*Y + Delta*dY);
}

// --- 核心 Mapping 函数 (强化版) ---
KerrConstants BabakNKOrbit::get_conserved_quantities(double M, double a, double p, double e, double iota,
    double E_g, double L_g, double Q_g) {
    double M_code = 1.0; 
    double r_p = p / (1.0 + e);
    double r_a = p / (1.0 - e);
    double E, Lz, Q;
    // 1. 智能初值猜测 
    // double denom_factor = p - 3.0 - e*e;
    // if (denom_factor < 0.01) denom_factor = 0.01;
    
    // double num_E = (p - 2.0 - 2.0*e) * (p - 2.0 + 2.0*e);
    // if (num_E < 0) num_E = 0;
    
    // double E_schw = sqrt(num_E / (p * denom_factor));
    // double L_schw = p / sqrt(denom_factor);
    
    // double E = E_schw;
    // if (E < 0.1) E = 0.9; 
    
    // double Lz = L_schw * cos(iota);
    // double Q = L_schw * L_schw * pow(sin(iota), 2);
    // 如果传入了有效的 Warm Start 值，直接使用
    if (E_g != 0.0) {
        E = E_g;
        Lz = L_g;
        Q = Q_g;
    } else {
        // 否则使用 Schwarzschild 近似 (Cold Start)
        double denom_factor = p - 3.0 - e*e;
        if (denom_factor < 0.01) denom_factor = 0.01;
        
        double num_E = (p - 2.0 - 2.0*e) * (p - 2.0 + 2.0*e);
        if (num_E < 0) num_E = 0;
        
        double E_schw = sqrt(num_E / (p * denom_factor));
        double L_schw = p / sqrt(denom_factor);
        
        E = E_schw;
        if (E < 0.1) E = 0.9;
        Lz = L_schw * cos(iota);
        Q = L_schw * L_schw * pow(sin(iota), 2);
    }
    // 2. Newton-Raphson 迭代
    const int MAX_ITER = 100;
    const double TOL = 1e-8;
    bool success = false;
    double resid_norm = 0.0;

    for(int iter=0; iter<MAX_ITER; ++iter) {
        double F1, F2, F3;
        // V(rp) = 0
        F1 = radial_potential(r_p, M_code, a, E, Lz, Q);
        // V(ra) = 0 (or V'(p) for circ)
        if (e < 1e-4) {
            F2 = radial_potential_deriv(p, M_code, a, E, Lz, Q);
        } else {
            F2 = radial_potential(r_a, M_code, a, E, Lz, Q);
        }
        // Polar constraint
        double c2 = pow(cos(iota), 2);
        double s2 = pow(sin(iota), 2);
        F3 = Q * c2 - Lz*Lz * s2;
        
        resid_norm = sqrt(F1*F1 + F2*F2 + F3*F3);
        if (resid_norm < TOL) {
            success = true;
            break;
        }
        
        // 解析雅可比
        auto get_V_param_derivs = [&](double r) -> array<double, 3> {
            double Delta = r*r - 2*M_code*r + a*a;
            double X = E*(r*r + a*a) - Lz*a;
            double dV_dE = 2*X*(r*r+a*a) - Delta * 2*(Lz - a*E)*(-a);
            double dV_dL = 2*X*(-a)      - Delta * 2*(Lz - a*E);
            double dV_dQ = -Delta;
            return {dV_dE, dV_dL, dV_dQ};
        };
        
        double J11, J12, J13, J21, J22, J23, J31, J32, J33;
        auto dVp = get_V_param_derivs(r_p);
        J11 = dVp[0]; J12 = dVp[1]; J13 = dVp[2];
        
        if (e < 1e-4) {
            double eps = 1e-6;
            double v_base = radial_potential_deriv(p, M_code, a, E, Lz, Q);
            J21 = (radial_potential_deriv(p, M_code, a, E+eps, Lz, Q) - v_base)/eps;
            J22 = (radial_potential_deriv(p, M_code, a, E, Lz+eps, Q) - v_base)/eps;
            J23 = (radial_potential_deriv(p, M_code, a, E, Lz, Q+eps) - v_base)/eps;
        } else {
            auto dVa = get_V_param_derivs(r_a);
            J21 = dVa[0]; J22 = dVa[1]; J23 = dVa[2];
        }
        // Row 3
        J31 = 0.0;
        J32 = -2.0 * Lz * s2;
        J33 = c2;
        // Cramer's Rule
        double det = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31);
        if (abs(det) < 1e-15) break;
        
        double invDet = 1.0 / det;
        double b1 = -F1, b2 = -F2, b3 = -F3;
        
        double dE = invDet * (b1*(J22*J33 - J23*J32) + b2*(J13*J32 - J12*J33) + b3*(J12*J23 - J13*J22));
        double dL = invDet * (b1*(J23*J31 - J21*J33) + b2*(J11*J33 - J13*J31) + b3*(J13*J21 - J11*J23));
        double dQ = invDet * (b1*(J21*J32 - J22*J31) + b2*(J12*J31 - J11*J32) + b3*(J11*J22 - J12*J21));
        
        double alpha = 1.0;
        if (abs(dE) > 0.1) alpha = 0.1 / abs(dE);
        
        E += alpha * dE;
        Lz += alpha * dL;
        Q += alpha * dQ;
    }
    
    if (!success) {
        // Mapping 失败不中断，返回0由调用者处理
        return {0,0,0,0,0,0,0,0,0,0};
    }
    
    // 3. 计算辅助参数
    KerrConstants k;
    k.E = E; k.Lz = Lz; k.Q = Q; k.r_p = r_p; k.r_a = r_a;
    
    double E2 = E*E;
    double inv_1_E2 = 1.0 / (1.0 - E2);
    double S_r = 2*M_code * inv_1_E2 - (r_p + r_a);
    double denom = r_p * r_a; if (denom < 1e-8) denom=1e-8;
    double P_r = (a*a * Q * inv_1_E2) / denom;
    
    double delta_r = sqrt(max(0.0, S_r*S_r - 4*P_r));
    k.r3 = (S_r + delta_r)/2.0;
    k.r4 = (S_r - delta_r)/2.0;
    if (k.r4 > k.r3) swap(k.r3, k.r4);
    
    k.beta = a*a * (1.0 - E2);
    double term_b = Q + Lz*Lz + a*a*(1.0 - E2);
    double delta_z = 0.0;
    if (abs(k.beta) > 1e-12) {
        delta_z = sqrt(max(0.0, term_b*term_b - 4*k.beta*Q));
        k.z_plus = (term_b + delta_z)/(2*k.beta);
        k.z_minus = (term_b - delta_z)/(2*k.beta);
    } else {
        k.z_plus = 0; k.z_minus = 0;
    }
    return k;
}

NKFluxes BabakNKOrbit::compute_gg06_fluxes(double p, double e, double iota, double a, double M, double mu) {
    NKFluxes flux = {0,0,0,0,0,0};
    double M_code = 1.0;
    double q_mass = mu / M;
    double safe_e = max(0.0, min(e, 0.999));
    
    double q_spin = a / M_code;
    double Mp = M_code / p; 
    double Mp15 = pow(Mp, 1.5), Mp2 = Mp*Mp, Mp35 = pow(Mp, 3.5), Mp5 = pow(Mp, 5.0);
    double cos_i = cos(iota), sin_i = sin(iota);
    double sin2_i = sin_i*sin_i, cos2_i = cos_i*cos_i;
    
    GG06Coeffs g = calc_g_coeffs(safe_e);
    double prefix = pow(1.0 - safe_e*safe_e, 1.5);
    
    // A. Fluxes (Specific)
    double factor_E = -(32.0/5.0) * q_mass * Mp5;
    double term_E = g.g[1] - q_spin*Mp15*g.g[2]*cos_i - Mp*g.g[3] + PI*Mp15*g.g[4] - Mp2*g.g[5] + q_spin*q_spin*Mp2*g.g[6] - (527.0/96.0)*q_spin*q_spin*Mp2*sin2_i;
    flux.dE_dt = factor_E * prefix * term_E; 
    
    double factor_L = -(32.0/5.0) * q_mass * Mp35;
    double term_L = g.g[9]*cos_i + q_spin*Mp15*(g.g10a - cos2_i*g.g10b) - Mp*g.g[11]*cos_i + PI*Mp15*g.g[12]*cos_i - Mp2*g.g[13]*cos_i + q_spin*q_spin*Mp2*cos_i*(g.g[14] - (45.0/8.0)*sin2_i);
    flux.dLz_dt = factor_L * prefix * term_L; 
    
    KerrConstants k_curr = get_conserved_quantities(M_code, a, p, safe_e, iota);
    if (k_curr.E == 0.0) return flux; 
    double Q_spec = k_curr.Q;
    double sqrt_Q_spec = sqrt(abs(Q_spec));
    
    double factor_Q = -(64.0/5.0) * q_mass * Mp35 * sin_i * prefix;
    double term_Q = g.g[9] - q_spin*Mp15*cos_i*g.g10b - Mp*g.g[11] + PI*Mp15*g.g[12] - Mp2*g.g[13] + q_spin*q_spin*Mp2*(g.g[14] - (45.0/8.0)*sin2_i);
    flux.dQ_dt = factor_Q * sqrt_Q_spec * term_Q; 

    // B. Jacobian Transformation
    double dp = 1e-5 * p;
    double de_step = 1e-5;
    double di = 1e-5;
    
    auto get_ELQ = [&](double tp, double te, double ti) -> array<double, 3> {
        KerrConstants k = get_conserved_quantities(M_code, a, tp, te, ti);
        return {k.E, k.Lz, k.Q};
    };
    
    auto v0 = get_ELQ(p, safe_e, iota);
    auto vp = get_ELQ(p+dp, safe_e, iota);
    auto ve = get_ELQ(p, safe_e+de_step, iota);
    auto vi = get_ELQ(p, safe_e, iota+di);
    
    double J11=(vp[0]-v0[0])/dp, J12=(ve[0]-v0[0])/de_step, J13=(vi[0]-v0[0])/di;
    double J21=(vp[1]-v0[1])/dp, J22=(ve[1]-v0[1])/de_step, J23=(vi[1]-v0[1])/di;
    double J31=(vp[2]-v0[2])/dp, J32=(ve[2]-v0[2])/de_step, J33=(vi[2]-v0[2])/di;
    
    double det = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31);
    if (abs(det) < 1e-16) return flux;
    
    double idet = 1.0/det;
    double y1=flux.dE_dt, y2=flux.dLz_dt, y3=flux.dQ_dt;
    
    flux.dp_dt = idet * (y1*(J22*J33 - J23*J32) + y2*(J13*J32 - J12*J33) + y3*(J12*J23 - J13*J22));
    flux.de_dt = idet * (y1*(J23*J31 - J21*J33) + y2*(J11*J33 - J13*J31) + y3*(J13*J21 - J11*J23));
    flux.diota_dt = idet * (y1*(J21*J32 - J22*J31) + y2*(J12*J31 - J11*J32) + y3*(J11*J22 - J12*J21));
    
    return flux;
}

// std::vector<OrbitState> BabakNKOrbit::evolve(double duration, double dt) {
//     std::vector<OrbitState> traj;
//     size_t est_steps = (size_t)(duration/dt);
//     traj.reserve(est_steps + 1000);
    
//     double t=0, p=p0, e=e0, iota=iota0, psi=0, chi=0, phi=0;
//     double last_E = 0.0, last_L = 0.0, last_Q = 0.0;
//     KerrConstants k_init = get_conserved_quantities(1.0, a_spin, p, e, iota); // Cold start
//     if (k_init.E != 0.0) {
//         last_E = k_init.E; last_L = k_init.Lz; last_Q = k_init.Q;
//     }

//     double last_print_t = 0;
//     double print_interval = max(10.0, duration/100.0);
//     while (t < duration) {
//         KerrConstants k = get_conserved_quantities(1.0, a_spin, p, e, iota, last_E, last_L, last_Q);
//         if (k.E == 0.0) {
//             printf("[C++ Warning] Mapping failed during evolution at t=%.1f\n", t);
//             break; 
//         }
//         last_E = k.E; last_L = k.Lz; last_Q = k.Q;
//         double r_val = p / (1.0 + e * cos(psi));
//         double z_val = k.z_minus * pow(cos(chi), 2);
//         double theta_val = acos(sqrt(max(0.0, z_val)));
//         traj.push_back({t, p, e, iota, psi, chi, phi, r_val, theta_val});

//         if (t - last_print_t > print_interval) {
//             printf("\r[C++ Integrating] t = %.1f / %.1f M (%.1f%%)", t, duration, (t/duration)*100.0);
//             fflush(stdout);
//             last_print_t = t;
//         }

//         if (p < 3.0 || e >= 0.999) break;

//         // RK4
//         auto get_derivs = [&](double cp, double ce, double ci, double cpsi, double cchi, double cphi) -> std::array<double, 6> {
//             double dp_dt=0, de_dt=0, diota_dt=0;
//             if (do_inspiral) {
                
//                 double mass_ratio = mu_phys / M_phys;
//                 NKFluxes f = compute_gg06_fluxes(cp, ce, ci, a_spin, 1.0, mass_ratio);
//                 dp_dt = f.dp_dt; de_dt = f.de_dt; diota_dt = f.diota_dt;
//             }
            
//             KerrConstants ck = get_conserved_quantities(1.0, a_spin, cp, ce, ci);
//             if (ck.E == 0.0) return {0,0,0,0,0,0};
            
//             double r = cp/(1+ce*cos(cpsi));
//             double z = ck.z_minus * pow(cos(cchi), 2);
//             double Delta = r*r - 2*r + a_spin*a_spin;
            
//             // V_t
//             double Vt = a_spin*(ck.Lz - a_spin*ck.E*(1-z)) + ((r*r+a_spin*a_spin)/Delta)*(ck.E*(r*r+a_spin*a_spin)-ck.Lz*a_spin);
            
//             // dchi
//             double gamma = ck.E * (pow(r*r+a_spin*a_spin, 2)/Delta - a_spin*a_spin) - (2*r*a_spin*ck.Lz)/Delta;
//             double denom = gamma + a_spin*a_spin*ck.E*z;
//             double dchi = sqrt(abs(ck.beta*(ck.z_plus-z))) / denom;
            
//             // dpsi
//             double Tr = (1-ck.E*ck.E)*(ck.r_a-r)*(r-ck.r_p)*(r-ck.r3)*(r-ck.r4);
//             double dr_dpsi = (cp*ce*sin(cpsi)) / pow(1+ce*cos(cpsi), 2);
//             double dpsi = 0;
//             if (abs(sin(cpsi)) < 1e-5) dpsi = sqrt(max(0.0, Tr)+1e-14)/(Vt*(abs(dr_dpsi)+1e-7));
//             else dpsi = sqrt(max(0.0, Tr))/(Vt*abs(dr_dpsi));
            
//             double Vphi = ck.Lz/(1-z) - a_spin*ck.E + (a_spin/Delta)*(ck.E*(r*r+a_spin*a_spin)-ck.Lz*a_spin);
//             double dphi = Vphi/Vt;
            
//             return {dp_dt, de_dt, diota_dt, dpsi, dchi, dphi};
//         };
        
//         auto k1 = get_derivs(p, e, iota, psi, chi, phi);
//         auto k2 = get_derivs(p+0.5*dt*k1[0], e+0.5*dt*k1[1], iota+0.5*dt*k1[2], psi+0.5*dt*k1[3], chi+0.5*dt*k1[4], phi+0.5*dt*k1[5]);
//         auto k3 = get_derivs(p+0.5*dt*k2[0], e+0.5*dt*k2[1], iota+0.5*dt*k2[2], psi+0.5*dt*k2[3], chi+0.5*dt*k2[4], phi+0.5*dt*k2[5]);
//         auto k4 = get_derivs(p+dt*k3[0], e+dt*k3[1], iota+dt*k3[2], psi+dt*k3[3], chi+dt*k3[4], phi+dt*k3[5]);
        
//         p += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);
//         e += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]);
//         iota += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]);
//         psi += dt/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3]);
//         chi += dt/6*(k1[4]+2*k2[4]+2*k3[4]+k4[4]);
//         phi += dt/6*(k1[5]+2*k2[5]+2*k3[5]+k4[5]);
        
//         t += dt;
//     }
//     printf("\n[C++ Integrating] Done. t=%.1f\n", t);
//     return traj;
// }
// 有状态的 Evolve
std::vector<OrbitState> BabakNKOrbit::evolve(double duration, double dt) {
    std::vector<OrbitState> traj;
    size_t est_steps = (size_t)(duration/dt);
    traj.reserve(est_steps + 100);
    
    double target_t = m_t + duration;
    double last_print_t = m_t;
    double print_interval = max(10.0, duration/10.0); // 每 10% 打印一次

    // 积分主循环
    while (m_t < target_t) {
        // 1. Mapping
        KerrConstants k = get_conserved_quantities(1.0, a_spin, m_p, m_e, m_iota);
        if (k.E == 0.0) {
            printf("[C++ Error] Mapping failed at t=%.2f\n", m_t);
            break; 
        }

        // 2. 记录 (Output)
        double r_val = m_p / (1.0 + m_e * cos(m_psi));
        double z_val = k.z_minus * pow(cos(m_chi), 2);
        double theta_val = acos(sqrt(max(0.0, z_val)));
        
        traj.push_back({m_t, m_p, m_e, m_iota, m_psi, m_chi, m_phi, r_val, theta_val});

        // 进度条 (相对于本次 Chunk)
        if (m_t - last_print_t > print_interval) {
            printf("\r[C++ Chunk] t = %.1f / %.1f (Target: %.1f)", m_t, target_t, target_t);
            fflush(stdout);
            last_print_t = m_t;
        }

        // 终止条件
        if (m_p < 3.0 || m_e >= 0.999) {
            printf("\n[C++ Stop] Plunge detected at t=%.2f\n", m_t);
            break;
        }

        // 3. RK4 Step (使用成员变量)
        auto get_derivs = [&](double cp, double ce, double ci, double cpsi, double cchi, double cphi) -> std::array<double, 6> {
            double dp_dt=0, de_dt=0, diota_dt=0;
            if (do_inspiral) {
                double q_mass = mu_phys / M_phys;
                NKFluxes f = compute_gg06_fluxes(cp, ce, ci, a_spin, 1.0, q_mass);
                dp_dt = f.dp_dt; de_dt = f.de_dt; diota_dt = f.diota_dt;
            }
            
            KerrConstants ck = get_conserved_quantities(1.0, a_spin, cp, ce, ci);
            if (ck.E == 0.0) return {0,0,0,0,0,0};
            
            double r = cp/(1+ce*cos(cpsi));
            double z = ck.z_minus * pow(cos(cchi), 2);
            double Delta = r*r - 2*r + a_spin*a_spin;
            
            double Vt = a_spin*(ck.Lz - a_spin*ck.E*(1-z)) + ((r*r+a_spin*a_spin)/Delta)*(ck.E*(r*r+a_spin*a_spin)-ck.Lz*a_spin);
            
            double gamma = ck.E * (pow(r*r+a_spin*a_spin, 2)/Delta - a_spin*a_spin) - (2*r*a_spin*ck.Lz)/Delta;
            double denom = gamma + a_spin*a_spin*ck.E*z;
            double dchi = sqrt(abs(ck.beta*(ck.z_plus-z))) / denom;
            
            double Tr = (1-ck.E*ck.E)*(ck.r_a-r)*(r-ck.r_p)*(r-ck.r3)*(r-ck.r4);
            double dr_dpsi = (cp*ce*sin(cpsi)) / pow(1+ce*cos(cpsi), 2);
            double dpsi = 0;
            if (abs(sin(cpsi)) < 1e-5) dpsi = sqrt(max(0.0, Tr)+1e-14)/(Vt*(abs(dr_dpsi)+1e-7));
            else dpsi = sqrt(max(0.0, Tr))/(Vt*abs(dr_dpsi));
            
            double Vphi = ck.Lz/(1-z) - a_spin*ck.E + (a_spin/Delta)*(ck.E*(r*r+a_spin*a_spin)-ck.Lz*a_spin);
            double dphi = Vphi/Vt;
            
            return {dp_dt, de_dt, diota_dt, dpsi, dchi, dphi};
        };
        
        auto k1 = get_derivs(m_p, m_e, m_iota, m_psi, m_chi, m_phi);
        // ... (标准 RK4，省略中间变量，使用临时变量计算 k2, k3, k4) ...
        // 为了代码简洁，这里你需要把之前的 RK4 逻辑复制过来，只是把 update 对象改成 m_p 等成员变量
        
        double dt2 = 0.5*dt;
        auto k2 = get_derivs(m_p+dt2*k1[0], m_e+dt2*k1[1], m_iota+dt2*k1[2], m_psi+dt2*k1[3], m_chi+dt2*k1[4], m_phi+dt2*k1[5]);
        auto k3 = get_derivs(m_p+dt2*k2[0], m_e+dt2*k2[1], m_iota+dt2*k2[2], m_psi+dt2*k2[3], m_chi+dt2*k2[4], m_phi+dt2*k2[5]);
        auto k4 = get_derivs(m_p+dt*k3[0],  m_e+dt*k3[1],  m_iota+dt*k3[2],  m_psi+dt*k3[3],  m_chi+dt*k3[4],  m_phi+dt*k3[5]);

        m_p    += dt/6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        m_e    += dt/6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
        m_iota += dt/6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
        m_psi  += dt/6.0 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
        m_chi  += dt/6.0 * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
        m_phi  += dt/6.0 * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]);
        
        m_t += dt;
    }
    
    return traj;
}
} // namespace