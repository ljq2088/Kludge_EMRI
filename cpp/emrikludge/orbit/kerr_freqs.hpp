#pragma once

#include <cmath>
#include <algorithm>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_elljac.h> // [修复1] 必须包含这个头文件
#include "nk_orbit.hpp"
#include "../emri_params.hpp"

namespace emrikludge {

struct KerrFreqs {
    double Omega_r;
    double Omega_theta;
    double Omega_phi;
    double Gamma; // dt/dlambda
};

class KerrFundamentalFrequencies {
public:
    static KerrFreqs compute(double M, double a, double p, double e, double iota) {
        // 1. 获取守恒量
        KerrConstants k = BabakNKOrbit::get_conserved_quantities(M, a, p, e, iota);
        if (k.E == 0.0) return {0,0,0,0};

        double E = k.E; double Lz = k.Lz; double Q = k.Q;
        // 根的定义: r1(远), r2(近)
        double r1 = k.r_a; 
        double r2 = k.r_p; 
        double r3 = k.r3; 
        double r4 = k.r4;
        
        double z_minus = k.z_minus; double z_plus = k.z_plus;
        double a2 = a*a; double E2 = E*E; double beta = a2 * (1.0 - E2);

        // --- 2. 径向部分 (Radial) ---
        double num_kr = (r1 - r2) * (r3 - r4);
        double den_kr = (r1 - r3) * (r2 - r4);
        double kr2 = num_kr / den_kr;
        if (kr2 < 0.0) kr2 = 0.0;
        if (kr2 > 1.0) kr2 = 1.0;
        double kr = sqrt(kr2);

        double Kr = gsl_sf_ellint_Kcomp(kr, GSL_PREC_DOUBLE);
        double gamma_r = sqrt((1.0 - E2) * (r1 - r3) * (r2 - r4)) / 2.0;
        double Lambda_r = 2.0 * Kr / gamma_r;
        double Upsilon_r = 2.0 * M_PI / Lambda_r;

        // --- 3. 极向部分 (Polar) ---
        // [修复3] 显式定义 kth2
        double kth2 = z_minus / z_plus;
        if (kth2 < 0.0) kth2 = 0.0;
        if (kth2 > 1.0) kth2 = 1.0;
        double kth = sqrt(kth2);

        double Kth = gsl_sf_ellint_Kcomp(kth, GSL_PREC_DOUBLE);
        double gamma_th = sqrt(beta * z_plus);
        double Lambda_th = 4.0 * Kth / gamma_th;
        double Upsilon_th = 2.0 * M_PI / Lambda_th;

        // --- 4. 计算 Gamma 和 Omega_phi ---
        
        // A. 极向积分 <cos^2 theta>
        double Eth = gsl_sf_ellint_Ecomp(kth, GSL_PREC_DOUBLE);
        // [修复3] 这里使用了 kth2
        double z_avg = z_minus * (1.0 + (Eth/Kth - 1.0)/(kth2)); 
        
        // <1/(1-z)> for Phi
        double n_z = z_minus; 
        double Pi_z = gsl_sf_ellint_Pcomp(n_z, kth, GSL_PREC_DOUBLE);
        double Upsilon_phi_z = Lz * (1.0 + (z_minus/kth2)*(Pi_z/Kth - 1.0)) / (1.0 - z_minus);

        // B. 径向积分 (使用数值积分替代复杂解析解)
        double sum_r2 = 0.0;
        double sum_term_phi = 0.0;
        int N_int = 20;
        double du = Kr / N_int;
        
        double h_mod = (r1-r2)/(r1-r3); // parameter h for r(u)

        for(int i=0; i<=N_int; ++i) {
            double u = i * du;
            
            // [修复1] 使用 gsl_sf_elljac_e 计算 sn
            // 注意 GSL 接受参数 m = k^2
            double sn, cn, dn;
            gsl_sf_elljac_e(u, kr2, &sn, &cn, &dn);
            
            double sn2 = sn * sn;
            // r(u) mapping
            double r_val = (r3 - r2 * h_mod * sn2) / (1.0 - h_mod * sn2);
            double Delta = r_val*r_val - 2.0*r_val + a2;
            
            double weight = (i==0 || i==N_int) ? 1.0 : (i%2==0 ? 2.0 : 4.0);
            
            // 积分项: r^2
            sum_r2 += weight * (r_val * r_val);
            
            // 积分项: Omega_phi 的径向部分
            double term = a * (2.0*M*E*r_val - a*Lz) / Delta;
            sum_term_phi += weight * term;
        }
        
        double r2_avg = (sum_r2 * du / 3.0) / Kr;
        double phi_r_term_avg = (sum_term_phi * du / 3.0) / Kr;

        // Gamma 计算 (Coordinate Time Dilation)
        // [修复2] 将 r_p + r_a 替换为 r1 + r2
        double Gamma = E * (r2_avg + a2 * z_avg) + 2.0 * M * E * (r1 + r2)/2.0; 
        
        double w_phi = (phi_r_term_avg + Upsilon_phi_z) / Gamma;
        double w_r   = Upsilon_r / Gamma;
        double w_th  = Upsilon_th / Gamma;

        return {w_r, w_th, w_phi, Gamma};
    }
};

} // namespace