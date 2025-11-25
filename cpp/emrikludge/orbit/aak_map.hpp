#pragma once
#include <cmath>
#include <iostream>
#include "kerr_freqs.hpp"

namespace emrikludge {

class AAKMap {
public:
    // 2PN 频率公式 (Barack & Cutler 2004)
    static void get_ak_frequencies(double M, double p, double e, double &n) {
        double a_sma = p / (1.0 - e*e);
        n = sqrt(M / (a_sma * a_sma * a_sma));
    }

    // 映射求解器：已知 Kerr 频率 -> 求 AK 参数
    // 我们主要通过调整 p_ak 来匹配 Omega_r，保持 e, iota 不变 (Standard AAK Map)
    // Omega_r_Kerr = n_AK (Mean motion)
    static void find_ak_parameters(double M, double a, 
                                   double Om_r_ref, double Om_th_ref, double Om_phi_ref,
                                   double p_phys, double e_phys, double i_phys,
                                   double &p_ak, double &e_ak, double &i_ak) {
        
        // 1. 策略：保持 e_ak = e_phys, i_ak = i_phys
        // 只调整 p_ak 使得 AK 的开普勒频率 n 等于 Kerr 的径向频率 Om_r
        // 这是 AAK 最重要的一步映射，因为它保证了相位累积的主频一致
        
        e_ak = e_phys;
        i_ak = i_phys;

        // 2. 求解 p_ak
        // 目标: n(p_ak, e_ak) = Om_r_ref
        // n = sqrt(M / a^3) = sqrt(M * (1-e^2)^3 / p^3)
        // => p^3 = M * (1-e^2)^3 / Om_r^2
        // => p = (M * (1-e^2)^3 / Om_r^2)^(1/3)
        
        double Y = 1.0 - e_ak * e_ak;
        double Y3 = Y*Y*Y;
        
        // 防止 Om_r 为 0
        if (Om_r_ref < 1e-8) {
            p_ak = p_phys;
            return;
        }

        double p3 = M * Y3 / (Om_r_ref * Om_r_ref);
        p_ak = pow(p3, 1.0/3.0);

        // 3. 进动频率修正 (Optional)
        // 真正的 AAK 还需要计算 nu_theta, nu_phi
        // 但在波形生成中，我们直接使用输入的 (Om_r, Om_th, Om_phi) 作为相位驱动
        // p_ak 仅用于计算振幅 A_n
        
        // 安全检查
        if (p_ak < 2.0) p_ak = 2.0;
    }
};

} // namespace