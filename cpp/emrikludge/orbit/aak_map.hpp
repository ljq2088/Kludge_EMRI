#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include "kerr_freqs.hpp"

namespace emrikludge {

struct MapTarget {
    double Om_r_tgt_dim;   // 无量纲目标频率 (Omega * M_phys)
    double Om_th_tgt_dim;
    double Om_phi_tgt_dim;
    
    double M_phys;         // 真实的物理质量
    double e_phys;
    double iota_phys;
};

class AAKMap {
public:
    // ---------------------------------------------------------
    // 1. AK 无量纲频率公式
    // ---------------------------------------------------------
    static void get_ak_dim_frequencies(double M_map, double a_map, double p_map, 
                                       double e_phys, double iota_phys,
                                       double &w_r_dim, double &w_th_dim, double &w_phi_dim) {
        if (p_map < 2.05) p_map = 2.05; 
        
        double Y = 1.0 - e_phys * e_phys;
        
        // n_dim = (M_map * Y / p_map)^1.5 ? 
        // 不，AK公式中 p_map 已经是几何单位。无量纲频率只取决于几何参数。
        // w_dim_AK = (Y / p_map)^1.5
        // M_map 的作用是在残差方程中调整物理时间尺度。
        
        double n_dim = pow(Y / p_map, 1.5);
        double epsilon = 1.0 / p_map;
        double SO = a_map; 
        
        w_r_dim = n_dim;
        
        double peri_adv = 3.0 * epsilon / Y;
        w_th_dim = w_r_dim * (1.0 + peri_adv);

        double lt_term = 2.0 * SO * pow(epsilon, 1.5);
        w_phi_dim = w_th_dim + n_dim * lt_term * std::abs(cos(iota_phys)); 
    }

    // ---------------------------------------------------------
    // 2. GSL 残差函数 (归一化版)
    // ---------------------------------------------------------
    static int map_residuals(const gsl_vector *x, void *params, gsl_vector *f) {
        struct MapTarget *tgt = (struct MapTarget *) params;

        // 提取变量 (全部为 O(1) 量级)
        double p_map = gsl_vector_get(x, 0);
        double q_M   = gsl_vector_get(x, 1); // q_M = M_map / M_phys (Expect ~1.0)
        double a_map = gsl_vector_get(x, 2); 

        // 恢复物理量
        double M_map = q_M * tgt->M_phys;

        // 保护
        if (p_map < 2.05) p_map = 2.05;
        if (M_map < 1.0) M_map = 1.0; 

        double w_r_dim, w_th_dim, w_phi_dim;
        get_ak_dim_frequencies(M_map, a_map, p_map, tgt->e_phys, tgt->iota_phys, 
                               w_r_dim, w_th_dim, w_phi_dim);

        // 构造归一化残差
        // 原方程: w_tgt_dim * M_map - w_map_dim * M_phys = 0
        // 代入 M_map = q_M * M_phys:
        // w_tgt_dim * q_M * M_phys - w_map_dim * M_phys = 0
        // 两边消去 M_phys:
        // w_tgt_dim * q_M - w_map_dim = 0
        
        double res_r   = tgt->Om_r_tgt_dim   * q_M - w_r_dim;
        double res_th  = tgt->Om_th_tgt_dim  * q_M - w_th_dim;
        double res_phi = tgt->Om_phi_tgt_dim * q_M - w_phi_dim;

        gsl_vector_set(f, 0, res_r);
        gsl_vector_set(f, 1, res_th);
        gsl_vector_set(f, 2, res_phi);

        return GSL_SUCCESS;
    }

    // ---------------------------------------------------------
    // 3. 映射求解器
    // ---------------------------------------------------------
    static void find_map_parameters(double M_phys, double a_phys, double p_phys,
                                    double e_phys, double iota_phys,
                                    double Om_r_kerr_dim, double Om_th_kerr_dim, double Om_phi_kerr_dim,
                                    double &M_map, double &a_map, double &p_map) {
        
        // Warm Start 逻辑: 
        // 如果传入的 guess 都是0 (第一次)，则初始化为物理参数
        if (M_map == 0.0 || p_map == 0.0) {
            M_map = M_phys;
            a_map = a_phys;
            p_map = p_phys;
        }

        if (Om_r_kerr_dim < 1e-8) return;

        const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
        static gsl_multiroot_fsolver *s = nullptr;
        if (!s) s = gsl_multiroot_fsolver_alloc(T, 3);

        MapTarget target;
        target.Om_r_tgt_dim   = Om_r_kerr_dim;
        target.Om_th_tgt_dim  = Om_th_kerr_dim;
        target.Om_phi_tgt_dim = Om_phi_kerr_dim;
        target.M_phys         = M_phys;
        target.e_phys         = e_phys;
        target.iota_phys      = iota_phys;

        gsl_multiroot_function f = {&map_residuals, 3, &target};

        gsl_vector *x = gsl_vector_alloc(3);
        // 设置初值 (注意转换 q_M)
        gsl_vector_set(x, 0, p_map);
        gsl_vector_set(x, 1, M_map / M_phys); // 初值设为比值 (~1.0)
        gsl_vector_set(x, 2, a_map);

        gsl_multiroot_fsolver_set(s, &f, x);

        int iter = 0;
        int status;
        do {
            iter++;
            status = gsl_multiroot_fsolver_iterate(s);
            if (status) break;
            // 因为残差已经归一化，1e-9 对应非常高的精度
            status = gsl_multiroot_test_residual(s->f, 1e-10);
        } while (status == GSL_CONTINUE && iter < 30);

        if (status == GSL_SUCCESS) {
            p_map = gsl_vector_get(s->x, 0);
            double q_M = gsl_vector_get(s->x, 1);
            M_map = q_M * M_phys; // 还原为物理值输出
            a_map = gsl_vector_get(s->x, 2);
        } else {
            // 失败回退
            M_map = M_phys;
            a_map = a_phys;
            p_map = p_phys;
        }
        gsl_vector_free(x);
    }
};

} // namespace