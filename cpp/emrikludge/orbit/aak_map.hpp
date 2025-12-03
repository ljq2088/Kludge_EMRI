#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include "kerr_freqs.hpp"

namespace emrikludge {

struct MapTarget {
    double Om_r_tgt_dim;   
    double Om_th_tgt_dim;
    double Om_phi_tgt_dim;
    
    double M_phys;         
    double e_phys;
    double iota_phys;
};

class AAKMap {
public:
    // ---------------------------------------------------------
    // 1. AK 频率公式 (输入改为 v = 1/sqrt(p))
    // ---------------------------------------------------------
    static void get_ak_dim_frequencies_from_v(double M_map, double a_map, double v_map, 
                                              double e_phys, double iota_phys,
                                              double &w_r_dim, double &w_th_dim, double &w_phi_dim) {
        
        // v = 1/sqrt(p) => p = 1/v^2
        // 安全检查: p > 2.05 => v < 1/sqrt(2.05) ~ 0.698
        if (v_map > 0.69) v_map = 0.69;
        if (v_map < 1e-3) v_map = 1e-3;

        double p_map = 1.0 / (v_map * v_map);
        double Y = 1.0 - e_phys * e_phys;
        
        // n_dim = (Y/p)^1.5 = (Y * v^2)^1.5 = Y^1.5 * v^3
        double n_dim = pow(Y, 1.5) * (v_map * v_map * v_map);
        
        // epsilon = M/p = v^2
        double epsilon = v_map * v_map;
        double SO = a_map; // a/M
        
        w_r_dim = n_dim;
        
        double peri_adv = 3.0 * epsilon / Y;
        w_th_dim = w_r_dim * (1.0 + peri_adv);

        double lt_term = 2.0 * SO * pow(epsilon, 1.5);
        w_phi_dim = w_th_dim + n_dim * lt_term * std::abs(cos(iota_phys)); 
    }

    // ---------------------------------------------------------
    // 2. GSL 残差函数 (求解 v, M, a)
    // ---------------------------------------------------------
    static int map_residuals(const gsl_vector *x, void *params, gsl_vector *f) {
        struct MapTarget *tgt = (struct MapTarget *) params;

        // [Crucial Change] Variable 0 is now v = 1/sqrt(p)
        double v_map = gsl_vector_get(x, 0);
        double M_map = gsl_vector_get(x, 1);
        double a_map = gsl_vector_get(x, 2);

        // Safety floors
        if (v_map < 1e-4) v_map = 1e-4;
        if (M_map < 1.0) M_map = 1.0;

        double w_r_map, w_th_map, w_phi_map;
        get_ak_dim_frequencies_from_v(M_map, a_map, v_map, tgt->e_phys, tgt->iota_phys, 
                                      w_r_map, w_th_map, w_phi_map);

        // Residuals: Omega_tgt * M_map - Omega_map * M_phys
        double res_r   = tgt->Om_r_tgt_dim   * M_map - w_r_map   * tgt->M_phys;
        double res_th  = tgt->Om_th_tgt_dim  * M_map - w_th_map  * tgt->M_phys;
        double res_phi = tgt->Om_phi_tgt_dim * M_map - w_phi_map * tgt->M_phys;

        // Scaling to O(1) for solver stability
        // Dividing by M_phys normalizes the residual magnitude
        double scale = 1.0 / tgt->M_phys;
        gsl_vector_set(f, 0, res_r * scale);
        gsl_vector_set(f, 1, res_th * scale);
        gsl_vector_set(f, 2, res_phi * scale);

        return GSL_SUCCESS;
    }

    // ---------------------------------------------------------
    // 3. 求解器
    // ---------------------------------------------------------
    static void find_map_parameters(double M_phys, double a_phys, double p_phys,
                                    double e_phys, double iota_phys,
                                    double Om_r_kerr, double Om_th_kerr, double Om_phi_kerr,
                                    double &M_map, double &a_map, double &p_map) {
        
        // Warm start logic needs to handle p -> v conversion
        double v_guess;
        if (p_map > 0.0) {
            v_guess = 1.0 / sqrt(p_map); // Warm start from previous p
        } else {
            v_guess = 1.0 / sqrt(p_phys); // Cold start
            M_map = M_phys;
            a_map = a_phys;
        }

        if (Om_r_kerr < 1e-12) return;

        const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
        static gsl_multiroot_fsolver *s = nullptr;
        if (!s) s = gsl_multiroot_fsolver_alloc(T, 3);

        MapTarget target;
        target.Om_r_tgt_dim   = Om_r_kerr;
        target.Om_th_tgt_dim  = Om_th_kerr;
        target.Om_phi_tgt_dim = Om_phi_kerr;
        target.M_phys         = M_phys;
        target.e_phys         = e_phys;
        target.iota_phys      = iota_phys;

        gsl_multiroot_function f = {&map_residuals, 3, &target};

        gsl_vector *x = gsl_vector_alloc(3);
        gsl_vector_set(x, 0, v_guess); // Set v
        gsl_vector_set(x, 1, M_map);
        gsl_vector_set(x, 2, a_map);

        gsl_multiroot_fsolver_set(s, &f, x);

        int iter = 0;
        int status;
        do {
            iter++;
            status = gsl_multiroot_fsolver_iterate(s);
            if (status) break;
            status = gsl_multiroot_test_residual(s->f, 1e-9);
        } while (status == GSL_CONTINUE && iter < 40);

        if (status == GSL_SUCCESS) {
            double v_res = gsl_vector_get(s->x, 0);
            M_map = gsl_vector_get(s->x, 1);
            a_map = gsl_vector_get(s->x, 2);
            
            // Convert v back to p for output
            p_map = 1.0 / (v_res * v_res);
        } else {
            // Mapping failure fallback
            if (p_map == 0.0) p_map = p_phys; 
            // Else keep previous p_map (best effort)
        }

        gsl_vector_free(x);
    }
};

} // namespace