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
    // 1. AK Frequency Formula
    // ---------------------------------------------------------
    static void get_ak_dim_frequencies(double M_map, double a_map, double p_map, 
                                       double e_phys, double iota_phys,
                                       double &w_r_dim, double &w_th_dim, double &w_phi_dim) {
        if (p_map < 2.1) p_map = 2.1;
        
        double Y = 1.0 - e_phys * e_phys;
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
    // 2. GSL Residuals (Normalized)
    // ---------------------------------------------------------
    static int map_residuals(const gsl_vector *x, void *params, gsl_vector *f) {
        struct MapTarget *tgt = (struct MapTarget *) params;

        // --- SCALING VARIABLES ---
        // x[0] = p_map (Order ~10)
        // x[1] = M_ratio = M_map / M_phys (Order ~1.0) <--- Key Fix
        // x[2] = a_map (Order ~0.1 - 1.0)

        double p_map = gsl_vector_get(x, 0);
        double M_ratio = gsl_vector_get(x, 1); 
        double a_map = gsl_vector_get(x, 2);

        // Recover physical M_map for calculation
        double M_map = M_ratio * tgt->M_phys;

        // Safety constraints
        if (p_map < 2.05) p_map = 2.05;
        if (M_map < 0.0) M_map = 1.0; 

        double w_r_map, w_th_map, w_phi_map;
        get_ak_dim_frequencies(M_map, a_map, p_map, tgt->e_phys, tgt->iota_phys, 
                               w_r_map, w_th_map, w_phi_map);

        // Residual Equation: w_tgt_phys = w_map_phys
        // Dimensionless: w_tgt_dim / M_phys = w_map_dim / M_map
        // Cross multiply: w_tgt_dim * M_map - w_map_dim * M_phys = 0
        // Substitute M_map = M_ratio * M_phys:
        // w_tgt_dim * M_ratio * M_phys - w_map_dim * M_phys = 0
        // Divide entire equation by M_phys (Normalize residuals to Order 1):
        // Result: w_tgt_dim * M_ratio - w_map_dim = 0
        
        double res_r   = tgt->Om_r_tgt_dim   * M_ratio - w_r_map;
        double res_th  = tgt->Om_th_tgt_dim  * M_ratio - w_th_map;
        double res_phi = tgt->Om_phi_tgt_dim * M_ratio - w_phi_map;

        gsl_vector_set(f, 0, res_r);
        gsl_vector_set(f, 1, res_th);
        gsl_vector_set(f, 2, res_phi);

        return GSL_SUCCESS;
    }

    // ---------------------------------------------------------
    // 3. Solver
    // ---------------------------------------------------------
    static void find_map_parameters(double M_phys, double a_phys, double p_phys,
                                    double e_phys, double iota_phys,
                                    double Om_r_kerr_dim, double Om_th_kerr_dim, double Om_phi_kerr_dim,
                                    double &M_map, double &a_map, double &p_map) {
        
        // [FIX 1] Warm Start Logic
        // Only reset to physical parameters if the input guesses are invalid (zero).
        // This ensures trajectory continuity.
        if (M_map <= 0.0 || p_map <= 0.0) {
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
        
        // [FIX 2] Scaling variables for the solver
        // We pass M_ratio (~1.0) instead of M_map (~1e6)
        gsl_vector_set(x, 0, p_map);
        gsl_vector_set(x, 1, M_map / M_phys); 
        gsl_vector_set(x, 2, a_map);

        gsl_multiroot_fsolver_set(s, &f, x);

        int iter = 0;
        int status;
        do {
            iter++;
            status = gsl_multiroot_fsolver_iterate(s);
            if (status) break;
            
            // Tighten tolerance slightly to prevent low-level jitter
            status = gsl_multiroot_test_residual(s->f, 1e-9); 
        } while (status == GSL_CONTINUE && iter < 30);

        if (status == GSL_SUCCESS) {
            p_map = gsl_vector_get(s->x, 0);
            double M_ratio = gsl_vector_get(s->x, 1);
            M_map = M_ratio * M_phys; // Restore physical mass
            a_map = gsl_vector_get(s->x, 2);
        } else {
            // If failed, reset to physical (or keep previous). Reset is safer to avoid drift.
            M_map = M_phys;
            a_map = a_phys;
            p_map = p_phys;
        }
        gsl_vector_free(x);
    }
};

} // namespace