#pragma once

#include <cmath>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include "kerr_freqs.hpp"

namespace emrikludge {

struct MapTarget {
    double Om_r_tgt_dim;   // Dimensionless Target Freq (Omega * M_phys)
    double Om_th_tgt_dim;
    double Om_phi_tgt_dim;
    
    double M_phys;         
    double e_phys;
    double iota_phys;
};

class AAKMap {
public:
    // ... (get_ak_dim_frequencies function remains the same) ...
    // Copy the implementation from previous steps, no changes needed here.
    static void get_ak_dim_frequencies(double M_map, double a_map, double p_map, 
                                       double e_phys, double iota_phys,
                                       double &w_r_dim, double &w_th_dim, double &w_phi_dim) {
        if (p_map < 2.1) p_map = 2.1;
        double Y = 1.0 - e_phys * e_phys;
        double n_dim = pow(Y / p_map, 1.5); // Dimensionless AK freq depends only on p_map
        
        double epsilon = 1.0 / p_map;
        double SO = a_map; // a_map is dimensionless spin (a/M)
        
        w_r_dim = n_dim;
        double peri_adv = 3.0 * epsilon / Y;
        w_th_dim = w_r_dim * (1.0 + peri_adv);
        double lt_term = 2.0 * SO * pow(epsilon, 1.5);
        w_phi_dim = w_th_dim + n_dim * lt_term * std::abs(cos(iota_phys)); 
    }

    // ---------------------------------------------------------
    // 2. GSL Residuals (NORMALIZED)
    // Variables x: { p_map, q_map, a_map }
    // where q_map = M_map / M_phys
    // ---------------------------------------------------------
    static int map_residuals(const gsl_vector *x, void *params, gsl_vector *f) {
        struct MapTarget *tgt = (struct MapTarget *) params;

        double p_map = gsl_vector_get(x, 0);
        double q_map = gsl_vector_get(x, 1); // Mass Ratio M_map / M_phys
        double a_map = gsl_vector_get(x, 2);

        // Recover Physical M_map for internal logic (if needed, though frequencies only need ratios)
        // Dimensionless AK frequencies actually DON'T depend on M_map absolute value, 
        // they depend on p_map (which is p/M).
        double M_map_abs = q_map * tgt->M_phys; 

        if (p_map < 2.05) p_map = 2.05;
        // q_map should be close to 1.0, allow some range
        if (q_map < 0.1) q_map = 0.1; 

        double w_r_map, w_th_map, w_phi_map;
        // Pass M_map_abs if your frequency function needs it, but standard AK dimensionless doesn't.
        // We pass it to satisfy the signature.
        get_ak_dim_frequencies(M_map_abs, a_map, p_map, tgt->e_phys, tgt->iota_phys, 
                               w_r_map, w_th_map, w_phi_map);

        // --- NORMALIZED RESIDUALS ---
        // Original: w_tgt * M_map - w_map * M_phys = 0
        // Substitute M_map = q * M_phys:
        // w_tgt * (q * M_phys) - w_map * M_phys = 0
        // Divide by M_phys (which is ~1e6):
        // Res = w_tgt * q - w_map
        
        // This keeps residuals order O(1) or O(1e-3), ideal for GSL.
        
        double res_r   = tgt->Om_r_tgt_dim   * q_map - w_r_map;
        double res_th  = tgt->Om_th_tgt_dim  * q_map - w_th_map;
        double res_phi = tgt->Om_phi_tgt_dim * q_map - w_phi_map;

        gsl_vector_set(f, 0, res_r);
        gsl_vector_set(f, 1, res_th);
        gsl_vector_set(f, 2, res_phi);

        return GSL_SUCCESS;
    }

    // ---------------------------------------------------------
    // 3. Solver (Handles Scaling & Warm Start)
    // ---------------------------------------------------------
    static void find_map_parameters(double M_phys, double a_phys, double p_phys,
                                    double e_phys, double iota_phys,
                                    double Om_r_kerr_dim, double Om_th_kerr_dim, double Om_phi_kerr_dim,
                                    double &M_map, double &a_map, double &p_map) { // NOTE: M_map is Physical here
        
        // --- Warm Start / Guess Logic ---
        double q_guess;
        
        if (M_map != 0.0 && p_map != 0.0) {
            // Warm start available.
            // Convert physical M_map back to ratio q for the solver
            q_guess = M_map / M_phys;
        } else {
            // Cold start
            p_map = p_phys;
            a_map = a_phys;
            q_guess = 1.0; // Initial guess: M_map = M_phys
        }

        if (Om_r_kerr_dim < 1e-12) return;

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
        gsl_vector_set(x, 0, p_map);
        gsl_vector_set(x, 1, q_guess); // Set q, NOT M_map
        gsl_vector_set(x, 2, a_map);

        gsl_multiroot_fsolver_set(s, &f, x);

        int iter = 0;
        int status;
        do {
            iter++;
            status = gsl_multiroot_fsolver_iterate(s);
            if (status) break;
            // Strict tolerance, but achievable because residuals are now O(1e-3) not O(1e3)
            status = gsl_multiroot_test_residual(s->f, 1e-9); 
        } while (status == GSL_CONTINUE && iter < 30);

        if (status == GSL_SUCCESS) {
            p_map = gsl_vector_get(s->x, 0);
            double q_res = gsl_vector_get(s->x, 1);
            a_map = gsl_vector_get(s->x, 2);
            
            // Convert q back to Physical M_map for output
            M_map = q_res * M_phys;
        } else {
            // Fallback
            M_map = M_phys;
            a_map = a_phys;
            p_map = p_phys;
        }

        gsl_vector_free(x);
    }
};

} // namespace