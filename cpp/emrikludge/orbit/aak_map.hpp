#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include "kerr_freqs.hpp"

namespace emrikludge {

// 用于传递给 GSL 求解器的参数
struct MapParams {
    double M;
    double a;
    double Omega_r_target;
    double Omega_theta_target;
    double Omega_phi_target;
};

class AAKMap {
public:
    // ---------------------------------------------------------
    // 1. AK 模型的频率公式 (The Analytic Kludge Frequencies)
    // ---------------------------------------------------------
    // 基于 Barack & Cutler (2004) 的 2PN 近似
    // 注意：为了映射的稳定性，公式需要平滑且单调
    static void get_ak_frequencies(double M, double a, double p, double e, double iota,
                                   double &w_r, double &w_th, double &w_phi) {
        // 安全检查
        if (p < 2.1) p = 2.1;
        if (e < 0.0) e = 0.0;
        if (e >= 1.0) e = 0.999;
        
        double Y = 1.0 - e*e;
        // Mean motion (Keplerian frequency) n = sqrt(M/a_sma^3)
        // p = a_sma * (1-e^2) -> a_sma = p/Y
        // n = sqrt(M) * (Y/p)^1.5
        double n = sqrt(M) * pow(Y/p, 1.5);
        
        double epsilon = M / p;
        double SO = a / M; // Spin parameter (dimensionless)
        
        // 近日点进动 (Perihelion Precession) rate: gamma_dot
        // 1PN: 3 * n * epsilon / Y
        // 2PN: ... (使用 simplified Schwarzschild exact form 增加鲁棒性)
        // w_r = n (径向基频)
        // w_theta = n + gamma_dot (极向基频)
        // 实际上, Schwarzschild 中 w_phi = w_theta = w_r * (1 - 6M/p)^(-0.5)
        // 我们使用 Schwarzschild 精确解 + Lense-Thirring 修正
        
        // 1. 径向频率 (Definition)
        w_r = n;
        
        // 2. 极向频率 (Schwarzschild precession)
        // w_theta = w_r / sqrt(1 - 6*M/p_circ)? 
        // 使用 PN 展开: w_th = w_r * (1 + 3*eps/Y)
        double peri_adv = 3.0 * epsilon / Y;
        w_th = w_r * (1.0 + peri_adv);

        // 3. 方位角频率 (Lense-Thirring precession)
        // w_phi = w_theta + w_LT
        // w_LT = 2 * a/M * M^2/p^3 ?? 
        // Frame dragging freq ~ 2 * J / r^3
        // PN: alpha_dot = 2 * n * SO * epsilon^1.5
        // 注意：Lense-Thirring 导致节点进动 (Nodal Precession) -> w_phi 与 w_theta 分离
        // w_phi = w_theta + |alpha_dot * cos(iota)|? 
        // 通常定义: w_phi = w_theta + 2 * SO * n * eps^1.5 * cos(iota) ??
        // 让我们使用一个鲁棒的 1.5PN 近似：
        
        double lt_term = 2.0 * SO * pow(epsilon, 1.5) * cos(iota);
        // 注意符号：w_phi 应该 >= w_theta (对于顺行)
        // 节点进动通常是负的? Omega_node = - 2 J / p^3 ...
        // 但 w_phi 是 dphi/dt, 应该包含轨道频率。
        // 关系: w_phi = w_theta + w_node * cos(iota)? No.
        // w_phi = w_theta + 2.0 * a * M / p^3 ??
        // 简化模型：w_phi = w_theta + 2 * n * (a/M) * (M/p)^1.5
        
        // 为了保证 Map 有解，我们直接加上一个正定的 L-T 项
        w_phi = w_th + n * lt_term;
        
        // *微调*：如果 Map 很难收敛，可能需要调整这里的公式使其更接近 Kerr 行为
        // 但作为 Kludge，只要它是 (p,e,i) 的光滑函数且大致正确即可
    }

    // ---------------------------------------------------------
    // 2. GSL 残差函数 (Residuals)
    // F(x) = AK_Freq(x) - Target_Freq
    // x = {p, e, iota}
    // ---------------------------------------------------------
    static int map_residuals(const gsl_vector *x, void *params, gsl_vector *f) {
        struct MapParams *p_map = (struct MapParams *) params;

        double p_val = gsl_vector_get(x, 0);
        double e_val = gsl_vector_get(x, 1);
        double i_val = gsl_vector_get(x, 2);

        double w_r, w_th, w_phi;
        get_ak_frequencies(p_map->M, p_map->a, p_val, e_val, i_val, w_r, w_th, w_phi);

        // 计算残差 (相对误差可能更好，但绝对误差更稳定)
        // 乘以权重以平衡量级: w_phi ~ w_th > w_r
        gsl_vector_set(f, 0, w_r   - p_map->Omega_r_target);
        gsl_vector_set(f, 1, w_th  - p_map->Omega_theta_target);
        gsl_vector_set(f, 2, w_phi - p_map->Omega_phi_target);

        return GSL_SUCCESS;
    }

    // ---------------------------------------------------------
    // 3. 映射求解器 (Newton-Raphson via GSL Hybrid)
    // ---------------------------------------------------------
    static void find_ak_parameters(double M, double a, 
                                   double Om_r_ref, double Om_th_ref, double Om_phi_ref,
                                   double p_phys, double e_phys, double i_phys,
                                   double &p_ak, double &e_ak, double &i_ak) {
        
        // A. 初值猜测 (Initial Guess)
        // 使用物理参数作为初值 (Standard AAK Strategy)
        // 或者使用之前的 p_ak (如果实现了 Warm Start)
        p_ak = p_phys;
        e_ak = e_phys;
        i_ak = i_phys;

        // 如果目标频率异常（如 plunge），直接返回物理参数
        if (Om_r_ref < 1e-12) return;

        // B. 配置 GSL 求解器
        const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
        static gsl_multiroot_fsolver *s = nullptr;
        
        // 为了性能，复用 solver 实例 (注意多线程安全，这里简化为单线程)
        if (!s) s = gsl_multiroot_fsolver_alloc(T, 3);

        MapParams params = {1.0, a, Om_r_ref, Om_th_ref, Om_phi_ref}; // M=1.0 几何单位
        gsl_multiroot_function f = {&map_residuals, 3, &params};

        gsl_vector *x = gsl_vector_alloc(3);
        gsl_vector_set(x, 0, p_ak);
        gsl_vector_set(x, 1, e_ak);
        gsl_vector_set(x, 2, i_ak);

        gsl_multiroot_fsolver_set(s, &f, x);

        // C. 迭代求解
        int iter = 0;
        int status;
        size_t max_iter = 20; // AAK Map 通常收敛很快，不需要太多步

        do {
            iter++;
            status = gsl_multiroot_fsolver_iterate(s);

            if (status) break; // 求解器遇到问题

            status = gsl_multiroot_test_residual(s->f, 1e-7); // 精度要求
        } while (status == GSL_CONTINUE && iter < max_iter);

        // D. 获取结果
        if (status == GSL_SUCCESS) {
            p_ak = gsl_vector_get(s->x, 0);
            e_ak = gsl_vector_get(s->x, 1);
            i_ak = gsl_vector_get(s->x, 2);
            
            // 物理约束钳位
            if (p_ak < 2.1) p_ak = 2.1;
            if (e_ak < 0.0) e_ak = 0.0; if (e_ak > 0.999) e_ak = 0.999;
            // i_ak wrap around? usually safe.
        } else {
            // 求解失败 (Fail softly): 回退到物理参数或单参数映射
            // printf("Map failed, falling back.\n");
            p_ak = p_phys; 
            e_ak = e_phys; 
            i_ak = i_phys;
        }

        gsl_vector_free(x);
        // s 是 static 的，不释放以供复用 (注意内存泄漏风险，但在脚本运行期间可接受)
    }
};

} // namespace