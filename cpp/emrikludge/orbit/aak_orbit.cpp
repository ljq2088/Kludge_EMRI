#include "aak_orbit.hpp"
#include "nk_orbit.hpp"
#include "kerr_freqs.hpp"
#include "aak_map.hpp"
#include <cmath>
#include <vector>
#include <iostream> // [NEW] for progress printing
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

namespace emrikludge {

// GSL 参数包
struct AAKGSLParams {
    double M;
    double a;
    double mu;
};

// GSL 导数函数: y = {p, e, iota, Phi_r, Phi_theta, Phi_phi}
int aak_derivs(double t, const double y[], double dydt[], void* params) {
    AAKGSLParams* p_in = static_cast<AAKGSLParams*>(params);
    
    double p = y[0];
    double e = y[1];
    double iota = y[2];
    
    // 1. 物理边界检查
    // AAK/NK 通量公式在 p < 2 附近会发散，必须截断
    if (p < 2.15 || e >= 0.999 || e < 0.0) {
        // 导数置零，让积分器平稳停下，或者返回 GSL_EDOM 强制停止
        return GSL_EDOM; 
    }

    // 2. 计算通量 (Bottleneck!)
    // compute_gg06_fluxes 内部有求根求解器，很慢。
    // 我们依赖 GSL 的步长控制来减少调用次数。
    NKFluxes flux = BabakNKOrbit::compute_gg06_fluxes(p, e, iota, p_in->a, p_in->M, p_in->mu);
    
    dydt[0] = flux.dp_dt;
    dydt[1] = flux.de_dt;
    dydt[2] = flux.diota_dt;

    // 3. 计算相位演化 (dPhi/dt = Omega_Kerr)
    KerrFreqs kf = KerrFundamentalFrequencies::compute(1.0, p_in->a, p, e, iota);
    
    dydt[3] = kf.Omega_r;
    dydt[4] = kf.Omega_theta;
    dydt[5] = kf.Omega_phi;

    return GSL_SUCCESS;
}

std::vector<AAKState> BabakAAKOrbit::evolve(double duration, double dt_sample) {
    std::vector<AAKState> traj;
    size_t est_steps = static_cast<size_t>(duration / dt_sample);
    traj.reserve(est_steps + 1000);

    // --- 1. GSL 初始化 (优化性能) ---
    // [Change 1] 使用 RKF45 代替 RK8PD。
    // RKF45 每步只需 6 次导数计算，而 RK8PD 需要 ~13 次。
    const gsl_odeiv2_step_type* T = gsl_odeiv2_step_rkf45; 
    
    gsl_odeiv2_step* s = gsl_odeiv2_step_alloc(T, 6);
    
    // [Change 2] 放宽容差到 1e-6。
    // 对于 Kludge 模型，1e-9 是过度优化，且会极大地增加计算耗时。
    gsl_odeiv2_control* c = gsl_odeiv2_control_y_new(1e-6, 1e-6); 
    gsl_odeiv2_evolve* e_solver = gsl_odeiv2_evolve_alloc(6);

    AAKGSLParams params;
    params.M = M_phys;
    params.a = a_spin;
    params.mu = mu_phys;

    gsl_odeiv2_system sys = {aak_derivs, NULL, 6, &params};

    // 初始状态
    double t = 0.0;
    double y[6] = {p0, e0, iota0, 0.0, 0.0, 0.0}; 
    double h = 1e-1; // 初始试探步长，稍微给大一点让 GSL 自己缩减

    // --- 2. 映射缓存 (Warm Start) ---
    double prev_M_map = M_phys;
    double prev_a_map = a_spin;
    double prev_p_map = p0;
    const double SMOOTHING_ALPHA = 0.1; // 强力平滑

    // --- 3. 演化循环 ---
    double t_target = 0.0;
    
    // 进度打印控制
    int print_interval = est_steps / 10; 
    if (print_interval < 1) print_interval = 1;
    int step_count = 0;

    while (t_target <= duration) {
        
        // 积分推进到采样点
        while (t < t_target) {
            int status = gsl_odeiv2_evolve_apply(e_solver, c, s, &sys, &t, t_target, &h, y);
            if (status != GSL_SUCCESS) {
                // 如果遇到 Domain Error (Plunge)，我们不再报错退出，而是优雅结束
                goto end_evolution;
            }
        }
        
        // 提取物理参数
        double p_curr = y[0];
        double e_curr = y[1];
        double i_curr = y[2];
        
        // 再次检查物理条件
        if (p_curr < 2.15 || e_curr >= 0.999 || e_curr < 0.0) break;

        // --- AAK Mapping ---
        KerrFreqs kf = KerrFundamentalFrequencies::compute(1.0, a_spin, p_curr, e_curr, i_curr);
        
        // Warm Start Mapping
        double M_map_raw = prev_M_map;
        double a_map_raw = prev_a_map;
        double p_map_raw = prev_p_map;
        
        AAKMap::find_map_parameters(M_phys, a_spin, p_curr, e_curr, i_curr,
                                    kf.Omega_r, kf.Omega_theta, kf.Omega_phi,
                                    M_map_raw, a_map_raw, p_map_raw);
        
        // Smoothing
        double M_map_smooth, a_map_smooth, p_map_smooth;
        if (traj.empty()) {
            M_map_smooth = M_map_raw;
            a_map_smooth = a_map_raw;
            p_map_smooth = p_map_raw;
        } else {
            M_map_smooth = (1.0 - SMOOTHING_ALPHA) * prev_M_map + SMOOTHING_ALPHA * M_map_raw;
            a_map_smooth = (1.0 - SMOOTHING_ALPHA) * prev_a_map + SMOOTHING_ALPHA * a_map_raw;
            p_map_smooth = (1.0 - SMOOTHING_ALPHA) * prev_p_map + SMOOTHING_ALPHA * p_map_raw;
        }
        
        prev_M_map = M_map_smooth;
        prev_a_map = a_map_smooth;
        prev_p_map = p_map_smooth;

        // 存储
        traj.push_back({
            t_target, 
            p_map_smooth, M_map_smooth, a_map_smooth, 
            e_curr, i_curr,                           
            y[3], y[4], y[5],                         
            kf.Omega_phi                              
        });
        
        // 简单的 C++ 端进度条 (防止用户以为卡死)
        if (step_count % print_interval == 0) {
            std::cout << "\r[C++ Evolve] t=" << t << " / " << duration 
                      << " (p=" << p_curr << ", e=" << e_curr << ")" << std::flush;
        }

        t_target += dt_sample;
        step_count++;
    }

end_evolution:
    std::cout << "\n[C++ Evolve] Finished at t=" << t << std::endl;

    gsl_odeiv2_evolve_free(e_solver);
    gsl_odeiv2_control_free(c);
    gsl_odeiv2_step_free(s);

    return traj;
}

} // namespace