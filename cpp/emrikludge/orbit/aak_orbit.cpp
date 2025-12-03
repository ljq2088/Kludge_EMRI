#include "aak_orbit.hpp"
#include "nk_orbit.hpp"
#include "kerr_freqs.hpp"
#include "aak_map.hpp"
#include <cmath>
#include <vector>

namespace emrikludge {

std::vector<AAKState> BabakAAKOrbit::evolve(double duration, double dt) {
    std::vector<AAKState> traj;
    
    double t = 0.0;
    double p = p0;
    double e = e0;
    double iota = iota0;
    
    double M_geo_unit = 1.0; 
    
    // 初始化缓存 (Warm Start & Smoothing)
    prev_M_map = 0.0;
    prev_a_map = 0.0;
    prev_p_map = 0.0;

    traj.reserve(static_cast<size_t>(duration / dt) + 100);

    // 平滑因子 (Alpha): 越小越平滑，但滞后越大。
    // 0.2 表示新值权重 20%，历史值权重 80%。
    // 这能极其有效地消除 "Solver Jitter"。
    const double SMOOTHING_ALPHA = 0.1; 

    size_t step = 0;
    while (t < duration) {
        // 1. 物理演化 (NK Flux)
        NKFluxes flux = BabakNKOrbit::compute_gg06_fluxes(p, e, iota, a_spin, M_phys, mu_phys);
        p += flux.dp_dt * dt;
        e += flux.de_dt * dt;
        iota += flux.diota_dt * dt;
        
        if (p < 3.0 || e >= 0.999 || e < 0.0) break;

        // 2. 计算 Kerr 频率
        KerrFreqs kf = KerrFundamentalFrequencies::compute(M_geo_unit, a_spin, p, e, iota);
        if (kf.Omega_r == 0.0) break;

        // 3. 映射求解 (Raw Solve)
        double M_map_raw, a_map_raw, p_map_raw;
        
        // 使用上一时刻的平滑值作为初值 (Warm Start)
        M_map_raw = prev_M_map;
        a_map_raw = prev_a_map;
        p_map_raw = prev_p_map;

        AAKMap::find_map_parameters(M_phys, a_spin, p, e, iota,
                                    kf.Omega_r, kf.Omega_theta, kf.Omega_phi,
                                    M_map_raw, a_map_raw, p_map_raw);

        // 4. 平滑处理 (Smoothing / Interpolation Proxy)
        double M_map_smooth, a_map_smooth, p_map_smooth;

        if (step == 0) {
            // 第一步直接接受
            M_map_smooth = M_map_raw;
            a_map_smooth = a_map_raw;
            p_map_smooth = p_map_raw;
        } else {
            // 后续步骤应用 EMA 滤波
            M_map_smooth = (1.0 - SMOOTHING_ALPHA) * prev_M_map + SMOOTHING_ALPHA * M_map_raw;
            a_map_smooth = (1.0 - SMOOTHING_ALPHA) * prev_a_map + SMOOTHING_ALPHA * a_map_raw;
            p_map_smooth = (1.0 - SMOOTHING_ALPHA) * prev_p_map + SMOOTHING_ALPHA * p_map_raw;
        }

        // 更新缓存 (供下一步 Warm Start 和平滑使用)
        prev_M_map = M_map_smooth;
        prev_a_map = a_map_smooth;
        prev_p_map = p_map_smooth;

        // 5. 累积相位
        m_Phi_r     += kf.Omega_r * dt;
        m_Phi_theta += kf.Omega_theta * dt;
        m_Phi_phi   += kf.Omega_phi * dt;
        
        // 6. 存储状态 (使用平滑后的值!)
        traj.push_back({t, p_map_smooth, M_map_smooth, a_map_smooth, e, iota, 
                        m_Phi_r, m_Phi_theta, m_Phi_phi, kf.Omega_phi});
        
        t += dt;
        step++;
    }
    
    return traj;
}

} // namespace