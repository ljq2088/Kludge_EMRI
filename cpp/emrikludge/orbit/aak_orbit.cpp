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

    prev_M_map = 0.0;
    prev_a_map = 0.0;
    prev_p_map = 0.0;
    
    // 用于计算 Dimensionless Kerr Frequencies 的基准
    double M_geo_unit = 1.0; 
    
    traj.reserve(static_cast<size_t>(duration / dt) + 100);

    while (t < duration) {
        // 1. 通量演化物理参数
        NKFluxes flux = BabakNKOrbit::compute_gg06_fluxes(p, e, iota, a_spin, M_phys, mu_phys);
        p += flux.dp_dt * dt;
        e += flux.de_dt * dt;
        iota += flux.diota_dt * dt;
        
        if (p < 3.0 || e >= 0.999 || e < 0.0) break;

        // 2. 计算精确 Kerr 无量纲频率
        KerrFreqs kf = KerrFundamentalFrequencies::compute(M_geo_unit, a_spin, p, e, iota);
        if (kf.Omega_r == 0.0) break;

        // 3. 映射 (传入 M_phys 以解出物理 M_map)
        double M_map_curr = prev_M_map;
        double a_map_curr = prev_a_map;
        double p_map_curr = prev_p_map;
        AAKMap::find_map_parameters(M_phys, a_spin, p, e, iota,
                                    kf.Omega_r, kf.Omega_theta, kf.Omega_phi,
                                    M_map_curr, a_map_curr, p_map_curr);
        prev_M_map = M_map_curr;
        prev_a_map = a_map_curr;
        prev_p_map = p_map_curr;

        // 4. 累积相位
        // 注意：相位累积需要物理频率 (1/Time)。
        // kf 是无量纲 (Omega * M)。所以 Physical Omega = kf / M_phys.
        // dt 是物理时间。
        // 所以 dPhi = (kf / M_phys) * dt.
        // 这里的 M_phys 必须和 dt 的单位一致 (比如都是秒，或都是几何单位M)。
        // 假设 dt 是以 M 为单位的时间 (e.g. 5.0M)，则 M_phys 在此公式中应视为 1?
        // 不，如果 dt 是几何单位 (dt_M)，那么 dPhi = kf * dt_M。
        // 如果 dt 是秒，dPhi = (kf / (M_phys_sec)) * dt_sec。
        // 通常 evolve 函数的 dt 是几何单位。如果是这样，直接乘 kf 即可。
        // 假设 evolve 传入的是 Dimensionless Time (t/M).
        m_Phi_r     += kf.Omega_r * dt;
        m_Phi_theta += kf.Omega_theta * dt;
        m_Phi_phi   += kf.Omega_phi * dt;
        
        // 5. 存储状态
        // 存入 kf.Omega_phi (无量纲) 供后续振幅计算参考
        traj.push_back({t, p_map_curr, M_map_curr, a_map_curr, e, iota, 
                        m_Phi_r, m_Phi_theta, m_Phi_phi, kf.Omega_phi});
        
        t += dt;
    }
    
    return traj;
}

} // namespace