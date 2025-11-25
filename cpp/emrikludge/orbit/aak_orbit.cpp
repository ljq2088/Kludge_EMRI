#include "aak_orbit.hpp" // <--- 必须包含这个，定义了 BabakAAKOrbit 和 AAKState
#include "nk_orbit.hpp"  // 复用 NK 的通量计算
#include "kerr_freqs.hpp"
#include "aak_map.hpp"
#include <cmath>
#include <vector>
#include <iostream>

namespace emrikludge { // <--- 必须包裹在命名空间内

// 实现 evolve 函数
std::vector<AAKState> BabakAAKOrbit::evolve(double duration, double dt) {
    std::vector<AAKState> traj;
    
    // 初始化变量
    double t = 0.0;   // <--- 修复：之前可能漏了定义 t
    double p = p0;
    double e = e0;
    double iota = iota0;
    
    // 预估容量
    size_t steps = static_cast<size_t>(duration / dt);
    traj.reserve(steps + 100);

    while (t < duration) {
        // 1. 计算通量 (复用 NK 模块的 GG06 通量)
        // 注意：compute_gg06_fluxes 是 BabakNKOrbit 的静态成员
        NKFluxes flux = BabakNKOrbit::compute_gg06_fluxes(p, e, iota, a_spin, M_phys, mu_phys);
        
        // 2. 更新物理参数 (简单的 Euler 步进，AAK 对轨道精度要求略低，但也可用 RK4)
        p += flux.dp_dt * dt;
        e += flux.de_dt * dt;
        iota += flux.diota_dt * dt;
        
        // 物理截断保护
        if (p < 3.0 || e >= 0.999 || e < 0.0) break;

        // 3. 计算精确基频 (The Augmented Part)
        // 假设 M_phys 已经归一化或者在内部处理，这里传入 1.0 作为几何单位质量
        KerrFreqs kf = KerrFundamentalFrequencies::compute(1.0, a_spin, p, e, iota);
        
        if (kf.Omega_r == 0.0) break; // Mapping 失败
        
        // 4. 映射到 AK 参数 (The Map)
        double p_ak, e_ak, i_ak;
        AAKMap::find_ak_parameters(1.0, a_spin, kf.Omega_r, kf.Omega_theta, kf.Omega_phi, 
                                   p, e, iota, 
                                   p_ak, e_ak, i_ak);
        
        // Fallback: 如果 Map 失败，回退到物理参数
        if (p_ak == 0.0) { p_ak = p; e_ak = e; i_ak = iota; }

        // 5. 累积相位
        // 注意：这里假设 Omega 是 dPhi/dt (Coordinate time frequency)
        // 如果 kerr_freqs 返回的是 Mino 频率，需要除以 Gamma
        // 这里的 kf 结构体里我们之前已经处理成 Coordinate Freq 了
        m_Phi_r     += kf.Omega_r * dt;
        m_Phi_theta += kf.Omega_theta * dt;
        m_Phi_phi   += kf.Omega_phi * dt;
        
        // 6. 存储状态
        // 列表初始化构造 AAKState
        traj.push_back({t, p_ak, e_ak, i_ak, m_Phi_r, m_Phi_theta, m_Phi_phi});
        
        t += dt;
    }
    
    return traj;
}

} // namespace emrikludge