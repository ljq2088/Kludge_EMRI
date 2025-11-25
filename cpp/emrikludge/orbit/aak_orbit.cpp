#include "nk_orbit.hpp" 
#include "aak_orbit.hpp"
#include "kerr_freqs.hpp" 
#include "aak_map.hpp" 
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio> 
#include <array>   

std::vector<AAKState> BabakAAKOrbit::evolve(double duration, double dt) {
    // ...
    
    while (t < duration) {
        // 1. 计算通量 (复用 NK 模块的 GG06)
        NKFluxes flux = BabakNKOrbit::compute_gg06_fluxes(p, e, iota, a, 1.0, mu/M);
        
        // 2. 更新物理参数 (Euler step for simplicity, or use RK4)
        p += flux.dp_dt * dt;
        e += flux.de_dt * dt;
        iota += flux.diota_dt * dt;
        
        // 3. 计算精确基频 (Step 1: The Augmented Part)
        KerrFreqs kf = KerrFundamentalFrequencies::compute(1.0, a, p, e, iota);
        if (kf.Omega_r == 0.0) break; // Plunge/Error
        
        // 4. 映射到 AK 参数 (Step 2: The Map)
        double p_ak, e_ak, i_ak;
        // 传入当前的物理参数 p, e, iota 作为参考
        AAKMap::find_ak_parameters(1.0, a, kf.Omega_r, kf.Omega_theta, kf.Omega_phi, 
                                   p, e, iota, 
                                   p_ak, e_ak, i_ak);
        
        // 存储状态 (存 p_ak 还是 p_phys? 为了波形生成，存 p_ak)
        // 但为了查看轨迹，可能也想存 p_phys。
        // 这里我们存 p_ak，因为它是波形生成的直接输入
        traj.push_back({t, p_ak, e_ak, i_ak, m_Phi_r, m_Phi_theta, m_Phi_phi});
        
        t += dt;
        
        if (p < 3.0 || e > 0.99) break;
    }
    return traj;
}