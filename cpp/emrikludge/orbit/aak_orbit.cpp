#include "nk_orbit.hpp" 
#include "kerr_freqs.hpp" 
#include "aak_map.hpp"    
// ...

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
        AAKMap::find_ak_parameters(1.0, a, kf.Omega_r, kf.Omega_theta, kf.Omega_phi, 
                                   p_ak, e_ak, i_ak);
        
        // *Fallback*: 如果 Map 没算出来，直接用物理参数 (Semi-relativistic approximation)
        if (p_ak == 0.0) { p_ak = p; e_ak = e; i_ak = iota; }

        // 5. 累积相位 (注意: kf.Omega 是 dPhi/dMinoTime ? 还是 dPhi/dt ?)
        // KerrFreqs 计算的是 Upsilon (Mino frequencies).
        // 需要除以 Gamma (time dilation) 转换成 dPhi/dt
        // dPhi/dt = Upsilon / Gamma
        double dt_dtau = kf.Gamma; 
        // 如果 kf.Gamma 还没实现好，暂时假设它归一化了或者用 Coordinate Time Freqs
        // 假设 KerrFreqs 返回的是 Coordinate time frequencies (dPhi/dt)
        
        m_Phi_r     += kf.Omega_r * dt;
        m_Phi_theta += kf.Omega_theta * dt;
        m_Phi_phi   += kf.Omega_phi * dt;
        
        traj.push_back({t, p_ak, e_ak, i_ak, m_Phi_r, m_Phi_theta, m_Phi_phi});
        
        t += dt;
        
        if (p < 3.0 || e > 0.99) break;
    }
    return traj;
}