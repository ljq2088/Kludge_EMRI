// cpp/emrikludge/emri_params.cpp
#include "emri_params.hpp"
#include <cmath>

namespace emrikludge {

EMRIParams::EMRIParams()
    : M(1.0e6)
    , mu(10.0)
    , a(0.0)
    , p0(10.0)
    , e0(0.0)
    , iota0(0.0)
    , thetaS(M_PI / 3.0)
    , phiS(0.0)
    , thetaK(M_PI / 3.0)
    , phiK(0.0)
    , dist(1.0)
    , Phi_phi0(0.0)
    , Phi_r0(0.0)
    , Phi_theta0(0.0)
    , T(1.0e7) // 举例，后面你可以按需要改默认值
    , dt(10.0)
    , use_eccentric(true)
    , use_equatorial(false)
{
    // 这里可以做一些简单的范围检查或归一化
    // 例如: 限制 a 在 (-M, M)，角度映射到 [0, 2π) or [0, π]
}

/// WaveformConfig 默认配置
WaveformConfig::WaveformConfig()
    : return_polarizations(false)
    , return_tdi_channels(false)
    , return_orbit(false)
    , tdi_mode("none")
{
}

} // namespace emrikludge
