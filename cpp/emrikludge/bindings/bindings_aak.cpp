// cpp/emrikludge/bindings/bindings_aak.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "emri_params.hpp"
#include "waveform/aak_waveform.hpp"

namespace py = pybind11;
using namespace emrikludge;

/// @brief 把 AAK 相关接口绑定到 Python 模块
///
/// 这个文件会在 CMake 中被编译为 Python 扩展模块的一部分，
/// 例如模块名叫 "_aak_cpu"（具体名字你可以在 CMake 中设定）。
void bind_aak(py::module_& m) {
    // 绑定 EMRIParams
    py::class_<EMRIParams>(m, "EMRIParams")
        .def(py::init<>())
        .def_readwrite("M", &EMRIParams::M)
        .def_readwrite("mu", &EMRIParams::mu)
        .def_readwrite("a", &EMRIParams::a)
        .def_readwrite("p0", &EMRIParams::p0)
        .def_readwrite("e0", &EMRIParams::e0)
        .def_readwrite("iota0", &EMRIParams::iota0)
        .def_readwrite("thetaS", &EMRIParams::thetaS)
        .def_readwrite("phiS", &EMRIParams::phiS)
        .def_readwrite("thetaK", &EMRIParams::thetaK)
        .def_readwrite("phiK", &EMRIParams::phiK)
        .def_readwrite("dist", &EMRIParams::dist)
        .def_readwrite("Phi_phi0", &EMRIParams::Phi_phi0)
        .def_readwrite("Phi_r0", &EMRIParams::Phi_r0)
        .def_readwrite("Phi_theta0", &EMRIParams::Phi_theta0)
        .def_readwrite("T", &EMRIParams::T)
        .def_readwrite("dt", &EMRIParams::dt)
        .def_readwrite("use_eccentric", &EMRIParams::use_eccentric)
        .def_readwrite("use_equatorial", &EMRIParams::use_equatorial);

    // WaveformConfig
    py::class_<WaveformConfig>(m, "WaveformConfig")
        .def(py::init<>())
        .def_readwrite("return_polarizations", &WaveformConfig::return_polarizations)
        .def_readwrite("return_tdi_channels", &WaveformConfig::return_tdi_channels)
        .def_readwrite("return_orbit", &WaveformConfig::return_orbit)
        .def_readwrite("tdi_mode", &WaveformConfig::tdi_mode);

    // AAKWaveformResult
    py::class_<AAKWaveformResult>(m, "AAKWaveformResult")
        .def_readonly("t", &AAKWaveformResult::t)
        .def_readonly("hI", &AAKWaveformResult::hI)
        .def_readonly("hII", &AAKWaveformResult::hII)
        .def_readonly("hplus", &AAKWaveformResult::hplus)
        .def_readonly("hcross", &AAKWaveformResult::hcross)
        .def_readonly("r", &AAKWaveformResult::r)
        .def_readonly("theta", &AAKWaveformResult::theta)
        .def_readonly("phi", &AAKWaveformResult::phi);

    // 核心函数绑定
    m.def("compute_aak_waveform_cpu",
          &compute_aak_waveform_cpu,
          py::arg("emri"),
          py::arg("wf_conf"),
          R"pbdoc(
          计算 CPU 版本的 AAK 波形。

          参数
          ----
          emri : EMRIParams
              EMRI 系统物理参数。
          wf_conf : WaveformConfig
              波形输出配置，例如是否返回轨道、极化等。

          返回
          ----
          AAKWaveformResult
              包含 t, hI, hII 以及可选的 hplus/hcross, 轨道信息。
          )pbdoc");
}

// 入口模块定义：根据你在 CMake 中设定的模块名来修改
PYBIND11_MODULE(_aak_cpu, m) {
    m.doc() = "CPU implementation of AAK EMRI waveform (bindings)";
    bind_aak(m);
}
