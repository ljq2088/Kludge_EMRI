// cpp/emrikludge/bindings/bindings_aak.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// 引入头文件
#include "emri_params.hpp"
#include "orbit/nk_orbit.hpp" 
#include "waveform/aak_waveform.hpp"

namespace py = pybind11;
using namespace emrikludge; // 打开命名空间

// 定义唯一的模块入口 _emrikludge
PYBIND11_MODULE(_emrikludge, m) {
    m.doc() = "EMRI Kludge C++ Extension (AAK + NK)";

    // -------------------------------------------------------
    // 1. 基础数据结构绑定
    // -------------------------------------------------------
    py::class_<EMRIParams>(m, "EMRIParams")
        .def(py::init<>())
        .def_readwrite("M", &EMRIParams::M)
        .def_readwrite("mu", &EMRIParams::mu)
        .def_readwrite("a", &EMRIParams::a)
        .def_readwrite("p0", &EMRIParams::p0)
        .def_readwrite("e0", &EMRIParams::e0)
        .def_readwrite("iota0", &EMRIParams::iota0)
        .def_readwrite("T", &EMRIParams::T)
        .def_readwrite("dt", &EMRIParams::dt)
        // ... 其他字段按需绑定 ...
        .def_readwrite("use_eccentric", &EMRIParams::use_eccentric);

    py::class_<WaveformConfig>(m, "WaveformConfig")
        .def(py::init<>())
        .def_readwrite("return_polarizations", &WaveformConfig::return_polarizations)
        .def_readwrite("return_orbit", &WaveformConfig::return_orbit);

    // 绑定 KerrConstants
    py::class_<KerrConstants>(m, "KerrConstants")
        .def(py::init<>())
        .def_readwrite("E", &KerrConstants::E)
        .def_readwrite("Lz", &KerrConstants::Lz)
        .def_readwrite("Q", &KerrConstants::Q)
        .def_readwrite("r3", &KerrConstants::r3)
        .def_readwrite("r4", &KerrConstants::r4)
        .def_readwrite("r_p", &KerrConstants::r_p)
        .def_readwrite("r_a", &KerrConstants::r_a)
        .def_readwrite("z_minus", &KerrConstants::z_minus)
        .def_readwrite("z_plus", &KerrConstants::z_plus)
        .def_readwrite("beta", &KerrConstants::beta);

    // -------------------------------------------------------
    // 2. AAK 模块绑定
    // -------------------------------------------------------
    py::class_<AAKWaveformResult>(m, "AAKWaveformResult")
        .def_readonly("t", &AAKWaveformResult::t)
        .def_readonly("hplus", &AAKWaveformResult::hplus)
        .def_readonly("hcross", &AAKWaveformResult::hcross);

    m.def("compute_aak_waveform_cpu",
          &compute_aak_waveform_cpu,
          py::arg("emri"),
          py::arg("wf_conf"),
          "Compute AAK waveform (CPU)");

    // -------------------------------------------------------
    // 3. NK 模块绑定 (Mapping 加速)
    // -------------------------------------------------------
    m.def("get_conserved_quantities_cpp", 
          &BabakNKOrbit::get_conserved_quantities,
          "Calculate E, Lz, Q using C++ Newton-Raphson",
          py::arg("M"), py::arg("a"), py::arg("p"), py::arg("e"), py::arg("iota"));
    
    m.def("compute_gg06_fluxes_cpp", &BabakNKOrbit::compute_gg06_fluxes, ...);
}