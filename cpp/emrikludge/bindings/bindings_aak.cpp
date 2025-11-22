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
    // 绑定 NKFluxes 结构体
    py::class_<NKFluxes>(m, "NKFluxes")
        .def(py::init<>())
        .def_readwrite("dE_dt", &NKFluxes::dE_dt)
        .def_readwrite("dLz_dt", &NKFluxes::dLz_dt)
        .def_readwrite("dQ_dt", &NKFluxes::dQ_dt)
        .def_readwrite("dp_dt", &NKFluxes::dp_dt)
        .def_readwrite("de_dt", &NKFluxes::de_dt)
        .def_readwrite("diota_dt", &NKFluxes::diota_dt);
        
    // 绑定 OrbitState 结构体 
    // 这样 C++ 返回的 vector<OrbitState> 才能被 Python 识别为对象列表
    py::class_<OrbitState>(m, "OrbitState")
        .def_readonly("t", &OrbitState::t)
        .def_readonly("p", &OrbitState::p)
        .def_readonly("e", &OrbitState::e)
        .def_readonly("iota", &OrbitState::iota)
        .def_readonly("r", &OrbitState::r)
        .def_readonly("theta", &OrbitState::theta)
        .def_readonly("phi", &OrbitState::phi)
        .def_readonly("psi", &OrbitState::psi)
        .def_readonly("chi", &OrbitState::chi);
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
    // 绑定 BabakNKOrbit 类及其方法
    // 注意：evolve 是成员函数，compute_gg06_fluxes 是静态函数
    py::class_<BabakNKOrbit>(m, "BabakNKOrbit_CPP")
        .def(py::init<double, double, double, double, double, double>(),
                py::arg("M"), py::arg("a"), py::arg("p"), py::arg("e"), py::arg("iota"), py::arg("mu"))
        .def("evolve", &BabakNKOrbit::evolve, 
                "Run full evolution in C++ (Fast RK4)",
                py::arg("duration"), py::arg("dt"))
        .def_static("get_conserved_quantities", &BabakNKOrbit::get_conserved_quantities)
        .def("compute_gg06_fluxes", &BabakNKOrbit::compute_gg06_fluxes);

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
    
    
}