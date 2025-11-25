#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "orbit/aak_orbit.hpp"
#include "orbit/kerr_freqs.hpp" 
#include "waveform/aak_waveform.hpp"

namespace py = pybind11;
using namespace emrikludge;

void init_bindings_aak(py::module &m) {
    // --- AAK 状态 ---
    py::class_<AAKState>(m, "AAKState")
        .def_readwrite("t", &AAKState::t)
        .def_readwrite("p_ak", &AAKState::p_ak)
        .def_readwrite("e_ak", &AAKState::e_ak)
        .def_readwrite("iota_ak", &AAKState::iota_ak)
        .def_readwrite("Phi_r", &AAKState::Phi_r)
        .def_readwrite("Phi_theta", &AAKState::Phi_theta)
        .def_readwrite("Phi_phi", &AAKState::Phi_phi);

    // --- AAK 轨道 ---
    py::class_<BabakAAKOrbit>(m, "BabakAAKOrbit")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("M"), py::arg("a"), py::arg("p"), py::arg("e"), py::arg("iota"), py::arg("mu"))
        .def("evolve", &BabakAAKOrbit::evolve,
             py::arg("duration"), py::arg("dt"),
             "Evolve AAK orbit and return trajectory with phases");

    // --- AAK 波形 ---
    m.def("generate_aak_waveform_cpp", &generate_aak_waveform_cpp,
          "Generate AAK waveform",
          py::arg("t"), py::arg("p"), py::arg("e"), py::arg("iota"),
          py::arg("Phi_r"), py::arg("Phi_th"), py::arg("Phi_phi"),
          py::arg("M"), py::arg("mu"), py::arg("dist"),
          py::arg("viewing_theta"), py::arg("viewing_phi"));

    // --- 频率计算 (调试用) ---
    py::class_<KerrFreqs>(m, "KerrFreqs")
        .def_readonly("Omega_r", &KerrFreqs::Omega_r)
        .def_readonly("Omega_theta", &KerrFreqs::Omega_theta)
        .def_readonly("Omega_phi", &KerrFreqs::Omega_phi)
        .def_readonly("Gamma", &KerrFreqs::Gamma);

    m.def("compute_kerr_freqs", &KerrFundamentalFrequencies::compute,
          "Compute exact Kerr fundamental frequencies",
          py::arg("M"), py::arg("a"), py::arg("p"), py::arg("e"), py::arg("iota"));
}