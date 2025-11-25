// cpp/emrikludge/bindings/bindings_module.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 声明子模块的初始化函数 (Spokes)
void init_bindings_nk(py::module &m);
void init_bindings_aak(py::module &m);


// 定义唯一的模块入口 (Hub)
PYBIND11_MODULE(_emrikludge, m) {
    m.doc() = "EMRI Kludge C++ Backend (NK & AAK)";
    
    // 依次调用子模块初始化
    init_bindings_nk(m);
    init_bindings_aak(m);
    // init_bindings_ak(m);
    // init_bindings_tdi(m);
}