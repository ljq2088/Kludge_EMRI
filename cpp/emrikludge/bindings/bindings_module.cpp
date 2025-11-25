#include <pybind11/pybind11.h>

namespace py = pybind11;

// 声明外部初始化函数
void init_bindings_nk(py::module &m);
void init_bindings_aak(py::module &m);
// void init_bindings_tdi(py::module &m); // 如果有 TDI

// 定义唯一的模块入口
PYBIND11_MODULE(_emrikludge, m) {
    m.doc() = "EMRI Kludge C++ Backend (NK & AAK)";
    
    // 依次初始化子模块
    init_bindings_nk(m);
    init_bindings_aak(m);
    // init_bindings_tdi(m);
}