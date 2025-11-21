// cpp/emrikludge/emri_params.hpp
#pragma once
#include <cmath>
#include <vector>

// 与 Python 端 dataclass 对应的结构体
struct KerrConstants {
    double E;
    double Lz;
    double Q;
    double r3;
    double r4;
    double r_p;
    double r_a;
    double z_minus;
    double z_plus;
    double beta;
};

// 核心 Mapping 函数声明
// 输入: a (无量纲自旋), p (半通径 p/M), e, iota (rad)
// 注意: 内部计算默认 M=1 (几何单位)，因为输入的 p 已经是 p/M
KerrConstants get_conserved_quantities_cpp(double a, double p, double e, double iota);