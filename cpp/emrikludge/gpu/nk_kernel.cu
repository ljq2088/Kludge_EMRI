// cpp/emrikludge/gpu/nk_kernel.cu
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

// 定义常数
#define PI 3.14159265358979323846

// --- 1. 辅助数学函数 (__device__) ---
// 所有的 std::pow, std::cos 等都要换成 CUDA 的 pow, cos

__device__ double d_max(double a, double b) { return a > b ? a : b; }
__device__ double d_min(double a, double b) { return a < b ? a : b; }

// --- 2. 移植 GG06 通量和 Mapping (__device__) ---
// 这里需要把你 nk_orbit.cpp 里的 compute_gg06_fluxes 和 get_conserved_quantities 
// 复制过来，把函数签名改成 __device__，并去掉 GSL 依赖。

// 简化的结构体
struct KerrConstantsGPU {
    double E, Lz, Q;
    double r_p, r_a, z_minus, z_plus, r3, r4, beta;
};

__device__ KerrConstantsGPU get_constants_gpu(double M, double a, double p, double e, double iota) {
    // ... (此处需填入 nk_orbit.cpp 中 get_conserved_quantities 的逻辑) ...
    // 注意：在 GPU 上做 Newton-Raphson 迭代可能会导致线程分歧(Warp Divergence)，
    // 建议设置较小的最大迭代次数（如 10 次）。
    
    KerrConstantsGPU k;
    // ... 计算 k.E, k.Lz 等 ...
    return k; // 占位
}

// --- 3. 导数方程 (__device__) ---
// 对应 nk_orbit.cpp 中的 gsl_derivs
__device__ void calculate_derivs(double t, const double y[], double dydt[], 
                                 double M, double a, double mu, bool do_inspiral) {
    double p = y[0];
    double e = y[1];
    double iota = y[2];
    double psi = y[3];
    double chi = y[4];
    double phi = y[5];

    // 物理截断
    if (p < 3.0 || e >= 0.999) {
        dydt[0]=0; dydt[1]=0; dydt[2]=0; dydt[3]=0; dydt[4]=0; dydt[5]=0;
        return;
    }

    // 1. 计算通量 (Radiation Reaction)
    double dp_dt=0, de_dt=0, diota_dt=0;
    if (do_inspiral) {
        // ... 填入 compute_gg06_fluxes 的逻辑 ...
    }

    // 2. 计算测地线运动
    KerrConstantsGPU k = get_constants_gpu(M, a, p, e, iota);
    
    // ... 填入 dpsi_dt, dchi_dt, dphi_dt 的计算逻辑 ...
    // 参考 nk_orbit.cpp 中的实现
    
    dydt[0] = dp_dt;
    dydt[1] = de_dt;
    dydt[2] = diota_dt;
    dydt[3] = 0.0; // 占位: dpsi_dt
    dydt[4] = 0.0; // 占位: dchi_dt
    dydt[5] = 0.0; // 占位: dphi_dt
}

// --- 4. RK4 步进器 (__device__) ---
__device__ void step_rk4(double t, double h, double y[], double M, double a, double mu) {
    double k1[6], k2[6], k3[6], k4[6];
    double y_temp[6];

    // K1
    calculate_derivs(t, y, k1, M, a, mu, true);

    // K2
    for(int i=0; i<6; i++) y_temp[i] = y[i] + 0.5 * h * k1[i];
    calculate_derivs(t + 0.5*h, y_temp, k2, M, a, mu, true);

    // K3
    for(int i=0; i<6; i++) y_temp[i] = y[i] + 0.5 * h * k2[i];
    calculate_derivs(t + 0.5*h, y_temp, k3, M, a, mu, true);

    // K4
    for(int i=0; i<6; i++) y_temp[i] = y[i] + h * k3[i];
    calculate_derivs(t + h, y_temp, k4, M, a, mu, true);

    // Update
    for(int i=0; i<6; i++) {
        y[i] += (h/6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
}

// --- 5. 全局 Kernel (__global__) ---
// 输入: d_params (N, 4) -> [M, a, mu, dt]
// 输入: d_init_state (N, 3) -> [p0, e0, iota0]
// 输出: d_output (N, steps, 9) -> 存储轨迹 [t, p, e, iota, psi, chi, phi, r, theta]
extern "C" __global__ void evolve_batch_kernel(
    int n_batch, int n_steps, double duration,
    const double* d_params,      // (N, 4)
    const double* d_init_state,  // (N, 3)
    double* d_output             // (N * steps * 9)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_batch) return;

    // 读取参数
    double M = d_params[idx*4 + 0];
    double a = d_params[idx*4 + 1];
    double mu = d_params[idx*4 + 2];
    double dt = d_params[idx*4 + 3]; // 采样步长

    // 状态向量
    double y[6];
    y[0] = d_init_state[idx*3 + 0]; // p
    y[1] = d_init_state[idx*3 + 1]; // e
    y[2] = d_init_state[idx*3 + 2]; // iota
    y[3] = 0.0; // psi
    y[4] = 0.0; // chi
    y[5] = 0.0; // phi

    // 积分循环
    // 假设积分步长 h 比采样步长 dt 小很多，这里简化设为 h = dt/10
    // 实际上应该在两帧之间做微步积分
    double h = dt * 0.1; 
    int micro_steps = 10;

    double t = 0.0;

    for (int step = 0; step < n_steps; step++) {
        // 记录当前步数据到全局内存
        // 输出格式: [t, p, e, iota, psi, chi, phi, r, theta]
        int out_base = idx * (n_steps * 9) + step * 9;
        
        // 计算 r, theta 用于输出 (简单的 mapping)
        double r_val = y[0] / (1.0 + y[1] * cos(y[3]));
        KerrConstantsGPU k = get_constants_gpu(M, a, y[0], y[1], y[2]);
        double z_val = k.z_minus * pow(cos(y[4]), 2); 
        // ... (这里需要应用你修复后的 theta 符号逻辑) ...
        double theta_val = acos(sqrt(d_max(0.0, z_val))); // 需修改

        d_output[out_base + 0] = t;
        d_output[out_base + 1] = y[0];
        d_output[out_base + 2] = y[1];
        d_output[out_base + 3] = y[2];
        d_output[out_base + 4] = y[3];
        d_output[out_base + 5] = y[4];
        d_output[out_base + 6] = y[5];
        d_output[out_base + 7] = r_val;
        d_output[out_base + 8] = theta_val;

        // 推进到下一个采样点
        for (int ms = 0; ms < micro_steps; ms++) {
            step_rk4(t, h, y, M, a, mu);
            t += h;
        }
    }
}