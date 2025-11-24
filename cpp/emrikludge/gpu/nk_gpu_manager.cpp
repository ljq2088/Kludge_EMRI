// // cpp/emrikludge/gpu/nk_gpu_manager.cpp
// #include <cuda_runtime.h>
// #include <vector>
// #include <stdexcept>

// // 声明 kernel
// extern "C" __global__ void evolve_batch_kernel(
//     int n_batch, int n_steps, double duration,
//     const double* d_params, const double* d_init_state, double* d_output
// );

// // 包装函数：接收 std::vector (来自 numpy)，返回 std::vector
// std::vector<double> launch_nk_batch_gpu(
//     int n_batch, int n_steps, double duration,
//     const std::vector<double>& h_params,     // flatten (N, 4)
//     const std::vector<double>& h_init_state  // flatten (N, 3)
// ) {
//     size_t size_params = n_batch * 4 * sizeof(double);
//     size_t size_state  = n_batch * 3 * sizeof(double);
//     size_t size_output = n_batch * n_steps * 9 * sizeof(double);

//     // 1. 分配显存
//     double *d_params, *d_state, *d_output;
//     cudaMalloc(&d_params, size_params);
//     cudaMalloc(&d_state, size_state);
//     cudaMalloc(&d_output, size_output);

//     // 2. 拷贝数据 H2D
//     cudaMemcpy(d_params, h_params.data(), size_params, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_state, h_init_state.data(), size_state, cudaMemcpyHostToDevice);

//     // 3. 启动 Kernel
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (n_batch + threadsPerBlock - 1) / threadsPerBlock;
//     evolve_batch_kernel<<<blocksPerGrid, threadsPerBlock>>>(
//         n_batch, n_steps, duration, d_params, d_init_state, d_output
//     );
    
//     cudaDeviceSynchronize(); // 等待完成

//     // 4. 拷贝结果 D2H
//     std::vector<double> h_output(n_batch * n_steps * 9);
//     cudaMemcpy(h_output.data(), d_output, size_output, cudaMemcpyDeviceToHost);

//     // 5. 释放显存
//     cudaFree(d_params);
//     cudaFree(d_state);
//     cudaFree(d_output);

//     return h_output;
// }