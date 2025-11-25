import os
import sys
import glob
from setuptools import setup, Extension, find_packages

# 尝试导入 pybind11，这是编译 C++ 扩展所必需的
try:
    import pybind11
except ImportError:
    print("Error: pybind11 is required to build this extension.")
    print("Please run: pip install pybind11")
    sys.exit(1)

# -------------------------------------------------------------------------
# C++ 源文件配置
# -------------------------------------------------------------------------
# 定义 C++ 源码根目录
CPP_SRC_ROOT = "cpp/emrikludge"

# 递归查找所有 .cpp 文件，防止漏掉依赖 (如 constants.cpp, kerr_geometry.cpp 等)
# 注意：我们排除了 main 函数入口文件(如果有)或测试文件，只包含库实现
cpp_sources = glob.glob(os.path.join(CPP_SRC_ROOT, "**", "*.cpp"), recursive=True)

# 打印一下找到的源文件，方便调试
print(f"Found {len(cpp_sources)} C++ source files.")

# -------------------------------------------------------------------------
# 扩展模块配置
# -------------------------------------------------------------------------
ext_modules = [
    Extension(
        # 模块全名：这决定了编译后的 .so 文件会放在 src/emrikludge/ 目录下
        name="emrikludge._emrikludge",
        
        # 源文件列表
        sources=[
    # Bindings (注意顺序，虽然编译不分先后)
    "cpp/emrikludge/bindings/bindings_module.cpp", # <--- 新的总入口
    # "cpp/emrikludge/bindings/bindings_common.cpp",
    "cpp/emrikludge/bindings/bindings_nk.cpp",
    "cpp/emrikludge/bindings/bindings_aak.cpp",
    # "cpp/emrikludge/bindings/bindings_ak.cpp",
    # "cpp/emrikludge/bindings/bindings_tdi.cpp",
    
    # Core Logic
    "cpp/emrikludge/emri_params.cpp",
    "cpp/emrikludge/constants.cpp",
    "cpp/emrikludge/orbit/nk_orbit.cpp",
    "cpp/emrikludge/orbit/aak_orbit.cpp", # <--- 确保包含
    "cpp/emrikludge/waveform/nk_waveform.cpp",
    "cpp/emrikludge/waveform/aak_waveform.cpp",
    "cpp/emrikludge/numrec/numrec.cpp",
    # ...
],
        
        # 头文件搜索路径
        include_dirs=[
            # 允许代码中使用 #include "orbit/nk_orbit.hpp" 这种相对路径
            "cpp",
            "cpp/emrikludge", 
            # pybind11 的头文件路径
            pybind11.get_include(),
            pybind11.get_include(user=True),
        ],
        library_dirs=[
            # ⚠️ 告诉链接器去哪里找 libgsl
            # os.path.join(sys.prefix, 'lib') 
        ],
        libraries=['gsl', 'gslcblas', 'm'], # 关键：链接 gsl, gslcblas 和 math 库
        # 编译参数
        language="c++",
        extra_compile_args=[
            "-std=c++17",  # 使用 C++17 标准 (或 c++14/11)
            "-O3",         # 开启最高优化，这对数值计算至关重要
            "-fPIC",       # 生成位置无关代码
            "-w"           # (可选) 忽略一些非致命警告，让输出清爽点
        ],
    ),
]
# setup.py 修改建议

# ...
def customize_compiler_for_nvcc(compiler):
    # 这是一个简化的 hook，用于把 .cu 文件传给 nvcc
    # 在实际生产中通常比较复杂，或者直接用 CMake
    pass 

# 或者，最务实的做法：使用 CMake 构建 (pybind11 官方推荐)
# 但为了保持你现有的 setup.py 结构，可以在 Extension 中添加库
nk_module = Extension(
    'emrikludge._emrikludge',
    sources=[
        'cpp/emrikludge/gpu/nk_kernel.cu',  # 新增
        'cpp/emrikludge/gpu/nk_gpu_manager.cpp', # 新增
        # ... 原有的 cpp 文件 ...
    ],
    include_dirs=[
        'cpp/emrikludge',
        '/usr/local/cuda/include' # 确保包含 CUDA 头文件
    ],
    library_dirs=['/usr/local/cuda/lib64'],
    libraries=['cudart'], # 链接 CUDA Runtime
    # ...
)
# -------------------------------------------------------------------------
# Setup 主函数
# -------------------------------------------------------------------------
setup(
    name="emrikludge",
    version="0.1.0",
    description="EMRI Kludge Waveforms with C++ Acceleration",
    
    # 自动查找 Python 包 (在 src 目录下)
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # 包含 C++ 扩展
    ext_modules=ext_modules,
    
    # 依赖项
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pybind11>=2.6.0",
    ],
    
    # 包含非 Python 文件 (如 .so, .hpp 等，如果在 MANIFEST.in 中定义了)
    include_package_data=True,
    zip_safe=False,
)