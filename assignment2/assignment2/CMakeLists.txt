cmake_minimum_required(VERSION 3.18)          # Минимальная версия CMake с поддержкой CUDA
project(assignment2 LANGUAGES CXX CUDA)       # Проект с C++ и CUDA

set(CMAKE_CXX_STANDARD 17)                    # Стандарт C++17
set(CMAKE_CUDA_STANDARD 17)                   # Стандарт CUDA C++17

find_package(OpenMP REQUIRED)                 # Подключаем OpenMP

add_executable(cpu_openmp cpu_openmp.cpp)     # CPU-версия
target_link_libraries(cpu_openmp OpenMP::OpenMP_CXX)

add_executable(gpu_cuda gpu_cuda.cu)          # GPU-версия
