cmake_minimum_required(VERSION 3.18)
project(practice8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

find_package(OpenMP REQUIRED)

add_executable(practice8 src/main.cu
        include/utils.h)

# КРИТИЧЕСКИ ВАЖНАЯ СТРОКА: Указывает путь к папке include
target_include_directories(practice8 PRIVATE include)

target_link_libraries(practice8 PRIVATE OpenMP::OpenMP_CXX)