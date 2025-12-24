# Минимальная версия CMake
cmake_minimum_required(VERSION 3.20)

# Объявляем проект с поддержкой C++ и CUDA
project(HeterogeneousParallelizationLab3 LANGUAGES CXX CUDA)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Устанавливаем стандарт CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Создаём исполняемый файл и добавляем CUDA-файл
add_executable(lab3
        src/main.cpp
        src/cpu_sorts.cpp
        src/gpu_sorts.cu
        src/gpu_sorts.cu
)

# Подключаем заголовочные файлы
target_include_directories(lab3 PRIVATE include)

# Указываем архитектуру GPU (безопасно для remote)
set_target_properties(lab3 PROPERTIES
        CUDA_ARCHITECTURES native
)
