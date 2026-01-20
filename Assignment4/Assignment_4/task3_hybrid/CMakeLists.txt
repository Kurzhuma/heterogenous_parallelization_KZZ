# Указываем минимально допустимую версию CMake, необходимую для корректной сборки проекта
cmake_minimum_required(VERSION 3.18)

# Объявляем проект с именем task3_hybrid_cpu_gpu с поддержкой языков CUDA и C++
project(task3_hybrid_cpu_gpu CUDA CXX)

# Устанавливаем стандарт языка C++ на уровне C++17
set(CMAKE_CXX_STANDARD 17)

# Явно включаем поддержку языка CUDA в рамках текущего проекта
enable_language(CUDA)

# Добавляем исполняемый файл проекта и указываем исходный файл main.cu
add_executable(task3_hybrid_cpu_gpu
        main.cu
)

# Включаем раздельную компиляцию CUDA для корректной компоновки CUDA-кода
set_target_properties(task3_hybrid_cpu_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
