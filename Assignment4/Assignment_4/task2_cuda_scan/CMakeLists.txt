# Указываем минимально допустимую версию CMake, необходимую для сборки данного проекта
cmake_minimum_required(VERSION 3.18)

# Объявляем проект с именем task2_cuda_scan с поддержкой языков CUDA и C++
project(task2_cuda_scan CUDA CXX)

# Устанавливаем стандарт языка C++ на уровне C++17
set(CMAKE_CXX_STANDARD 17)

# Явно включаем поддержку языка CUDA в рамках текущего проекта
enable_language(CUDA)

# Добавляем исполняемый файл проекта и указываем исходный CUDA-файл main.cu
add_executable(task2_cuda_scan
        main.cu
)

# Включаем раздельную компиляцию CUDA для корректной компоновки CUDA-кода
set_target_properties(task2_cuda_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

