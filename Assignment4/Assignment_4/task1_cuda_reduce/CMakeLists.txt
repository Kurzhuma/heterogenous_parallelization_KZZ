# Указываем минимально требуемую версию CMake для корректной сборки проекта
cmake_minimum_required(VERSION 3.18)

# Объявляем проект с именем task1_cuda_reduce с поддержкой языков CUDA и C++
project(task1_cuda_reduce CUDA CXX)

# Устанавливаем стандарт языка C++ на уровне C++17
set(CMAKE_CXX_STANDARD 17)

# Явно включаем поддержку языка CUDA в проекте
enable_language(CUDA)

# Добавляем исполняемый файл проекта и указываем исходный файл main.cu
add_executable(task1_cuda_reduce
        main.cu
)

# Включаем раздельную компиляцию CUDA для корректной линковки и сборки CUDA-кода
set_target_properties(task1_cuda_reduce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

