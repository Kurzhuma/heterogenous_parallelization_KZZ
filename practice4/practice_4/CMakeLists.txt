# Указываем минимально допустимую версию CMake, необходимую для сборки проекта
cmake_minimum_required(VERSION 3.18)

# Объявляем проект с именем practice4 и указываем, что используются языки CUDA и C++
project(practice4 CUDA CXX)

# Устанавливаем стандарт C++17 для компиляции C++ кода проекта
set(CMAKE_CXX_STANDARD 17)

# Явно включаем поддержку языка CUDA в проекте
enable_language(CUDA)

# Объявляем исполняемый файл practice4 и перечисляем исходные файлы проекта
add_executable(practice4
        main.cu
        gpu_memory.cu
)

# Включаем раздельную компиляцию CUDA, необходимую для корректной линковки нескольких .cu файлов
set_target_properties(practice4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

