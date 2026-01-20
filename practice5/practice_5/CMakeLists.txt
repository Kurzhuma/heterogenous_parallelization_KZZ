# Указываем минимально требуемую версию CMake для корректной сборки проекта
cmake_minimum_required(VERSION 3.18)

# Объявляем проект с именем practice5 и указываем, что используются языки CUDA и C++
project(practice5 CUDA CXX)

# Устанавливаем стандарт языка C++ версии 17 для всего проекта
set(CMAKE_CXX_STANDARD 17)

# Явно включаем поддержку языка CUDA в проекте
enable_language(CUDA)

# Создаём исполняемый файл practice5 и указываем исходные файлы проекта
add_executable(practice5
        main.cu
        gpu_structures.cu
)

# Включаем раздельную компиляцию CUDA, необходимую для корректной линковки нескольких .cu файлов
set_target_properties(practice5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


