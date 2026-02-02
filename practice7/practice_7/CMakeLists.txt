# Указываем минимально допустимую версию CMake
cmake_minimum_required(VERSION 3.18)

# Объявляем проект и указываем поддержку C++ и CUDA
# Добавление CUDA здесь критически важно для CLion
project(practice7 CXX CUDA)

# Устанавливаем стандарт языка C++ версии 17
set(CMAKE_CXX_STANDARD 17)

# Настройка архитектуры для вашей видеокарты RTX
# 75-86 охватывает почти все современные карты GeForce RTX
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Ищем OpenMP (нужен для точных замеров времени CPU/GPU в коде)
find_package(OpenMP REQUIRED)

# Создаём исполняемый файл и указываем путь к вашему main.cu
# Файлы .cu CLion теперь будет распознавать корректно
add_executable(practice7
        src/main.cu
)

# Указываем путь к папке с заголовочным файлом utils.h
target_include_directories(practice7 PRIVATE include)

# Подключаем библиотеки OpenMP для корректной линковки
target_link_libraries(practice7 PRIVATE OpenMP::OpenMP_CXX)