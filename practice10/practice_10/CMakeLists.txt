cmake_minimum_required(VERSION 3.18)
# Явное указание языков: CXX (для OpenMP/MPI) и CUDA
project(PracticalWork10 LANGUAGES CXX CUDA)

# Настройки стандартов программирования
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Настройка архитектуры под ваши карты (RTX 20-й и 30-й серий)
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# 1. Поиск необходимых пакетов в системе
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# ---------------------------------------------------------
# ЗАДАНИЕ 1: OpenMP (CPU Parallelism)
add_executable(task1_omp src/task1_omp.cpp)
# Подключение библиотек OpenMP
target_link_libraries(task1_omp PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------
# ЗАДАНИЕ 2: CUDA Memory Patterns
add_executable(task2_cuda src/task2_cuda.cu)
# Подключение папки с utils.h
target_include_directories(task2_cuda PRIVATE include)

# ---------------------------------------------------------
# ЗАДАНИЕ 3: Hybrid (CUDA + OpenMP)
add_executable(task3_hyb src/task3_hyb.cu)
# Подключение OpenMP и папки include
target_include_directories(task3_hyb PRIVATE include)
target_link_libraries(task3_hyb PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------
# ЗАДАНИЕ 4: MPI Scaling
add_executable(task4_mpi src/task4_mpi.cpp)
# Использование переменных MPI для заголовочных файлов и линковки
target_include_directories(task4_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(task4_mpi PRIVATE ${MPI_CXX_LIBRARIES})