cmake_minimum_required(VERSION 3.18)
project(practice9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Поиск пакета MPI
find_package(MPI REQUIRED)

add_executable(practice9 src/main.cpp)

# Подключение заголовочных файлов и библиотек MPI
target_include_directories(practice9 PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(practice9 PRIVATE ${MPI_CXX_LIBRARIES})