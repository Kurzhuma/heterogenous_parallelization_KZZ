cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(practice2 Practice2.cpp)

find_package(OpenMP REQUIRED)

target_link_libraries(practice2 PRIVATE OpenMP::OpenMP_CXX)

