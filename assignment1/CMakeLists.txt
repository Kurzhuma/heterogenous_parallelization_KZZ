cmake_minimum_required(VERSION 3.20)
project(lab1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Assignment1 assignment1.cpp)

find_package(OpenMP REQUIRED)

target_link_libraries(Assignment1 PRIVATE OpenMP::OpenMP_CXX)

