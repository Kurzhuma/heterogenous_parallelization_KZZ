cmake_minimum_required(VERSION 3.20)      # Минимальная версия CMake

project(Lab3 LANGUAGES CXX)                # Имя проекта

set(CMAKE_CXX_STANDARD 17)                 # Используем C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)        # Стандарт обязателен

find_package(OpenMP REQUIRED)              # Подключаем OpenMP

add_executable(Lab3                        # Создаём исполняемый файл
        practice1.cpp                           # Единственный cpp-файл
)

target_link_libraries(Lab3                 # Линкуем OpenMP
        OpenMP::OpenMP_CXX
)
