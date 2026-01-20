# Указываем минимально допустимую версию CMake, необходимую для корректной сборки проекта
cmake_minimum_required(VERSION 3.18)

# Объявляем проект с именем practice6 и указываем, что используется язык программирования C++
project(practice6 CXX)

# Устанавливаем стандарт языка C++ версии 17 для всего проекта
set(CMAKE_CXX_STANDARD 17)

# Ищем установленную в системе библиотеку OpenCL и делаем её обязательной для сборки проекта
find_package(OpenCL REQUIRED)

# Создаём исполняемый файл practice6 и указываем исходные файлы, из которых он будет собран
add_executable(practice6
        main.cpp
        opencl_utils.cpp
)

# Указываем путь к заголовочным файлам OpenCL для текущей цели сборки
target_include_directories(practice6 PRIVATE ${OpenCL_INCLUDE_DIRS})

# Подключаем библиотеки OpenCL на этапе линковки исполняемого файла
target_link_libraries(practice6 PRIVATE ${OpenCL_LIBRARIES})
