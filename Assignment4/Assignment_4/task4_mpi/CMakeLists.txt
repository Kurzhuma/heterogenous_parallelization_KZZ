# Указываем минимально допустимую версию CMake
cmake_minimum_required(VERSION 3.18)

# Определяем имя проекта и используемый язык программирования (C++)
project(task4_mpi_reduce CXX)

# Устанавливаем стандарт языка C++
set(CMAKE_CXX_STANDARD 17)

# Явно задаём путь к заголовочным файлам MS-MPI
set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")

# Явно задаём путь к библиотеке MS-MPI для 64-битной архитектуры
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

# Создаём исполняемый файл проекта и указываем исходный файл
add_executable(task4_mpi_reduce
        main.cpp
)

# Подключаем каталог с заголовочными файлами MPI к цели сборки
target_include_directories(task4_mpi_reduce PRIVATE ${MPI_CXX_INCLUDE_PATH})

# Линкуем исполняемый файл с библиотекой MS-MPI
target_link_libraries(task4_mpi_reduce PRIVATE ${MPI_CXX_LIBRARIES})


