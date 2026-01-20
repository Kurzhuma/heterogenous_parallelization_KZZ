// Подключаем заголовочный файл с объявлениями функций работы с OpenCL
#include "opencl_utils.h"

// Подключаем основной заголовочный файл библиотеки OpenCL
#include <CL/cl.h>

// Подключаем стандартную библиотеку для работы с динамическими массивами
#include <vector>

// Подключаем стандартную библиотеку для ввода и вывода данных
#include <iostream>

// Подключаем стандартную библиотеку для работы с файлами
#include <fstream>

// Подключаем стандартную библиотеку для измерения времени выполнения
#include <chrono>

// Определяем вспомогательную функцию для загрузки исходного кода OpenCL-ядра из файла
static std::string loadKernel(const char* path)
{
    // Открываем файл с исходным кодом ядра
    std::ifstream file(path);

    // Считываем содержимое файла в строку и возвращаем её
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Определяем функцию для выбора подходящего OpenCL-устройства (CPU или GPU)
static cl_device_id getDevice(bool use_gpu)
{
    // Объявляем переменную для хранения количества доступных платформ OpenCL
    cl_uint numPlatforms = 0;

    // Запрашиваем количество доступных платформ OpenCL
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    // Создаём вектор для хранения идентификаторов платформ OpenCL
    std::vector<cl_platform_id> platforms(numPlatforms);

    // Получаем список всех доступных платформ OpenCL
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // Запускаем цикл по всем найденным платформам OpenCL
    for (cl_uint i = 0; i < numPlatforms; ++i)
    {
        // Объявляем переменную для хранения количества устройств на текущей платформе
        cl_uint numDevices = 0;

        // В зависимости от флага выбираем тип устройства: GPU или CPU
        cl_device_type type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

        // Запрашиваем количество устройств выбранного типа на текущей платформе
        cl_int err = clGetDeviceIDs(platforms[i], type, 0, nullptr, &numDevices);

        // Если устройств нет или произошла ошибка, переходим к следующей платформе
        if (err != CL_SUCCESS || numDevices == 0)
            continue;

        // Создаём вектор для хранения идентификаторов найденных устройств
        std::vector<cl_device_id> devices(numDevices);

        // Получаем список устройств выбранного типа на текущей платформе
        clGetDeviceIDs(platforms[i], type, numDevices, devices.data(), nullptr);

        // Возвращаем первое найденное подходящее устройство
        return devices[0];
    }

    // Выводим сообщение об ошибке, если подходящее устройство не найдено
    std::cerr << "No suitable OpenCL device found!" << std::endl;

    // Принудительно завершаем выполнение программы с кодом ошибки
    exit(1);
}

// Определяем функцию выполнения поэлементного сложения векторов через OpenCL
void run_vector_add(int N, bool use_gpu)
{
    // Выводим в консоль информацию о том, используется ли CPU или GPU
    std::cout << (use_gpu ? "OpenCL GPU Vector Add" : "OpenCL CPU Vector Add") << std::endl;

    // Получаем подходящее OpenCL-устройство
    auto device = getDevice(use_gpu);

    // Объявляем переменную для хранения кода ошибки OpenCL
    cl_int err;

    // Создаём OpenCL-контекст для выбранного устройства
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    // Создаём командную очередь для выбранного устройства
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Создаём векторы для входных данных и результата
    std::vector<float> A(N), B(N), C(N);

    // Инициализируем входные векторы тестовыми значениями
    for (int i = 0; i < N; ++i) { A[i] = i; B[i] = 2*i; }

    // Создаём буфер OpenCL для первого входного вектора
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, A.data(), nullptr);

    // Создаём буфер OpenCL для второго входного вектора
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, B.data(), nullptr);

    // Создаём буфер OpenCL для выходного вектора
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N, nullptr, nullptr);

    // Загружаем исходный код OpenCL-ядра из файла
    auto src = loadKernel("kernel_vector_add.cl");

    // Получаем указатель на строку с исходным кодом
    const char* csrc = src.c_str();

    // Получаем длину исходного кода
    size_t len = src.size();

    // Создаём программу OpenCL из исходного кода
    cl_program program = clCreateProgramWithSource(context, 1, &csrc, &len, &err);

    // Компилируем программу под выбранное устройство
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Создаём объект ядра из скомпилированной программы
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);

    // Передаём первый аргумент ядра (вектор A)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);

    // Передаём второй аргумент ядра (вектор B)
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);

    // Передаём третий аргумент ядра (вектор C)
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // Задаём глобальный размер рабочей области
    size_t global = N;

    // Фиксируем начальный момент времени перед запуском ядра
    auto t1 = std::chrono::high_resolution_clock::now();

    // Запускаем OpenCL-ядро на выполнение
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    // Ожидаем завершения всех команд в очереди
    clFinish(queue);

    // Фиксируем конечный момент времени после выполнения ядра
    auto t2 = std::chrono::high_resolution_clock::now();

    // Считываем результат вычислений из памяти устройства в память хоста
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float)*N, C.data(), 0, nullptr, nullptr);

    // Вычисляем время выполнения в миллисекундах
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Выводим время выполнения в консоль
    std::cout << "Time: " << ms << " ms" << std::endl;

    // Освобождаем буферы OpenCL
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Освобождаем объект ядра
    clReleaseKernel(kernel);

    // Освобождаем программу OpenCL
    clReleaseProgram(program);

    // Освобождаем командную очередь
    clReleaseCommandQueue(queue);

    // Освобождаем контекст OpenCL
    clReleaseContext(context);
}

// Определяем функцию выполнения умножения матриц через OpenCL
void run_matmul(int N, int M, int K, bool use_gpu)
{
    // Выводим в консоль информацию о том, используется ли CPU или GPU
    std::cout << (use_gpu ? "OpenCL GPU MatMul" : "OpenCL CPU MatMul") << std::endl;

    // Получаем подходящее OpenCL-устройство
    auto device = getDevice(use_gpu);

    // Объявляем переменную для хранения кода ошибки OpenCL
    cl_int err;

    // Создаём OpenCL-контекст для выбранного устройства
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    // Создаём командную очередь для выбранного устройства
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Создаём векторы для хранения матриц A, B и C
    std::vector<float> A(N*M), B(M*K), C(N*K);

    // Инициализируем матрицу A тестовыми значениями
    for (int i = 0; i < N*M; ++i) A[i] = 1.0f;

    // Инициализируем матрицу B тестовыми значениями
    for (int i = 0; i < M*K; ++i) B[i] = 1.0f;

    // Создаём буфер OpenCL для матрицы A
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N*M, A.data(), nullptr);

    // Создаём буфер OpenCL для матрицы B
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*M*K, B.data(), nullptr);

    // Создаём буфер OpenCL для матрицы C
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N*K, nullptr, nullptr);

    // Загружаем исходный код OpenCL-ядра умножения матриц из файла
    auto src = loadKernel("kernel_matmul.cl");

    // Получаем указатель на строку с исходным кодом
    const char* csrc = src.c_str();

    // Получаем длину исходного кода
    size_t len = src.size();

    // Создаём программу OpenCL из исходного кода
    cl_program program = clCreateProgramWithSource(context, 1, &csrc, &len, &err);

    // Компилируем программу под выбранное устройство
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Создаём объект ядра из скомпилированной программы
    cl_kernel kernel = clCreateKernel(program, "matmul", &err);

    // Передаём первый аргумент ядра (матрица A)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);

    // Передаём второй аргумент ядра (матрица B)
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);

    // Передаём третий аргумент ядра (матрица C)
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // Передаём размер N в качестве аргумента ядра
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // Передаём размер M в качестве аргумента ядра
    clSetKernelArg(kernel, 4, sizeof(int), &M);

    // Передаём размер K в качестве аргумента ядра
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // Задаём двумерный глобальный размер рабочей области
    size_t global[2] = { (size_t)N, (size_t)K };

    // Фиксируем начальный момент времени перед запуском ядра
    auto t1 = std::chrono::high_resolution_clock::now();

    // Запускаем OpenCL-ядро умножения матриц на выполнение
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);

    // Ожидаем завершения всех команд в очереди
    clFinish(queue);

    // Фиксируем конечный момент времени после выполнения ядра
    auto t2 = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения в миллисекундах
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Выводим время выполнения в консоль
    std::cout << "Time: " << ms << " ms" << std::endl;

    // Освобождаем буферы OpenCL
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Освобождаем объект ядра
    clReleaseKernel(kernel);

    // Освобождаем программу OpenCL
    clReleaseProgram(program);

    // Освобождаем командную очередь
    clReleaseCommandQueue(queue);

    // Освобождаем контекст OpenCL
    clReleaseContext(context);
}
