// Подключаем стандартную библиотеку для потоков ввода-вывода
#include <iostream>

// Подключаем библиотеку для работы с динамическими массивами std::vector
#include <vector>

// Подключаем CUDA Runtime API для работы с GPU
#include <cuda_runtime.h>

// Определяем константу размера массива
#define N 1000000





// ================================
// Функция проверки ошибок CUDA
// ================================

// Объявляем функцию, принимающую код ошибки CUDA
void checkCuda(cudaError_t err)
{
    // Проверяем, успешно ли завершилась CUDA-функция
    if (err != cudaSuccess)
    {
        // Выводим текстовое описание ошибки CUDA
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        // Прерываем выполнение программы, так как дальнейшая работа некорректна
        exit(1);
    }
}

// ================================
// ЗАДАНИЕ 1: Умножение (global)
// ================================

// Объявляем CUDA-ядро для умножения элементов с использованием глобальной памяти
__global__ void multiply_global(float* data, float value)
{
    // Вычисляем глобальный индекс текущего потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверяем, что индекс не выходит за пределы массива
    if (idx < N)
        // Умножаем элемент массива на заданное значение
        data[idx] = data[idx] * value;
}

// ================================
// ЗАДАНИЕ 1: Умножение (shared)
// ================================

// Объявляем CUDA-ядро для умножения элементов с использованием shared memory
__global__ void multiply_shared(float* data, float value)
{
    // Объявляем массив в разделяемой памяти для текущего блока
    __shared__ float buffer[256];

    // Вычисляем глобальный индекс текущего потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Получаем локальный индекс потока внутри блока
    int tid = threadIdx.x;

    // Если индекс в пределах массива — копируем данные из глобальной памяти в shared
    if (idx < N)
        buffer[tid] = data[idx];

    // Синхронизируем все потоки блока перед вычислениями
    __syncthreads();

    // Выполняем умножение в разделяемой памяти
    if (idx < N)
        buffer[tid] = buffer[tid] * value;

    // Синхронизируем потоки перед записью результата
    __syncthreads();

    // Записываем результат обратно в глобальную память
    if (idx < N)
        data[idx] = buffer[tid];
}

// ================================
// ЗАДАНИЕ 2: Сложение векторов
// ================================

// Объявляем CUDA-ядро для поэлементного сложения двух массивов
__global__ void vector_add(float* a, float* b, float* c)
{
    // Вычисляем глобальный индекс текущего потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверяем, что индекс не выходит за пределы массива
    if (idx < N)
        // Записываем сумму элементов массивов a и b в массив c
        c[idx] = a[idx] + b[idx];
}

// ================================
// ЗАДАНИЕ 3: Коалесцированный доступ
// ================================

// Объявляем CUDA-ядро с последовательным (коалесцированным) доступом к памяти
__global__ void coalesced_access(float* data)
{
    // Вычисляем глобальный индекс текущего потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверяем, что индекс не выходит за пределы массива
    if (idx < N)
        // Умножаем элемент массива на 2 с последовательным доступом
        data[idx] = data[idx] * 2.0f;
}

// ================================
// ЗАДАНИЕ 3: Некоалесцированный доступ
// ================================

// Объявляем CUDA-ядро с разрозненным (некоалесцированным) доступом к памяти
__global__ void noncoalesced_access(float* data)
{
    // Вычисляем глобальный индекс текущего потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Проверяем, что индекс не выходит за пределы массива
    if (idx < N)
    {
        // Вычисляем "плохой" индекс, нарушающий последовательность доступа
        int bad_idx = (idx * 32) % N;
        // Выполняем операцию по разрозненному адресу памяти
        data[bad_idx] = data[bad_idx] * 2.0f;
    }
}

// ================================
// Функции замера времени
// ================================

// Функция измерения времени выполнения ядра с коалесцированным доступом
float measure_coalesced(float* d_data, dim3 grid, dim3 block)
{
    // Объявляем CUDA-события начала и конца
    cudaEvent_t start, stop;
    // Создаем событие начала
    cudaEventCreate(&start);
    // Создаем событие конца
    cudaEventCreate(&stop);

    // Записываем событие начала
    cudaEventRecord(start);
    // Запускаем CUDA-ядро
    coalesced_access<<<grid, block>>>(d_data);
    // Проверяем наличие ошибок запуска ядра
    checkCuda(cudaGetLastError());
    // Ждем завершения выполнения ядра
    checkCuda(cudaDeviceSynchronize());
    // Записываем событие окончания
    cudaEventRecord(stop);
    // Ожидаем завершения события
    cudaEventSynchronize(stop);

    // Объявляем переменную для хранения времени
    float ms;
    // Вычисляем прошедшее время между событиями
    cudaEventElapsedTime(&ms, start, stop);

    // Удаляем событие начала
    cudaEventDestroy(start);
    // Удаляем событие окончания
    cudaEventDestroy(stop);
    // Возвращаем измеренное время
    return ms;
}

// Функция измерения времени выполнения ядра с некоалесцированным доступом
float measure_noncoalesced(float* d_data, dim3 grid, dim3 block)
{
    // Объявляем CUDA-события начала и конца
    cudaEvent_t start, stop;
    // Создаем событие начала
    cudaEventCreate(&start);
    // Создаем событие конца
    cudaEventCreate(&stop);

    // Записываем событие начала
    cudaEventRecord(start);
    // Запускаем CUDA-ядро
    noncoalesced_access<<<grid, block>>>(d_data);
    // Проверяем наличие ошибок запуска ядра
    checkCuda(cudaGetLastError());
    // Ждем завершения выполнения ядра
    checkCuda(cudaDeviceSynchronize());
    // Записываем событие окончания
    cudaEventRecord(stop);
    // Ожидаем завершения события
    cudaEventSynchronize(stop);

    // Объявляем переменную для хранения времени
    float ms;
    // Вычисляем прошедшее время между событиями
    cudaEventElapsedTime(&ms, start, stop);

    // Удаляем событие начала
    cudaEventDestroy(start);
    // Удаляем событие окончания
    cudaEventDestroy(stop);
    // Возвращаем измеренное время
    return ms;
}

// ================================
// main
// ================================

// Точка входа в программу
int main()
{
    // Переменная для хранения количества CUDA-устройств
    int deviceCount = 0;
    // Получаем количество CUDA-устройств в системе
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    // Проверяем, не произошла ли ошибка
    if (err != cudaSuccess)
    {
        // Выводим сообщение об ошибке
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        // Завершаем программу с кодом ошибки
        return 1;
    }

    // Выводим количество найденных CUDA-устройств
    std::cout << "CUDA devices: " << deviceCount << std::endl;

    // Выбираем первое CUDA-устройство
    checkCuda(cudaSetDevice(0));
    // Принудительно инициализируем CUDA-контекст
    checkCuda(cudaFree(0));

    // Создаем вектор данных для основного массива
    std::vector<float> h_data(N, 1.0f);
    // Создаем вектор данных для массива A
    std::vector<float> h_a(N, 1.0f);
    // Создаем вектор данных для массива B
    std::vector<float> h_b(N, 2.0f);
    // Создаем вектор данных для массива C
    std::vector<float> h_c(N, 0.0f);

    // Объявляем указатель на массив в памяти GPU для основного массива
    float* d_data;
    // Объявляем указатель на массив A в памяти GPU
    float* d_a;
    // Объявляем указатель на массив B в памяти GPU
    float* d_b;
    // Объявляем указатель на массив C в памяти GPU
    float* d_c;

    // Выделяем память на GPU для основного массива
    checkCuda(cudaMalloc(&d_data, N * sizeof(float)));
    // Выделяем память на GPU для массива A
    checkCuda(cudaMalloc(&d_a, N * sizeof(float)));
    // Выделяем память на GPU для массива B
    checkCuda(cudaMalloc(&d_b, N * sizeof(float)));
    // Выделяем память на GPU для массива C
    checkCuda(cudaMalloc(&d_c, N * sizeof(float)));

    // Копируем основной массив с CPU на GPU
    checkCuda(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    // Копируем массив A с CPU на GPU
    checkCuda(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    // Копируем массив B с CPU на GPU
    checkCuda(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Определяем плохой размер блока (32 потока)
    dim3 badBlock(32);
    // Определяем хороший размер блока (256 потоков)
    dim3 goodBlock(256);

    // Вычисляем размер сетки для плохого блока
    dim3 badGrid((N + badBlock.x - 1) / badBlock.x);
    // Вычисляем размер сетки для хорошего блока
    dim3 goodGrid((N + goodBlock.x - 1) / goodBlock.x);

    // ================================
    // Задание 1
    // ================================

    // Создаем CUDA-события для замера времени
    cudaEvent_t s1, e1;
    // Создаем событие начала
    cudaEventCreate(&s1);
    // Создаем событие конца
    cudaEventCreate(&e1);

    // Запускаем таймер
    cudaEventRecord(s1);
    // Запускаем ядро умножения через глобальную память
    multiply_global<<<goodGrid, goodBlock>>>(d_data, 2.0f);
    // Проверяем наличие ошибок
    checkCuda(cudaGetLastError());
    // Ждем завершения выполнения
    checkCuda(cudaDeviceSynchronize());
    // Останавливаем таймер
    cudaEventRecord(e1);
    // Ждем завершения события
    cudaEventSynchronize(e1);

    // Объявляем переменную для хранения времени
    float t_global;
    // Получаем измеренное время
    cudaEventElapsedTime(&t_global, s1, e1);

    // Создаем события для замера времени shared версии
    cudaEvent_t s2, e2;
    // Создаем событие начала
    cudaEventCreate(&s2);
    // Создаем событие конца
    cudaEventCreate(&e2);

    // Запускаем таймер
    cudaEventRecord(s2);
    // Запускаем ядро умножения через shared memory
    multiply_shared<<<goodGrid, goodBlock>>>(d_data, 2.0f);
    // Проверяем наличие ошибок
    checkCuda(cudaGetLastError());
    // Ждем завершения выполнения
    checkCuda(cudaDeviceSynchronize());
    // Останавливаем таймер
    cudaEventRecord(e2);
    // Ждем завершения события
    cudaEventSynchronize(e2);

    // Объявляем переменную для хранения времени
    float t_shared;
    // Получаем измеренное время
    cudaEventElapsedTime(&t_shared, s2, e2);

    // Выводим время выполнения глобальной версии
    std::cout << "Task 1: Global memory time: " << t_global << " ms" << std::endl;
    // Выводим время выполнения версии с shared memory
    std::cout << "Task 1: Shared memory time: " << t_shared << " ms" << std::endl;

    // ================================
    // Задание 2
    // ================================

    // Массив тестируемых размеров блока
    int blockSizes[3] = {64, 128, 256};

    // Цикл по размерам блоков
    for (int i = 0; i < 3; ++i)
    {
        // Формируем размер блока
        dim3 blk(blockSizes[i]);
        // Формируем размер сетки
        dim3 grd((N + blk.x - 1) / blk.x);

        // Создаем события
        cudaEvent_t s, e;
        // Создаем событие начала
        cudaEventCreate(&s);
        // Создаем событие конца
        cudaEventCreate(&e);

        // Запускаем таймер
        cudaEventRecord(s);
        // Запускаем ядро сложения векторов
        vector_add<<<grd, blk>>>(d_a, d_b, d_c);
        // Проверяем наличие ошибок
        checkCuda(cudaGetLastError());
        // Ждем завершения выполнения
        checkCuda(cudaDeviceSynchronize());
        // Останавливаем таймер
        cudaEventRecord(e);
        // Ждем завершения события
        cudaEventSynchronize(e);

        // Объявляем переменную для хранения времени
        float t;
        // Получаем измеренное время
        cudaEventElapsedTime(&t, s, e);

        // Выводим результат для текущего размера блока
        std::cout << "Task 2: Block size " << blockSizes[i] << " time: " << t << " ms" << std::endl;
    }

    // ================================
    // Задание 3
    // ================================

    // Измеряем время выполнения коалесцированного варианта
    float t_coalesced = measure_coalesced(d_data, goodGrid, goodBlock);
    // Измеряем время выполнения некоалесцированного варианта
    float t_noncoalesced = measure_noncoalesced(d_data, goodGrid, goodBlock);

    // Выводим результаты
    std::cout << "Task 3: Coalesced access time: " << t_coalesced << " ms" << std::endl;
    std::cout << "Task 3: Non-coalesced access time: " << t_noncoalesced << " ms" << std::endl;

    // ================================
    // Задание 4
    // ================================

    // Объявляем переменные для хранения времени
    float t_bad, t_good;

    // Создаем события для плохой конфигурации
    cudaEvent_t sb, eb;
    // Создаем событие начала
    cudaEventCreate(&sb);
    // Создаем событие конца
    cudaEventCreate(&eb);

    // Запускаем таймер
    cudaEventRecord(sb);
    // Запускаем ядро с плохой конфигурацией
    coalesced_access<<<badGrid, badBlock>>>(d_data);
    // Проверяем наличие ошибок
    checkCuda(cudaGetLastError());
    // Ждем завершения выполнения
    checkCuda(cudaDeviceSynchronize());
    // Останавливаем таймер
    cudaEventRecord(eb);
    // Ждем завершения события
    cudaEventSynchronize(eb);
    // Получаем время выполнения
    cudaEventElapsedTime(&t_bad, sb, eb);

    // Создаем события для хорошей конфигурации
    cudaEvent_t sg, eg;
    // Создаем событие начала
    cudaEventCreate(&sg);
    // Создаем событие конца
    cudaEventCreate(&eg);

    // Запускаем таймер
    cudaEventRecord(sg);
    // Запускаем ядро с хорошей конфигурацией
    coalesced_access<<<goodGrid, goodBlock>>>(d_data);
    // Проверяем наличие ошибок
    checkCuda(cudaGetLastError());
    // Ждем завершения выполнения
    checkCuda(cudaDeviceSynchronize());
    // Останавливаем таймер
    cudaEventRecord(eg);
    // Ждем завершения события
    cudaEventSynchronize(eg);
    // Получаем время выполнения
    cudaEventElapsedTime(&t_good, sg, eg);

    // Выводим время для плохой конфигурации
    std::cout << "Task 4: Bad configuration time: " << t_bad << " ms" << std::endl;
    // Выводим время для хорошей конфигурации
    std::cout << "Task 4: Good configuration time: " << t_good << " ms" << std::endl;

    // Освобождаем память на GPU для основного массива
    cudaFree(d_data);
    // Освобождаем память на GPU для массива A
    cudaFree(d_a);
    // Освобождаем память на GPU для массива B
    cudaFree(d_b);
    // Освобождаем память на GPU для массива C
    cudaFree(d_c);

    // Завершаем программу с кодом 0 (успех)
    return 0;
}
