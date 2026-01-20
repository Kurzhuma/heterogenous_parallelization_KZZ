// Подключаем стандартную библиотеку для ввода и вывода данных
#include <iostream>

// Подключаем стандартную библиотеку для работы с динамическими массивами
#include <vector>

// Подключаем библиотеку для измерения времени выполнения
#include <chrono>

// Подключаем заголовочный файл CUDA Runtime API
#include <cuda_runtime.h>

// Определяем макрос для проверки корректности выполнения вызовов CUDA API
#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } } while(0)

// ================================
// KERNEL: Scan внутри одного блока
// ================================

// Объявляем CUDA-ядро для вычисления префиксной суммы внутри одного блока
__global__ void scan_block_kernel(const float* in, float* out, float* block_sums, int n)
{
    // Объявляем разделяемую память для хранения данных блока (до 1024 потоков)
    __shared__ float temp[1024];

    // Получаем локальный индекс потока внутри блока
    int tid = threadIdx.x;

    // Вычисляем глобальный индекс элемента массива
    int gid = blockIdx.x * blockDim.x + tid;

    // Загружаем входные данные в разделяемую память
    if (gid < n)
        temp[tid] = in[gid];
    else
        temp[tid] = 0.0f;

    // Синхронизируем все потоки блока перед началом вычислений
    __syncthreads();

    // Выполняем inclusive scan внутри блока
    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        // Объявляем временную переменную для хранения частичной суммы
        float val = 0.0f;

        // Если текущий поток имеет доступ к предыдущему элементу, читаем его
        if (tid >= offset)
            val = temp[tid - offset];

        // Синхронизируем потоки перед записью
        __syncthreads();

        // Добавляем полученное значение к текущему элементу
        temp[tid] += val;

        // Синхронизируем потоки после записи
        __syncthreads();
    }

    // Записываем результат сканирования в глобальную память
    if (gid < n)
        out[gid] = temp[tid];

    // Если поток является последним в блоке, сохраняем сумму блока
    if (tid == blockDim.x - 1)
        block_sums[blockIdx.x] = temp[tid];
}

// ================================
// KERNEL: Добавление префиксов блоков
// ================================

// Объявляем CUDA-ядро для добавления смещений блоков к каждому элементу
__global__ void add_block_offsets(float* data, const float* block_offsets, int n)
{
    // Вычисляем глобальный индекс элемента
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Если индекс допустим и блок не первый, добавляем смещение
    if (gid < n && blockIdx.x > 0)
        data[gid] += block_offsets[blockIdx.x - 1];
}

// Объявляем точку входа в программу
int main()
{
    // Задаём размер массива
    const int N = 1'000'000;

    // Задаём размер блока потоков
    const int BLOCK = 1024;

    // Вычисляем количество блоков
    const int GRID = (N + BLOCK - 1) / BLOCK;

    // Создаём и инициализируем входной массив на CPU
    std::vector<float> h(N, 1.0f);

    // Создаём массив для CPU-результата
    std::vector<float> cpu(N);

    // ================================
    // CPU prefix sum
    // ================================

    // Запоминаем время начала вычисления на CPU
    auto t1 = std::chrono::high_resolution_clock::now();

    // Инициализируем первый элемент префиксной суммы
    cpu[0] = h[0];

    // Последовательно вычисляем префиксную сумму на CPU
    for (int i = 1; i < N; ++i)
        cpu[i] = cpu[i - 1] + h[i];

    // Запоминаем время окончания вычисления на CPU
    auto t2 = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения CPU-версии в миллисекундах
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ================================
    // GPU
    // ================================

    // Объявляем указатели на массивы в памяти GPU
    float *d_in, *d_out, *d_block_sums, *d_block_scan;

    // Выделяем память под входной массив на GPU
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));

    // Выделяем память под выходной массив на GPU
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Выделяем память под массив сумм блоков
    CUDA_CHECK(cudaMalloc(&d_block_sums, GRID * sizeof(float)));

    // Выделяем память под массив просканированных сумм блоков
    CUDA_CHECK(cudaMalloc(&d_block_scan, GRID * sizeof(float)));

    // Копируем входные данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Создаём CUDA-события для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запускаем таймер GPU
    cudaEventRecord(start);

    // 1) Выполняем scan внутри каждого блока
    scan_block_kernel<<<GRID, BLOCK>>>(d_in, d_out, d_block_sums, N);

    // 2) Выполняем сканирование сумм блоков на CPU (упрощённая версия)

    // Создаём буфер на CPU для сумм блоков
    std::vector<float> h_block(GRID);

    // Копируем суммы блоков с GPU на CPU
    CUDA_CHECK(cudaMemcpy(h_block.data(), d_block_sums, GRID * sizeof(float), cudaMemcpyDeviceToHost));

    // Последовательно вычисляем префиксную сумму для блоков
    for (int i = 1; i < GRID; ++i)
        h_block[i] += h_block[i - 1];

    // Копируем просканированные суммы блоков обратно на GPU
    CUDA_CHECK(cudaMemcpy(d_block_scan, h_block.data(), GRID * sizeof(float), cudaMemcpyHostToDevice));

    // 3) Добавляем смещения блоков ко всем элементам
    add_block_offsets<<<GRID, BLOCK>>>(d_out, d_block_scan, N);

    // Останавливаем таймер GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Получаем время выполнения GPU-версии
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // ================================
    // Проверка результата
    // ================================

    // Создаём массив для хранения результата с GPU
    std::vector<float> gpu(N);

    // Копируем результат с GPU на CPU
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Выводим результаты работы программы
    std::cout << "Task 2: CUDA Prefix Sum (Shared Memory)\n";
    std::cout << "Array size: " << N << "\n";
    std::cout << "CPU time = " << cpu_ms << " ms\n";
    std::cout << "GPU time = " << gpu_ms << " ms\n";
    std::cout << "Last element CPU = " << cpu.back() << "\n";
    std::cout << "Last element GPU = " << gpu.back() << "\n";

    // ================================
    // Cleanup
    // ================================

    // Освобождаем память на GPU
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    cudaFree(d_block_scan);

    // Уничтожаем CUDA-события
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Возвращаем код успешного завершения программы
    return 0;
}
