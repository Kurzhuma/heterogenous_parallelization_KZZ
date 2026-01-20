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

// Объявляем CUDA-ядро для вычисления суммы элементов массива с использованием глобальной памяти
__global__ void reduce_kernel(const float* in, float* out, int n)
{
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Если индекс находится в допустимых пределах, добавляем элемент к общей сумме
    if (idx < n)
        atomicAdd(out, in[idx]);
}

// Объявляем точку входа в программу
int main()
{
    // Задаём размер массива
    const int N = 1'000'000;

    // Вычисляем половину размера массива для гибридной обработки
    const int HALF = N / 2;

    // Создаём и инициализируем входной массив на CPU
    std::vector<float> h(N, 1.0f);

    // ================================
    // CPU ONLY
    // ================================

    // Фиксируем время начала вычислений на CPU
    auto t1 = std::chrono::high_resolution_clock::now();

    // Объявляем переменную для хранения суммы, вычисляемой на CPU
    float cpu_sum = 0.0f;

    // Последовательно вычисляем сумму всех элементов массива на CPU
    for (int i = 0; i < N; ++i)
        cpu_sum += h[i];

    // Фиксируем время окончания вычислений на CPU
    auto t2 = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения CPU-версии в миллисекундах
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ================================
    // GPU ONLY
    // ================================

    // Объявляем указатели на массивы в памяти GPU
    float *d_in, *d_out;

    // Выделяем память на GPU под входной массив
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));

    // Выделяем память на GPU под выходную сумму
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // Копируем входные данные с CPU в память GPU
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Инициализируем выходную сумму на GPU нулевым значением
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // Объявляем CUDA-события для измерения времени выполнения
    cudaEvent_t gstart, gstop;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gstop);

    // Запускаем таймер GPU
    cudaEventRecord(gstart);

    // Запускаем CUDA-ядро для вычисления суммы на GPU
    reduce_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);

    // Останавливаем таймер GPU
    cudaEventRecord(gstop);

    // Ожидаем завершения выполнения ядра
    cudaEventSynchronize(gstop);

    // Объявляем переменную для хранения времени выполнения на GPU
    float gpu_ms;

    // Вычисляем время выполнения GPU-версии
    cudaEventElapsedTime(&gpu_ms, gstart, gstop);

    // Объявляем переменную для хранения суммы, вычисленной на GPU
    float gpu_sum;

    // Копируем результат суммы из памяти GPU в память CPU
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // ================================
    // HYBRID CPU + GPU
    // ================================

    // Сбрасываем значение выходной суммы на GPU перед гибридным вычислением
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // Фиксируем время начала гибридных вычислений
    auto h1 = std::chrono::high_resolution_clock::now();

    // ================================
    // CPU half
    // ================================

    // Объявляем переменную для хранения частичной суммы, вычисляемой на CPU
    float cpu_part = 0.0f;

    // Вычисляем сумму первой половины массива на CPU
    for (int i = 0; i < HALF; ++i)
        cpu_part += h[i];

    // ================================
    // GPU half
    // ================================

    // Копируем вторую половину массива в память GPU
    CUDA_CHECK(cudaMemcpy(d_in, h.data() + HALF, HALF * sizeof(float), cudaMemcpyHostToDevice));

    // Запускаем CUDA-ядро для вычисления суммы второй половины массива
    reduce_kernel<<<(HALF + 255) / 256, 256>>>(d_in, d_out, HALF);

    // Ожидаем завершения вычислений на GPU
    CUDA_CHECK(cudaDeviceSynchronize());

    // Объявляем переменную для хранения частичной суммы, вычисленной на GPU
    float gpu_part;

    // Копируем частичную сумму с GPU в память CPU
    CUDA_CHECK(cudaMemcpy(&gpu_part, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // Вычисляем итоговую гибридную сумму как сумму двух частичных результатов
    float hybrid_sum = cpu_part + gpu_part;

    // Фиксируем время окончания гибридных вычислений
    auto h2 = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения гибридной версии
    double hybrid_ms = std::chrono::duration<double, std::milli>(h2 - h1).count();

    // ================================
    // OUTPUT
    // ================================

    // Выводим заголовок задания
    std::cout << "Task 3: Hybrid CPU + GPU\n";

    // Выводим размер массива
    std::cout << "Array size: " << N << "\n\n";

    // Выводим результат и время выполнения CPU-версии
    std::cout << "CPU only:    sum = " << cpu_sum << ", time = " << cpu_ms << " ms\n";

    // Выводим результат и время выполнения GPU-версии
    std::cout << "GPU only:    sum = " << gpu_sum << ", time = " << gpu_ms << " ms\n";

    // Выводим результат и время выполнения гибридной версии
    std::cout << "HYBRID:      sum = " << hybrid_sum << ", time = " << hybrid_ms << " ms\n";

    // ================================
    // CLEANUP
    // ================================

    // Освобождаем память, выделенную на GPU под входной массив
    cudaFree(d_in);

    // Освобождаем память, выделенную на GPU под выходную сумму
    cudaFree(d_out);

    // Уничтожаем CUDA-событие начала измерения времени
    cudaEventDestroy(gstart);

    // Уничтожаем CUDA-событие окончания измерения времени
    cudaEventDestroy(gstop);

    // Возвращаем код успешного завершения программы
    return 0;
}
