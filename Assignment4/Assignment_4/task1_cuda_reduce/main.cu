// Подключаем стандартную библиотеку для ввода и вывода данных
#include <iostream>

// Подключаем стандартную библиотеку для работы с динамическими массивами
#include <vector>

// Подключаем стандартную библиотеку для измерения времени выполнения
#include <chrono>

// Подключаем заголовочный файл CUDA Runtime API
#include <cuda_runtime.h>

// Определяем макрос для проверки корректности выполнения вызовов CUDA API
#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } } while(0)

// Объявляем CUDA-ядро для вычисления суммы элементов массива с использованием глобальной памяти и atomicAdd
__global__ void reduce_global_kernel(const float* in, float* out, int n)
{
    // Вычисляем глобальный индекс текущего потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за пределы массива
    if (idx < n)

        // Атомарно добавляем значение текущего элемента массива к общей сумме в глобальной памяти
        atomicAdd(out, in[idx]);
}

// Определяем точку входа в программу
int main()
{
    // Задаём размер массива согласно условию задания
    const int N = 100000;

    // Создаём вектор на стороне CPU и инициализируем его значениями 1.0
    std::vector<float> h(N, 1.0f);

    // ---------------- CPU version ----------------

    // Фиксируем момент времени перед началом вычислений на CPU
    auto t1 = std::chrono::high_resolution_clock::now();

    // Инициализируем переменную для хранения суммы на CPU
    float cpu_sum = 0.0f;

    // Выполняем последовательное суммирование элементов массива на CPU
    for (int i = 0; i < N; ++i)
        cpu_sum += h[i];

    // Фиксируем момент времени после завершения вычислений на CPU
    auto t2 = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения CPU-версии в миллисекундах
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---------------- GPU version ----------------

    // Объявляем указатель на входной массив в памяти GPU
    float* d_in;

    // Объявляем указатель на переменную для хранения результата в памяти GPU
    float* d_out;

    // Выделяем память на GPU под входной массив
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));

    // Выделяем память на GPU под выходную переменную (сумму)
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // Копируем входные данные из памяти CPU в память GPU
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Инициализируем выходную переменную на GPU нулевым значением
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // Объявляем CUDA-события для измерения времени выполнения
    cudaEvent_t start, stop;

    // Создаём CUDA-событие начала измерения времени
    cudaEventCreate(&start);

    // Создаём CUDA-событие окончания измерения времени
    cudaEventCreate(&stop);

    // Записываем событие начала выполнения GPU-ядра
    cudaEventRecord(start);

    // Запускаем CUDA-ядро с вычислением суммы с использованием глобальной памяти
    reduce_global_kernel<<<(N + 255) / 256, 256>>>(d_in, d_out, N);

    // Записываем событие окончания выполнения GPU-ядра
    cudaEventRecord(stop);

    // Ожидаем завершения выполнения GPU-ядра
    cudaEventSynchronize(stop);

    // Объявляем переменную для хранения времени выполнения на GPU
    float gpu_ms;

    // Вычисляем прошедшее время между событиями start и stop
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // Объявляем переменную для хранения результата суммирования на GPU
    float gpu_sum;

    // Копируем результат вычислений из памяти GPU в память CPU
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- Output ----------------

    // Выводим в консоль заголовок текущего задания
    std::cout << "Task 1: CUDA Reduce (Global Memory)\n";

    // Выводим размер массива
    std::cout << "Array size: " << N << "\n";

    // Выводим результат и время выполнения для CPU-версии
    std::cout << "CPU sum = " << cpu_sum << ", time = " << cpu_ms << " ms\n";

    // Выводим результат и время выполнения для GPU-версии
    std::cout << "GPU sum = " << gpu_sum << ", time = " << gpu_ms << " ms\n";

    // ---------------- Cleanup ----------------

    // Освобождаем память входного массива на GPU
    cudaFree(d_in);

    // Освобождаем память выходной переменной на GPU
    cudaFree(d_out);

    // Уничтожаем CUDA-событие начала измерения времени
    cudaEventDestroy(start);

    // Уничтожаем CUDA-событие окончания измерения времени
    cudaEventDestroy(stop);

    // Возвращаем код успешного завершения программы
    return 0;
}
