/**
 * ЗАДАНИЕ 3. ПРОФИЛИРОВАНИЕ ГИБРИДНОГО ПРИЛОЖЕНИЯ CPU + GPU
 */

// Подключение библиотеки ввода-вывода
#include <iostream>
// Подключение заголовочных файлов CUDA
#include <cuda_runtime.h>
// Подключение интерфейса OpenMP для параллельных вычислений на CPU
#include <omp.h>
// Подключение системных функций Windows
#include <windows.h>
// Подключение макросов проверки ошибок CUDA
#include "utils.h"

int main() {
    // Установка кодировки вывода UTF-8 для Windows
    SetConsoleOutputCP(65001);

    // Определение размера массива данных (4 миллиона элементов)
    const int N = 1 << 22;
    // Объявление указателей для хоста и устройства
    float *h_data, *d_data;

    // Выделение "закрепленной" (pinned) памяти на хосте для асинхронных операций
    CHECK_CUDA(cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault));
    // Выделение глобальной памяти на стороне графического ускорителя
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    // Создание объекта асинхронного потока исполнения CUDA Stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Регистрация времени начала комплексной гибридной операции
    double t_start = omp_get_wtime();

    // Инициализация асинхронного копирования данных из CPU в GPU через поток
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Разделение нагрузки: запуск параллельной секции OpenMP на центральном процессоре
    #pragma omp parallel
    {
        // Проверка номера текущего потока для вывода информационного сообщения
        if(omp_get_thread_num() == 0) {
            std::cout << "CPU инициировал расчеты параллельно с передачей данных на GPU..." << std::endl;
        }
        // Имитация полезной нагрузки на CPU (цикл с интенсивными вычислениями)
        for(int i = 0; i < 1000000; ++i) { volatile int dummy = i * i; }
    }

    // Принудительная блокировка до завершения всех операций в созданном потоке GPU
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Фиксация общего времени завершения гибридного алгоритма
    double t_end = omp_get_wtime();

    // Вывод итоговых временных показателей профилирования
    std::cout << "Общее время гибридной обработки данных: " << t_end - t_start << " сек." << std::endl;

    // Освобождение ресурсов потока и памяти
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFreeHost(h_data);

    return 0;
}