//
// Practice lab 3
//

// Подключаем CUDA runtime API для работы с GPU
#include <cuda_runtime.h>

// Подключаем заголовок с параметрами запуска CUDA-ядер
#include <device_launch_parameters.h>

// Подключаем стандартный контейнер vector
#include <vector>

// Подключаем стандартный ввод-вывод
#include <iostream>

// Объявляем CUDA-ядро для блочной параллельной сортировки (bitonic sort)
__global__ void block_sort(int* data, int n)
{
    // Получаем индекс потока внутри блока
    int tid = threadIdx.x;

    // Вычисляем глобальный индекс элемента массива
    int idx = blockIdx.x * blockDim.x + tid;

    // Проверяем, что индекс не выходит за границы массива
    if (idx < n)
    {
        // Запускаем внешний цикл bitonic sort
        for (int k = 2; k <= blockDim.x; k <<= 1)
        {
            // Запускаем внутренний цикл bitonic sort
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                // Вычисляем индекс элемента для сравнения
                int ixj = idx ^ j;

                // Проверяем корректность индекса сравнения
                if (ixj > idx && ixj < n)
                {
                    // Проверяем направление сортировки (возрастающее)
                    if ((idx & k) == 0 && data[idx] > data[ixj])
                    {
                        // Меняем элементы местами
                        int temp = data[idx];
                        data[idx] = data[ixj];
                        data[ixj] = temp;
                    }

                    // Проверяем направление сортировки (убывающее)
                    if ((idx & k) != 0 && data[idx] < data[ixj])
                    {
                        // Меняем элементы местами
                        int temp = data[idx];
                        data[idx] = data[ixj];
                        data[ixj] = temp;
                    }
                }

                // Синхронизируем потоки внутри блока
                __syncthreads();
            }
        }
    }
}

// Объявляем функцию GPU-сортировки слиянием (блочная параллельная версия)
void gpu_merge_sort(std::vector<int>& arr)
{
    // Объявляем указатель на память GPU
    int* d_data = nullptr;

    // Вычисляем размер памяти для массива
    size_t bytes = arr.size() * sizeof(int);

    // Выделяем память на GPU
    cudaMalloc(&d_data, bytes);

    // Копируем данные из памяти CPU в память GPU
    cudaMemcpy(d_data, arr.data(), bytes, cudaMemcpyHostToDevice);

    // Задаём количество потоков в блоке
    int threads = 256;

    // Вычисляем количество блоков
    int blocks = (arr.size() + threads - 1) / threads;

    // Объявляем CUDA-событие начала
    cudaEvent_t start;

    // Объявляем CUDA-событие окончания
    cudaEvent_t stop;

    // Создаём событие начала
    cudaEventCreate(&start);

    // Создаём событие окончания
    cudaEventCreate(&stop);

    // Фиксируем начало выполнения GPU-кода
    cudaEventRecord(start);

    // Запускаем CUDA-ядро сортировки
    block_sort<<<blocks, threads>>>(d_data, arr.size());

    // Фиксируем окончание выполнения GPU-кода
    cudaEventRecord(stop);

    // Ожидаем завершения всех потоков GPU
    cudaEventSynchronize(stop);

    // Объявляем переменную для хранения времени выполнения
    float ms = 0.0f;

    // Вычисляем время выполнения GPU-ядра
    cudaEventElapsedTime(&ms, start, stop);

    // Копируем отсортированные данные обратно с GPU на CPU
    cudaMemcpy(arr.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    // Выводим время выполнения GPU-сортировки
    std::cout << "GPU Merge Sort: " << ms << " ms" << std::endl;

    // Освобождаем память GPU
    cudaFree(d_data);
}
