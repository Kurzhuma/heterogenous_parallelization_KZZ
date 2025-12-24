// Подключаем стандартный ввод-вывод
#include <iostream>

// Подключаем контейнер vector
#include <vector>

// Подключаем генератор случайных чисел
#include <random>

// Подключаем библиотеку измерения времени
#include <chrono>

// Подключаем алгоритмы стандартной библиотеки
#include <algorithm>

// Подключаем заголовок с сортировками
#include "sorting.h"

// Функция генерации массива
std::vector<int> generate_array(size_t size)
{
    // Создаём вектор заданного размера
    std::vector<int> arr(size);

    // Создаём генератор случайных чисел
    std::mt19937 gen(42);

    // Задаём диапазон значений
    std::uniform_int_distribution<> dist(0, 1000000);

    // Заполняем массив случайными значениями
    for (size_t i = 0; i < size; ++i)
    {
        arr[i] = dist(gen);
    }

    // Возвращаем массив
    return arr;
}

// Главная функция программы
int main()
{
    // Задаём размеры массивов
    std::vector<size_t> sizes = {10000, 100000, 1000000};

    // Перебираем размеры массивов
    for (size_t size : sizes)
    {
        // Генерируем исходный массив
        std::vector<int> base = generate_array(size);

        // Создаём копию для Merge Sort
        std::vector<int> arr_merge = base;

        // Создаём копию для Quick Sort
        std::vector<int> arr_quick = base;

        // Создаём копию для Heap Sort
        std::vector<int> arr_heap = base;

        // Выводим размер массива
        std::cout << "\nArray size: " << size << std::endl;

        // Фиксируем старт времени Merge Sort
        auto m_start = std::chrono::high_resolution_clock::now();

        // Запускаем Merge Sort
        cpu_merge_sort(arr_merge);

        // Фиксируем конец времени Merge Sort
        auto m_end = std::chrono::high_resolution_clock::now();

        // Выводим время Merge Sort
        std::cout << "CPU Merge Sort: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count()
                  << " ms" << std::endl;

        // Фиксируем старт времени Quick Sort
        auto q_start = std::chrono::high_resolution_clock::now();

        // Запускаем Quick Sort
        cpu_quick_sort(arr_quick);

        // Фиксируем конец времени Quick Sort
        auto q_end = std::chrono::high_resolution_clock::now();

        // Выводим время Quick Sort
        std::cout << "CPU Quick Sort: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(q_end - q_start).count()
                  << " ms" << std::endl;

        // Фиксируем старт времени Heap Sort
        auto h_start = std::chrono::high_resolution_clock::now();

        // Запускаем Heap Sort
        cpu_heap_sort(arr_heap);

        // Фиксируем конец времени Heap Sort
        auto h_end = std::chrono::high_resolution_clock::now();

        // Выводим время Heap Sort
        std::cout << "CPU Heap Sort: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(h_end - h_start).count()
                  << " ms" << std::endl;

        // Запускаем GPU Merge Sort
        gpu_merge_sort(base);


    }

    // Завершаем программу
    return 0;
}

