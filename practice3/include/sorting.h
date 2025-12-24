// Защита от повторного включения заголовка
#pragma once

// Подключаем контейнер vector
#include <vector>

// Объявление CPU Merge Sort
void cpu_merge_sort(std::vector<int>& arr);

// Объявление CPU Quick Sort
void cpu_quick_sort(std::vector<int>& arr);

// Объявление CPU Heap Sort
void cpu_heap_sort(std::vector<int>& arr);

// Объявление GPU Merge Sort
void gpu_merge_sort(std::vector<int>& arr);
