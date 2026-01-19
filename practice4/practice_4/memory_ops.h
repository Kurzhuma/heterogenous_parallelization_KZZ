#pragma once

// Редукция суммы ТОЛЬКО через  global memory
void gpu_reduce_global(const float* d_in, int n, float& result, float& time_ms);

// Редукция суммы через shared memory
void gpu_reduce_shared(const float* d_in, int n, float& result, float& time_ms);

// Сортировка: локальная bubble + merge через shared
void gpu_sort_with_local_and_shared(float* d_data, int n, float& time_local_ms, float& time_merge_ms);
