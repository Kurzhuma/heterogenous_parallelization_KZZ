/**
* Вспомогательные инструменты для практических работ по CUDA
 */
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * Макрос для проверки ошибок выполнения функций CUDA API.
 * В случае ошибки выводит описание, файл и строку, после чего завершает программу.
 */
#define CHECK_CUDA(call) { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
cudaGetErrorString(err), __FILE__, __LINE__); \
exit(EXIT_FAILURE); \
} \
}

#endif // UTILS_H