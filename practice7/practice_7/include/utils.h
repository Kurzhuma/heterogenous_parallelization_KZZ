/**
* Утилиты для практической работы №7
 */
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <cuda_runtime.h>

// Макрос проверки ошибок CUDA: если что-то пойдет не так, он укажет строку
#define CHECK_CUDA(call) { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
exit(EXIT_FAILURE); \
} \
}

#endif