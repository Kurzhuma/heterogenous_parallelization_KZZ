/**
* ЗАГОЛОВОЧНЫЙ ФАЙЛ ВСПОМОГАТЕЛЬНЫХ УТИЛИТ
 * Назначение: Обработка системных ошибок CUDA и макросы для профилирования.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * Макрос для верификации успешного завершения вызовов CUDA API.
 * В случае обнаружения ошибки выводит подробную информацию в поток stderr.
 */
#define CHECK_CUDA(call)                                                 \
{                                                                    \
const cudaError_t error = call;                                  \
if (error != cudaSuccess)                                        \
{                                                                \
fprintf(stderr, "Ошибка CUDA: %s\n", cudaGetErrorString(error)); \
fprintf(stderr, "Файл: %s, Строка: %d\n", __FILE__, __LINE__); \
exit(EXIT_FAILURE);                                          \
}                                                                \
}

/**
 * Макрос для проверки состояния ядра после его асинхронного запуска.
 */
#define CHECK_LAST_CUDA_ERROR() CHECK_CUDA(cudaGetLastError())

#endif // UTILS_H