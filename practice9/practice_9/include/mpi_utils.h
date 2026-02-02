/**
 * ВСПОМОГАТЕЛЬНЫЕ ИНСТРУМЕНТЫ ДЛЯ РАБОТЫ С MPI
 * Задачи: Генерация данных, инициализация матриц и визуализация результатов.
 */

#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

/**
 * Инициализация вектора случайными числами с плавающей запятой
 */
// Функция заполнения контейнера данными для статистического анализа
inline void generateRandomData(std::vector<double>& data, size_t n) {
    // Установка базового значения для генератора случайных чисел
    srand(static_cast<unsigned int>(time(NULL)));
    // Последовательное заполнение массива значениями в диапазоне [0, 100)
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<double>(rand() % 100);
    }
}

/**
 * Генерация квадратной матрицы для метода Гаусса или алгоритма Флойда
 */
// Функция формирования тестовой матрицы смежности или коэффициентов
inline void generateMatrix(std::vector<double>& matrix, int n, bool isGraph = false) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                // Расстояние до самой себя или диагональное преобладание
                matrix[i * n + j] = 0.0;
            } else {
                // Инициализация весов ребер или коэффициентов системы уравнений
                matrix[i * n + j] = static_cast<double>(rand() % 20 + 1);
            }
        }
    }
}

/**
 * Вывод матрицы в консоль (используется только процессом 0 для малых N)
 */
// Функция форматированного отображения результирующих данных
inline void printMatrix(const std::vector<double>& matrix, int n) {
    // Ограничение вывода для предотвращения переполнения консоли при больших N
    int limit = (n > 10) ? 10 : n;
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j) {
            // Форматированный вывод элемента с фиксированной шириной
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * n + j];
        }
        std::cout << std::endl;
    }
    if (n > 10) std::cout << "... (вывод ограничен первыми 10 строками)" << std::endl;
}

#endif // MPI_UTILS_H