#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>

// Константы для определения размерностей входных данных
const int N_STATS = 1000000; // Количество элементов для статистического анализа
const int N_GAUSS = 512;     // Размерность матрицы для метода Гаусса
const int N_FLOYD = 512;     // Количество вершин графа для алгоритма Флойда

// --- ЗАДАНИЕ 1: Распределенное вычисление статистических показателей ---
void run_task1(int rank, int size) {
    // Фиксация времени начала выполнения задачи на текущем процессе
    double start_time = MPI_Wtime();
    
    // Инициализация контейнеров для данных и параметров распределения
    std::vector<double> data;
    std::vector<int> sendcounts(size); // Массив количеств элементов для каждого процесса
    std::vector<int> displs(size);    // Массив смещений элементов в исходном массиве

    // Логика формирования данных на главном процессе (rank 0)
    if (rank == 0) {
        // Выделение памяти под полный объем входных данных
        data.resize(N_STATS);
        // Заполнение массива случайными значениями
        for (int i = 0; i < N_STATS; ++i) data[i] = (double)(rand() % 100);

        // Расчет параметров для MPI_Scatterv с целью корректной обработки остатка
        int sum = 0;
        for (int i = 0; i < size; ++i) {
            // Определение размера порции данных для i-го процесса
            sendcounts[i] = N_STATS / size + (i < (N_STATS % size) ? 1 : 0);
            // Определение позиции начала порции данных i-го процесса в общем массиве
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

    // Расчет количества элементов, которое получит текущий процесс
    int local_n = N_STATS / size + (rank < (N_STATS % size) ? 1 : 0);
    // Выделение локального буфера для приема распределяемых данных
    std::vector<double> local_data(local_n);

    // Распределение данных от корня ко всем процессам с учетом переменного размера порций
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_data.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Инициализация переменных для накопления локальных сумм
    double local_sum = 0, local_sq_sum = 0;
    // Проход по локальному набору данных для вычисления промежуточных сумм
    for (double val : local_data) {
        local_sum += val;             // Сумма элементов
        local_sq_sum += val * val;     // Сумма квадратов элементов
    }

    // Переменные для хранения агрегированных результатов
    double global_sum, global_sq_sum;
    // Глобальное суммирование локальных результатов на процессе 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // Глобальное суммирование локальных сумм квадратов на процессе 0
    MPI_Reduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод итоговых статистических показателей процессом-координатором
    if (rank == 0) {
        // Расчет среднего арифметического значения
        double mean = global_sum / N_STATS;
        // Расчет стандартного отклонения на основе разности средних
        double std_dev = sqrt((global_sq_sum / N_STATS) - (mean * mean));
        // Фиксация времени окончания выполнения задачи
        double end_time = MPI_Wtime();
        std::cout << "[TASK 1] Mean: " << mean << ", StdDev: " << std_dev << std::endl;
        std::cout << "Execution time: " << end_time - start_time << " seconds." << std::endl;
    }
}

// --- ЗАДАНИЕ 2: Распределенное решение системы уравнений методом Гаусса ---
void run_task2(int rank, int size) {
    // Начало замера времени выполнения вычислительного ядра
    double start_time = MPI_Wtime();
    // Расчет количества строк матрицы, обрабатываемых одним процессом
    int rows_per_proc = N_GAUSS / size;
    // Локальный буфер для хранения части строк матрицы
    std::vector<double> local_matrix(rows_per_proc * N_GAUSS);
    // Вспомогательный массив для хранения ведущей строки на каждой итерации
    std::vector<double> pivot_row(N_GAUSS);

    // Инициализация и первичное распределение данных
    if (rank == 0) {
        // Создание исходной матрицы коэффициентов
        std::vector<double> matrix(N_GAUSS * N_GAUSS, 1.0);
        // Распределение блоков строк матрицы равномерно между всеми процессами
        MPI_Scatter(matrix.data(), rows_per_proc * N_GAUSS, MPI_DOUBLE, 
                    local_matrix.data(), rows_per_proc * N_GAUSS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        // Прием строк матрицы на подчиненных процессах
        MPI_Scatter(NULL, 0, MPI_DOUBLE, local_matrix.data(), rows_per_proc * N_GAUSS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Основной цикл прямого хода метода Гаусса по количеству неизвестных
    for (int i = 0; i < N_GAUSS; ++i) {
        // Определение ранга процесса, владеющего текущей ведущей строкой
        int root = i / rows_per_proc;
        // Копирование ведущей строки в буфер процессом-владельцем
        if (rank == root) {
            pivot_row.assign(local_matrix.begin() + (i % rows_per_proc) * N_GAUSS, 
                             local_matrix.begin() + (i % rows_per_proc) * N_GAUSS + N_GAUSS);
        }
        // Рассылка текущей ведущей строки всем процессам в коммуникаторе
        MPI_Bcast(pivot_row.data(), N_GAUSS, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Итерация по локальным строкам для исключения неизвестных
        for (int j = 0; j < rows_per_proc; ++j) {
            // Определение глобального индекса текущей обрабатываемой строки
            int global_row = rank * rows_per_proc + j;
            // Выполнение операции только для строк, находящихся ниже ведущей
            if (global_row > i) {
                // Вычисление множителя для обнуления элемента в столбце i
                double factor = local_matrix[j * N_GAUSS + i] / pivot_row[i];
                // Модификация коэффициентов текущей строки
                for (int k = i; k < N_GAUSS; ++k) local_matrix[j * N_GAUSS + k] -= factor * pivot_row[k];
            }
        }
    }

    // Вывод итогового времени выполнения процессами-координаторами
    if (rank == 0) {
        double end_time = MPI_Wtime();
        std::cout << "[TASK 2] Gauss completed. Execution time: " << end_time - start_time << " seconds." << std::endl;
    }
}

// --- ЗАДАНИЕ 3: Параллельный алгоритм Флойда-Уоршелла для поиска путей ---
void run_task3(int rank, int size) {
    // Фиксация времени начала алгоритма поиска кратчайших путей
    double start_time = MPI_Wtime();
    // Определение количества строк матрицы смежности на один процесс
    int rows_per_proc = N_FLOYD / size;
    // Локальное хранилище для строк матрицы графа
    std::vector<double> local_rows(rows_per_proc * N_FLOYD);

    // Генерация графа и распределение топологии
    if (rank == 0) {
        // Инициализация матрицы весов ребер (с условными бесконечными значениями)
        std::vector<double> matrix(N_FLOYD * N_FLOYD, 10.0);
        // Обнуление главной диагонали (расстояние до самой вершины равно 0)
        for(int i=0; i<N_FLOYD; i++) matrix[i*N_FLOYD+i] = 0;
        // Распределение фрагментов графа по узлам вычислительной системы
        MPI_Scatter(matrix.data(), rows_per_proc * N_FLOYD, MPI_DOUBLE, local_rows.data(), rows_per_proc * N_FLOYD, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        // Прием фрагментов графа на рабочих процессах
        MPI_Scatter(NULL, 0, MPI_DOUBLE, local_rows.data(), rows_per_proc * N_FLOYD, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Внешний итерационный цикл по промежуточным вершинам k
    for (int k = 0; k < N_FLOYD; ++k) {
        // Временный буфер для хранения строки весов относительно вершины k
        std::vector<double> k_row(N_FLOYD);
        // Идентификация процесса, ответственного за хранение информации о вершине k
        int root = k / rows_per_proc;
        // Извлечение данных процессом-владельцем вершины k
        if (rank == root) {
            std::copy(local_rows.begin() + (k % rows_per_proc) * N_FLOYD, local_rows.begin() + (k % rows_per_proc) * N_FLOYD + N_FLOYD, k_row.begin());
        }
        // Рассылка данных о путях через вершину k всем участникам обмена
        MPI_Bcast(k_row.data(), N_FLOYD, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // Обновление локальных весов путей в соответствии с принципом динамического программирования
        for (int i = 0; i < rows_per_proc; ++i) {
            for (int j = 0; j < N_FLOYD; ++j) {
                // Проверка: является ли путь через вершину k короче текущего известного пути
                local_rows[i * N_FLOYD + j] = std::min(local_rows[i * N_FLOYD + j], local_rows[i * N_FLOYD + k] + k_row[j]);
            }
        }
    }

    // Завершение мониторинга времени и вывод статистики
    if (rank == 0) {
        double end_time = MPI_Wtime();
        std::cout << "[TASK 3] Floyd-Warshall completed. Execution time: " << end_time - start_time << " seconds." << std::endl;
    }
}

// Точка входа в приложение
int main(int argc, char** argv) {
    // Инициализация инфраструктуры MPI и разбор аргументов командной строки
    MPI_Init(&argc, &argv);
    // Определение ранга текущего процесса и общего количества запущенных процессов
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Последовательный запуск трех вычислительных задач
    run_task1(rank, size);
    run_task2(rank, size);
    run_task3(rank, size);

    // Корректное завершение работы MPI и освобождение системных ресурсов
    MPI_Finalize();
    return 0;
}