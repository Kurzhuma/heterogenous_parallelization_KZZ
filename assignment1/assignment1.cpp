//   Kurmash Zhumagozhayev, ADA-2401M

//   Assignment 1
// __________________________________________________________________________________________________________________________

// Подключаем библиотеку для стандартного ввода и вывода
#include <iostream>

// Подключаем библиотеку для работы с rand() и srand()
#include <cstdlib>

// Подключаем библиотеку для работы со временем
#include <ctime>

// Подключаем библиотеку для измерения времени выполнения
#include <chrono>

// Проверяем, поддерживается ли OpenMP
#ifdef _OPENMP

// Подключаем библиотеку OpenMP
#include <omp.h>

// Завершаем директиву условной компиляции
#endif

// Используем стандартное пространство имён
using namespace std;

// Точка входа в программу
int main() {

    // =========================
    // ЗАДАНИЕ 1
    // =========================

    // Объявляем размер массива для первого задания
    const int N1 = 50000;

    // Динамически выделяем память под массив целых чисел
    int* array1 = new int[N1];

    // Инициализируем генератор случайных чисел текущим временем
    srand(time(nullptr));

    // Заполняем массив случайными числами
    for (int i = 0; i < N1; i++)
        // Генерируем число от 1 до 100
        array1[i] = rand() % 100 + 1;

    // Переменная для хранения суммы элементов массива
    long long sum1 = 0;

    // Суммируем элементы массива
    for (int i = 0; i < N1; i++)
        // Добавляем текущий элемент к сумме
        sum1 += array1[i];

    // Вычисляем среднее значение элементов массива
    double avg1 = static_cast<double>(sum1) / N1;

    // Выводим среднее значение на экран
    cout << "Task 1 - Average value: " << avg1 << endl;

    // Освобождаем динамически выделенную память
    delete[] array1;

    // =========================
    // ЗАДАНИЕ 2
    // =========================

    // Объявляем размер массива для второго задания
    const int N2 = 1000000;

    // Динамически выделяем память под массив
    int* array2 = new int[N2];

    // Заполняем массив случайными числами
    for (int i = 0; i < N2; i++)
        // Присваиваем случайное значение
        array2[i] = rand();

    // Фиксируем начальное время последовательного алгоритма
    auto start_seq = chrono::high_resolution_clock::now();

    // Инициализируем минимальное значение
    int min_seq = array2[0];

    // Инициализируем максимальное значение
    int max_seq = array2[0];

    // Последовательный поиск минимума и максимума
    for (int i = 1; i < N2; i++) {
        // Проверяем, меньше ли текущий элемент минимума
        if (array2[i] < min_seq)
            // Обновляем минимум
            min_seq = array2[i];

        // Проверяем, больше ли текущий элемент максимума
        if (array2[i] > max_seq)
            // Обновляем максимум
            max_seq = array2[i];
    }

    // Фиксируем конечное время последовательного алгоритма
    auto end_seq = chrono::high_resolution_clock::now();

    // Вычисляем время выполнения последовательного алгоритма
    chrono::duration<double> time_seq = end_seq - start_seq;

    // Выводим минимальное значение
    cout << "Task 2 - Sequential Min: " << min_seq << endl;

    // Выводим максимальное значение
    cout << "Task 2 - Sequential Max: " << max_seq << endl;

    // Выводим время выполнения
    cout << "Task 2 - Sequential Time: " << time_seq.count() << " s" << endl;

    // =========================
    // ЗАДАНИЕ 3
    // =========================

    // Инициализируем минимум для параллельного алгоритма
    int min_par = array2[0];

    // Инициализируем максимум для параллельного алгоритма
    int max_par = array2[0];

    // Фиксируем начальное время параллельного алгоритма
    auto start_par = chrono::high_resolution_clock::now();

    // Параллельный цикл поиска минимума и максимума с редукцией
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N2; i++) {
        // Проверяем минимум
        if (array2[i] < min_par)
            // Обновляем минимум
            min_par = array2[i];

        // Проверяем максимум
        if (array2[i] > max_par)
            // Обновляем максимум
            max_par = array2[i];
    }

    // Фиксируем конечное время параллельного алгоритма
    auto end_par = chrono::high_resolution_clock::now();

    // Вычисляем время выполнения параллельного алгоритма
    chrono::duration<double> time_par = end_par - start_par;

    // Выводим минимум
    cout << "Task 3 - Parallel Min: " << min_par << endl;

    // Выводим максимум
    cout << "Task 3 - Parallel Max: " << max_par << endl;

    // Выводим время выполнения
    cout << "Task 3 - Parallel Time: " << time_par.count() << " s" << endl;

    // Освобождаем память массива из заданий 2 и 3
    delete[] array2;

    // =========================
    // ЗАДАНИЕ 4
    // =========================

    // Объявляем размер массива для четвертого задания
    const int N4 = 5000000;

    // Динамически выделяем память под массив
    int* array4 = new int[N4];

    // Заполняем массив случайными числами
    for (int i = 0; i < N4; i++)
        // Генерируем число
        array4[i] = rand() % 100;

    // Переменная для суммы в последовательном варианте
    long long sum_seq_avg = 0;

    // Фиксируем начальное время последовательного вычисления
    auto start_avg_seq = chrono::high_resolution_clock::now();

    // Последовательное суммирование
    for (int i = 0; i < N4; i++)
        // Добавляем элемент к сумме
        sum_seq_avg += array4[i];

    // Фиксируем конечное время
    auto end_avg_seq = chrono::high_resolution_clock::now();

    // Вычисляем среднее значение
    double avg_seq = static_cast<double>(sum_seq_avg) / N4;

    // Вычисляем время выполнения
    chrono::duration<double> time_avg_seq = end_avg_seq - start_avg_seq;

    // Переменная для суммы в параллельном варианте
    long long sum_par_avg = 0;

    // Фиксируем начальное время параллельного вычисления
    auto start_avg_par = chrono::high_resolution_clock::now();

    // Параллельное суммирование с редукцией
    #pragma omp parallel for reduction(+:sum_par_avg)
    for (int i = 0; i < N4; i++)
        // Добавляем элемент к общей сумме
        sum_par_avg += array4[i];

    // Фиксируем конечное время
    auto end_avg_par = chrono::high_resolution_clock::now();

    // Вычисляем среднее значение
    double avg_par = static_cast<double>(sum_par_avg) / N4;

    // Вычисляем время выполнения
    chrono::duration<double> time_avg_par = end_avg_par - start_avg_par;

    // Выводим последовательное среднее значение
    cout << "Task 4 - Sequential Average: " << avg_seq << endl;

    // Выводим время последовательного алгоритма
    cout << "Task 4 - Sequential Time: " << time_avg_seq.count() << " s" << endl;

    // Выводим параллельное среднее значение
    cout << "Task 4 - Parallel Average: " << avg_par << endl;

    // Выводим время параллельного алгоритма
    cout << "Task 4 - Parallel Time: " << time_avg_par.count() << " s" << endl;

    // Освобождаем память массива
    delete[] array4;

    // Завершаем выполнение программы
    return 0;
}
