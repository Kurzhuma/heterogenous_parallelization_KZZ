// Библиотека ввода/вывода данных
#include <iostream>

// Библиотека для динамических массивов (vector)
#include <vector>

// Библиотека для измерения времени выполнения
#include <chrono>

// Библиотека для генерации случайных чисел
#include <random>

#ifdef _OPENMP
#include <omp.h>        // Библиотека OpenMP для параллельных вычислений
#endif

using namespace std;    // Применение стандартного пространства имен

// Функция последовательного вычисления среднего значения массива

// Принимаем массив по константной ссылке
double average_sequential(const vector<int>& arr) {

    // Переменная для суммы элементов
    long long sum = 0;

    // Проходим по всем элементам массива
    for (size_t i = 0; i < arr.size(); i++) {

        // Добавляем текущий элемент к сумме
        sum += arr[i];
    }

    // Возвращаем среднее значение
    return static_cast<double>(sum) / arr.size();
}

// Функция параллельного вычисления среднего значения массива

// Принимаем массив по константной ссылке
double average_parallel(const vector<int>& arr) {

    // Общая сумма (будет использоваться reduction)
    long long sum = 0;


    // Параллельный цикл с редукцией суммы
    #pragma omp parallel for reduction(+:sum)

    // Приводим размер к int для OpenMP
    for (int i = 0; i < static_cast<int>(arr.size()); i++) {

        // Каждый поток добавляет свою часть
        sum += arr[i];
    }

    // Возвращаем среднее значение
    return static_cast<double>(sum) / arr.size();
}

// Главная функция программы
int main() {

    // Размер массива (1 млн элементов)
    const int N = 1'000'000;


    // Создаем динамический массив через vector
    vector<int> array(N);


    // Рандиизатор
    random_device rd;

    // Генератор случайных чисел
    mt19937 gen(rd());

    // Диапазон случайных чисел от 1 до 100
    uniform_int_distribution<> dist(1, 100);


    // Заполняем массив
    for (int i = 0; i < N; i++) {

        // Генерируем случайное число
        array[i] = dist(gen);
    }
    // Начало замера времени (последовательно)
    auto start_seq = chrono::high_resolution_clock::now();

    // Последовательное вычисление среднего
    double avg_seq = average_sequential(array);

    // Конец замера времени
    auto end_seq = chrono::high_resolution_clock::now();

    // Начало замера времени (параллельно)
    auto start_par = chrono::high_resolution_clock::now();

    // Параллельное вычисление среднего
    double avg_par = average_parallel(array);

    // Конец замера времени
    auto end_par = chrono::high_resolution_clock::now();


    // Время последовательного выполнения
    chrono::duration<double> time_seq = end_seq - start_seq;

    // Время параллельного выполнения
    chrono::duration<double> time_par = end_par - start_par;

    // Вывод среднего значения (последовательно)
    cout << "Sequential average: " << avg_seq
         << ", Time: " << time_seq.count() << " sec" << endl;


    // Вывод среднего значения (параллельно)
    cout << "Parallel average:   " << avg_par
         << ", Time: " << time_par.count() << " sec" << endl;


    // Завершаем программу
    return 0;
}