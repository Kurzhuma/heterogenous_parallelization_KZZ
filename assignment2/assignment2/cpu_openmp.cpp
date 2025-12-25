// Assignment 2

/* Подключаем библиотеку для вывода результатов измерений и диагностических сообщений */
#include <iostream>

/* Используем контейнер vector для хранения динамических массивов целых чисел */
#include <vector>

/* Подключаем генератор случайных чисел для формирования тестовых массивов */
#include <random>

/* Используем chrono для точного измерения времени выполнения алгоритмов */
#include <chrono>

/* Подключаем алгоритмы STL, включая swap */
#include <algorithm>

/* Проверяем, поддерживает ли компилятор OpenMP */
#ifdef _OPENMP
/* Подключаем заголовок OpenMP для работы с потоками */
#include <omp.h>
#endif

/* Используем стандартное пространство имён для упрощения кода */
using namespace std;

/* Функция создаёт массив из n случайных целых чисел */
vector<int> generateArray(int n)
{
    /* Создаём источник энтропии */
    random_device rd;

    /* Инициализируем генератор Mersenne Twister */
    mt19937 gen(rd());

    /* Определяем диапазон случайных значений */
    uniform_int_distribution<> dist(0, 100000);

    /* Выделяем память под массив */
    vector<int> a(n);

    /* Заполняем массив случайными числами */
    for (int i = 0; i < n; ++i)
    {
        /* Присваиваем очередному элементу случайное значение */
        a[i] = dist(gen);
    }

    /* Возвращаем сформированный массив */
    return a;
}

/* Последовательный алгоритм поиска минимума и максимума */
void minmax_seq(const vector<int>& a, int& mn, int& mx)
{
    /* Инициализируем минимум первым элементом массива */
    mn = a[0];

    /* Инициализируем максимум первым элементом массива */
    mx = a[0];

    /* Последовательно просматриваем элементы массива */
    for (int i = 1; i < a.size(); ++i)
    {
        /* Проверяем условие обновления минимума */
        if (a[i] < mn)
        {
            /* Обновляем минимальное значение */
            mn = a[i];
        }

        /* Проверяем условие обновления максимума */
        if (a[i] > mx)
        {
            /* Обновляем максимальное значение */
            mx = a[i];
        }
    }
}

/* Параллельный алгоритм поиска минимума и максимума с OpenMP */
void minmax_omp(const vector<int>& a, int& mn, int& mx)
{
    /* Инициализируем минимум */
    mn = a[0];

    /* Инициализируем максимум */
    mx = a[0];

    /* Параллельный цикл с редукцией для корректного вычисления min и max */
    #pragma omp parallel for reduction(min:mn) reduction(max:mx)
    for (int i = 0; i < a.size(); ++i)
    {
        /* Проверяем локальный минимум */
        if (a[i] < mn)
        {
            /* Обновляем минимум */
            mn = a[i];
        }

        /* Проверяем локальный максимум */
        if (a[i] > mx)
        {
            /* Обновляем максимум */
            mx = a[i];
        }
    }
}

/* Последовательная реализация сортировки выбором */
void selection_sort_seq(vector<int>& a)
{
    /* Получаем размер массива */
    int n = a.size();

    /* Внешний цикл сортировки */
    for (int i = 0; i < n - 1; ++i)
    {
        /* Считаем текущий индекс минимальным */
        int min_i = i;

        /* Поиск минимального элемента в неотсортированной части */
        for (int j = i + 1; j < n; ++j)
        {
            /* Сравниваем элементы массива */
            if (a[j] < a[min_i])
            {
                /* Обновляем индекс минимума */
                min_i = j;
            }
        }

        /* Переставляем элементы местами */
        swap(a[i], a[min_i]);
    }
}

/* Параллельная версия сортировки выбором с OpenMP */
void selection_sort_omp(vector<int>& a)
{
    /* Получаем размер массива */
    int n = a.size();

    /* Внешний цикл остаётся последовательным */
    for (int i = 0; i < n - 1; ++i)
    {
        /* Индекс минимального элемента */
        int min_i = i;

        /* Параллельный поиск минимума */
        #pragma omp parallel for
        for (int j = i + 1; j < n; ++j)
        {
            /* Критическая секция для предотвращения гонки данных */
            #pragma omp critical
            {
                /* Сравниваем элементы */
                if (a[j] < a[min_i])
                {
                    /* Обновляем индекс минимума */
                    min_i = j;
                }
            }
        }

        /* Переставляем элементы */
        swap(a[i], a[min_i]);
    }
}

/* Главная функция программы */
int main()
{
    /* Выводим заголовок CPU-части */
    cout << "Assignment 2 | CPU OpenMP\n";

#ifdef _OPENMP
    /* Выводим количество доступных потоков OpenMP */
    cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    /* Сообщаем об отсутствии поддержки OpenMP */
    cout << "OpenMP disabled\n";
#endif

    /* Формируем массив из 10000 элементов */
    vector<int> a = generateArray(10000);

    /* Объявляем переменные минимума и максимума */
    int mn, mx;

    /* Запоминаем время начала последовательного алгоритма */
    auto t1 = chrono::high_resolution_clock::now();

    /* Выполняем последовательный min/max */
    minmax_seq(a, mn, mx);

    /* Запоминаем время окончания */
    auto t2 = chrono::high_resolution_clock::now();

    /* Выводим время выполнения последовательной версии */
    cout << "Min/Max sequential time: "
         << chrono::duration<double, milli>(t2 - t1).count()
         << " ms\n";

    /* Запоминаем время начала параллельного алгоритма */
    t1 = chrono::high_resolution_clock::now();

    /* Выполняем параллельный min/max */
    minmax_omp(a, mn, mx);

    /* Запоминаем время окончания */
    t2 = chrono::high_resolution_clock::now();

    /* Выводим время выполнения OpenMP-версии */
    cout << "Min/Max OpenMP time:     "
         << chrono::duration<double, milli>(t2 - t1).count()
         << " ms\n";

    /* Проверяем сортировку для разных размеров массивов */
    for (int n : {1000, 10000})
    {
        /* Генерируем массив */
        vector<int> b = generateArray(n);

        /* Создаём копию массива */
        vector<int> c = b;

        /* Засекаем время последовательной сортировки */
        t1 = chrono::high_resolution_clock::now();

        /* Выполняем последовательную сортировку */
        selection_sort_seq(b);

        /* Фиксируем время окончания */
        t2 = chrono::high_resolution_clock::now();

        /* Выводим время последовательной сортировки */
        cout << "Selection sort sequential (" << n << "): "
             << chrono::duration<double, milli>(t2 - t1).count()
             << " ms\n";

        /* Засекаем время параллельной сортировки */
        t1 = chrono::high_resolution_clock::now();

        /* Выполняем параллельную сортировку */
        selection_sort_omp(c);

        /* Фиксируем время окончания */
        t2 = chrono::high_resolution_clock::now();

        /* Выводим время OpenMP-сортировки */
        cout << "Selection sort OpenMP (" << n << "): "
             << chrono::duration<double, milli>(t2 - t1).count()
             << " ms\n";
    }

    /* Завершаем программу */
    return 0;
}

