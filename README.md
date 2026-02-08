# Метод смешанной аппроксимации

C++ библиотека для реализации метода смешанной аппроксимации, предназначенного для представления границ областей в многомерном пространстве параметров.

## Описание

Метод смешанной аппроксимации используется для нахождения полинома `F(x)` степени `n`, который минимизирует составной функционал:

```
J = Σ_i |f(x_i) - F(x_i)|² / σ_i
    + Σ_j B_j / |y_j^* - F(y_j)|²
    + γ ∫_a^b (F''(x))² dx
```

При этом полином должен точно проходить через заданные интерполяционные узлы:
```
F(z_e) = f(z_e), e = 1, ..., m
```

### Компоненты функционала

1. **Аппроксимирующий критерий** - обеспечивает близость `F(x)` к данным в точках `x_i` с учётом их надёжности (весов `σ_i`)

2. **Отталкивающий критерий** - формирует "барьер" вокруг точек `y_j`, заставляя `F(x)` избегать этих областей

3. **Регуляризация (сглаживание)** - обеспечивает гладкость решения и устойчивость к шуму

## Структура проекта

```
mixed_boundary_approximation/
├── include/
│   └── mixed_approximation/
│       ├── types.h              # Основные структуры данных
│       ├── polynomial.h         # Класс Polynomial
│       ├── validator.h          # Валидатор входных данных
│       ├── functional.h         # Вычисление функционала и градиента
│       ├── optimizer.h          # Интерфейс оптимизатора
│       └── mixed_approximation.h # Основной класс
├── src/
│   ├── polynomial.cpp
│   ├── validator.cpp
│   ├── functional.cpp
│   ├── optimizer.cpp
│   └── mixed_approximation.cpp
├── examples/
│   └── simple_example.cpp       # Пример использования
├── tests/
│   ├── test_basic.cpp           # Базовые тесты
│   └── CMakeLists.txt
├── CMakeLists.txt
└── README.md
```

## Сборка

### Требования

- CMake 3.10 или выше
- C++17 совместимый компилятор (GCC 7+, Clang 5+, MSVC 2017+)

### Инструкция

```bash
# Создание build директории
mkdir build
cd build

# Конфигурация
cmake ..

# Сборка
cmake --build .

# Запуск тестов (если включены)
ctest

# Запуск примера
./examples/simple_example
```

## Использование

Базовый пример использования:

```cpp
#include "mixed_approximation/mixed_approximation.h"

using namespace mixed_approx;

int main() {
    // Создание конфигурации
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    config.gamma = 0.1;
    
    // Задание точек
    config.approx_points = {
        WeightedPoint(0.0, 0.0, 1.0),
        WeightedPoint(0.5, 1.0, 1.0),
        WeightedPoint(1.0, 0.0, 1.0)
    };
    
    config.repel_points = {
        WeightedPoint(0.5, 10.0, 100.0)  // избегать значения 10 в x=0.5
    };
    
    config.interp_nodes = {
        InterpolationNode(0.0, 0.0),
        InterpolationNode(1.0, 0.0)
    };
    
    // Валидация
    std::string error = Validator::validate(config);
    if (!error.empty()) {
        std::cerr << error << std::endl;
        return 1;
    }
    
    // Создание и решение
    MixedApproximation method(config);
    auto optimizer = std::make_unique<AdaptiveGradientDescentOptimizer>();
    OptimizationResult result = method.solve(std::move(optimizer));
    
    if (result.success) {
        std::cout << "Solution found!\n";
        Polynomial poly(result.coefficients);
        // Использование полинома...
    }
    
    return 0;
}
```

## Классы

### Polynomial

Класс для работы с алгебраическими полиномами.

- `evaluate(x)` - вычисление значения в точке x
- `derivative(x)` - первая производная
- `second_derivative(x)` - вторая производная
- `squared_error(x, target)` - квадрат отклонения от целевого значения
- `gradient_squared_error(x, target)` - градиент квадрата отклонения

### Functional

Вычисление функционала и его градиента.

- `evaluate(poly)` - значение функционала
- `gradient(poly)` - градиент по коэффициентам полинома
- `get_components(poly)` - получение отдельных компонент

### Validator

Проверка корректности входных данных.

- `validate(config)` - полная валидация
- `check_disjoint_sets()` - проверка непересечения множеств точек
- `check_positive_weights()` - проверка положительности весов
- `check_interpolation_nodes_count()` - проверка m ≤ n+1

### Optimizer

Базовый класс оптимизаторов.

- `GradientDescentOptimizer` - градиентный спуск с постоянным шагом
- `AdaptiveGradientDescentOptimizer` - градиентный спуск с адаптивным шагом

### MixedApproximation

Основной класс, объединяющий все компоненты.

- `solve(optimizer)` - выполнение оптимизации
- `check_interpolation_conditions()` - проверка интерполяционных условий
- `compute_repel_distances()` - расстояния до отталкивающих точек

## Параметры

### ApproximationConfig

- `polynomial_degree` - степень полинома (n)
- `interval_start`, `interval_end` - интервал определения [a, b]
- `gamma` - коэффициент регуляризации (γ ≥ 0)
- `approx_points` - аппроксимирующие точки (x_i, f(x_i), σ_i)
- `repel_points` - отталкивающие точки (y_j, y_j^*, B_j)
- `interp_nodes` - интерполяционные узлы (z_e, f(z_e))
- `epsilon` - минимальный порог для численной устойчивости (1e-8)
- `interpolation_tolerance` - допуск для интерполяционных условий (1e-10)

## Математические детали

### Параметризация с интерполяционными ограничениями

Для обеспечения интерполяционных условий `F(z_e) = f(z_e)` используется параметризация:

```
F(x) = P_int(x) + Q(x) · Π(x - z_e)
```

где:
- `P_int(x)` - интерполяционный полином Лагранжа, проходящий через все узлы `(z_e, f(z_e))`
- `Q(x)` - полином степени `n - m`
- `Π(x - z_e)` - произведение `(x - z_1)(x - z_2)...(x - z_m)`

Таким образом, интерполяционные условия выполняются автоматически при любом `Q(x)`.

### Градиент функционала

Градиент вычисляется как сумма градиентов трех компонент:

```
∇J = ∇J_approx + ∇J_repel + ∇J_reg
```

#### Аппроксимирующий компонент

```
∇_a |f(x_i) - F(x_i)|² / σ_i = -2 · (f(x_i) - F(x_i)) / σ_i · [x_i^n, x_i^{n-1}, ..., 1]
```

#### Отталкивающий компонент

```
∇_a B_j / |y_j^* - F(y_j)|² = 2 · B_j · (y_j^* - F(y_j)) / |y_j^* - F(y_j)|⁴ · [y_j^n, y_j^{n-1}, ..., 1]
```

Для численной устойчивости используется защитный порог `ε`:

```
|y_j^* - F(y_j)| → max(|y_j^* - F(y_j)|, ε)
```

#### Регуляризационный компонент

```
∇_a ∫ (F''(x))² dx = 2 ∫ F''(x) · ∂F''(x)/∂a_k dx
```

Для полинома `F(x) = Σ a_k x^k`:

```
∂F''(x)/∂a_k = k·(k-1)·x^{k-2}  (для k ≥ 2)
```

Интеграл вычисляется аналитически.

## Численная устойчивость

1. **Защита от деления на ноль** в отталкивающем члене через параметр `epsilon`
2. **Схема Горнера** для вычисления значений полинома
3. **Проверка градиента** для ранней остановки
4. **Адаптивный шаг** в оптимизаторе

## Ограничения и возможные улучшения

### Текущие ограничения

1. Поддерживается только одномерный случай (x ∈ R)
2. Оптимизаторы - простые реализации градиентного спуска
3. Нет поддержки L-BFGS-B или более продвинутых методов
4. Параметризация работает только при m ≤ n+1

### Возможные улучшения

1. Реализация L-BFGS-B (через библиотеку LBFGSpp или собственную)
2. Поддержка многомерных случаев (полиномы нескольких переменных)
3. Векторизация вычислений для производительности
4. Добавление сплайнов и RBF как альтернатив полиномам
5. Автоматическая настройка параметров (кросс-валидация)

## Лицензия

Проект создан в образовательных и исследовательских целях.

## Ссылки

1. Zykin S.V. Mixed approximation // Mathematical models for the representation of information and data processing problems. 1986.
2. Zykin S.V. Representation of data analysis results in multidimensional parameter space // Journal of Physics: Conference Series. 2020.
3. Zykin S.V., Zykin V.S. Domains identification by the parameter values in multidimensional space // Journal of Physics: Conference Series. 2021.

## Автор

Проект создан на основе работ С.В. Зыкина (Институт математики им. С.Л. Соболева СО РАН).
