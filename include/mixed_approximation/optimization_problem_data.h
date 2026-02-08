#ifndef MIXED_APPROXIMATION_OPTIMIZATION_PROBLEM_DATA_H
#define MIXED_APPROXIMATION_OPTIMIZATION_PROBLEM_DATA_H

#include "types.h"
#include <vector>
#include <string>
#include <array>

namespace mixed_approx {

/**
 * @brief Структура для хранения данных задачи оптимизации
 * 
 * Инкапсулирует все входные данные для функционала:
 * - Аппроксимирующие точки {x_i, f(x_i), σ_i}
 * - Отталкивающие точки {y_j, y_j^*, B_j}
 * - Интерполяционные узлы {z_e, f(z_e)} (уже учтены в параметризации)
 * - Параметр регуляризации γ
 * - Границы интервала [a, b]
 */
struct OptimizationProblemData {
    // Аппроксимирующие точки
    std::vector<double> approx_x;      // координаты x_i
    std::vector<double> approx_f;      // целевые значения f(x_i)
    std::vector<double> approx_weight;  // веса 1/σ_i
    
    // Отталкивающие точки
    std::vector<double> repel_y;       // координаты y_j
    std::vector<double> repel_forbidden; // запрещённые значения y_j^*
    std::vector<double> repel_weight;   // веса барьеров B_j
    
    // Интерполяционные узлы (уже в параметризации)
    std::vector<double> interp_z;       // координаты z_e
    std::vector<double> interp_f;       // значения f(z_e)
    
    // Параметры
    double gamma;                       // коэффициент регуляризации
    double interval_a;                  // левая граница
    double interval_b;                  // правая граница
    double epsilon;                     // минимальный порог для защиты от деления на ноль
    
    // Конструктор по умолчанию
    OptimizationProblemData()
        : gamma(0.0), interval_a(0.0), interval_b(1.0), epsilon(1e-8) {}
    
    // Конструктор из ApproximationConfig
    explicit OptimizationProblemData(const ApproximationConfig& config);
    
    // Проверка валидности данных
    bool is_valid() const;
    
    // Размерность задачи
    size_t num_approx_points() const { return approx_x.size(); }
    size_t num_repel_points() const { return repel_y.size(); }
    size_t num_interp_nodes() const { return interp_z.size(); }
};

/**
 * @brief Структура для кэширования предварительных вычислений
 * 
 * Фаза 1: Кэширование значений компонентов в точках данных
 * Фаза 2: Кэширование базисных функций
 */
struct OptimizationCache {
    // ============== Фаза 1: Кэширование значений компонентов ==============
    
    // Для аппроксимирующих точек {x_i}
    std::vector<double> P_at_x;   // P_int(x_i)
    std::vector<double> W_at_x;    // W(x_i)
    std::vector<std::vector<double>> phi_at_x;  // φ_k(x_i) для каждого базиса [точка][базис]
    
    // Для отталкивающих точек {y_j}
    std::vector<double> P_at_y;    // P_int(y_j)
    std::vector<double> W_at_y;    // W(y_j)
    std::vector<std::vector<double>> phi_at_y;  // φ_k(y_j) для каждого базиса [точка][базис]
    
    // ============== Фаза 2: Кэширование для регуляризации ==============
    
    // Узлы квадратуры
    std::vector<double> quad_points;     // узлы в координатах [a, b]
    std::vector<double> quad_weights;    // веса квадратуры
    
    // Значения в узлах квадратуры
    std::vector<double> W_at_quad;        // W(x_k)
    std::vector<double> W1_at_quad;      // W'(x_k)
    std::vector<double> W2_at_quad;      // W''(x_k)
    std::vector<double> P2_at_quad;      // P_int''(x_k)
    
    // Матрица жёсткости K[k][l] = ∫ φ_k''(x) φ_l''(x) dx (только верхний треугольник)
    std::vector<double> stiffness_matrix; // хранится как row-major: K[k*dim + l] для k <= l
    int stiffness_dim;                    // размерность матрицы = n_free
    
    // ============== Фаза 3: Кэширование базисных функций ==============
    
    // Значения базисных функций и их производных до второго порядка
    // [точка][базис][производная] где производная: 0 - значение, 1 - первая, 2 - вторая
    std::vector<std::vector<std::array<double, 3>>> basis_cache;
    
    // Флаги состояния
    bool data_cache_valid;    // кэш точек данных
    bool quad_cache_valid;    // кэш квадратуры
    bool basis_cache_valid;   // кэш базисных функций
    bool stiffness_valid;     // матрица жёсткости
    
    // Конструктор
    OptimizationCache()
        : stiffness_dim(0)
        , data_cache_valid(false)
        , quad_cache_valid(false)
        , basis_cache_valid(false)
        , stiffness_valid(false) {}
    
    // Очистка кэша
    void clear();
    
    // Проверка готовности
    bool is_ready(int n_free) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OPTIMIZATION_PROBLEM_DATA_H
