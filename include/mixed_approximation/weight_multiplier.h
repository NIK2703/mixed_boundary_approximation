#ifndef MIXED_APPROXIMATION_WEIGHT_MULTIPLIER_H
#define MIXED_APPROXIMATION_WEIGHT_MULTIPLIER_H

#include "types.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

/**
 * @brief Весовой множитель W(x) = ∏_{e=1..m} (x - z_e)
 */
struct WeightMultiplier {
    std::vector<double> roots;          // отсортированные узлы z_e (в исходных координатах)
    std::vector<double> roots_norm;     // узлы в нормализованных координатах [-1, 1] (если нормализация включена)
    std::vector<double> coeffs;         // коэффициенты полинома W(x) в порядке убывания степеней [w_m=1, w_{m-1}, ..., w_0]
    double min_root_distance;           // минимальное расстояние между корнями
    bool use_direct_evaluation;         // использовать прямое вычисление через корни
    
    // Параметры нормализации координат
    double shift;                       // сдвиг для нормализации: x_norm = (x - shift) / scale
    double scale;                       // масштаб для нормализации
    bool is_normalized;                 // флаг: были ли корни нормализованы
    
    // Кэшированные значения (опционально, для ускорения)
    std::vector<double> cache_x_vals;   // закэшированные значения W(x) в точках {x_i}
    std::vector<double> cache_y_vals;   // закэшированные значения W(y) в точках {y_j}
    std::vector<double> cache_x_deriv1; // закэшированные значения W'(x) в точках {x_i}
    std::vector<double> cache_y_deriv1; // закэшированные значения W'(y) в точках {y_j}
    std::vector<double> cache_x_deriv2; // закэшированные значения W''(x) в точках {x_i}
    std::vector<double> cache_y_deriv2; // закэшированные значения W''(y) в точках {y_j}
    bool caches_ready;                  // флаг: готовы ли кэши
    
    WeightMultiplier()
        : min_root_distance(0.0)
        , use_direct_evaluation(true)
        , shift(0.0)
        , scale(1.0)
        , is_normalized(false)
        , caches_ready(false) {}
    
    /**
     * @brief Вычисление значения W(x) в точке
     * @param x точка вычисления (в исходных координатах)
     * @return значение W(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Вычисление первой/второй производной W'(x) или W''(x) в точке
     * @param x точка вычисления (в исходных координатах)
     * @param order порядок производной (1 или 2)
     * @return значение производной
     */
    double evaluate_derivative(double x, int order = 1) const;
    
    /**
     * @brief Построение полинома W(x) по заданным корням
     * @param roots узлы интерполяции (в исходных координатах, не обязательно отсортированные)
     * @param interval_start начало интервала [a, b] для оценки необходимости нормализации
     * @param interval_end конец интервала
     * @param enable_normalization включать ли нормализацию координат
     */
    void build_from_roots(const std::vector<double>& roots,
                         double interval_start = 0.0,
                         double interval_end = 1.0,
                         bool enable_normalization = true);
    
    /**
     * @brief Получение степени полинома
     * @return степень (равна числу корней)
     */
    int degree() const { return static_cast<int>(roots.size()); }
    
    /**
     * @brief Верификация корректности построения
     * Проверяет: W(z_e) ≈ 0, моничность, согласованность представлений
     * @param tolerance допуск для нулевых значений в корнях
     * @return true, если все проверки пройдены
     */
    bool verify_construction(double tolerance = 1e-10) const;
    
    /**
     * @brief Построение кэшей значений в заданных точках
     * @param points_x точки для кэширования W(x), W'(x), W''(x)
     * @param points_y точки для кэширования W(y), W'(y), W''(y)
     */
    void build_caches(const std::vector<double>& points_x,
                      const std::vector<double>& points_y);
    
    /**
     * @brief Очистка кэшей
     */
    void clear_caches();
    
    /**
     * @brief Умножение полинома W(x) на полином Q(x)
     * Выполняет свёртку коэффициентов
     * @param q_coeffs коэффициенты Q(x) в порядке убывания степеней [q_{deg_Q}, ..., q_0]
     * @return коэффициенты произведения Q(x)·W(x) в порядке убывания степеней
     */
    std::vector<double> multiply_by_Q(const std::vector<double>& q_coeffs) const;
    
    /**
     * @brief Вычисление значения Q(x)·W(x) без явного построения коэффициентов
     * @param x точка вычисления (в исходных координатах)
     * @param q_coeffs коэффициенты Q(x) в порядке убывания степеней
     * @return значение Q(x)·W(x)
     */
    double evaluate_product(double x, const std::vector<double>& q_coeffs) const;
    
    /**
     * @brief Получение коэффициентов в порядке возрастания степеней
     * @return коэффициенты [w_0, w_1, ..., w_m] для свёртки и других операций
     */
    std::vector<double> get_coeffs_ascending() const {
        if (coeffs.empty()) return {};
        // coeffs хранится в нисходящем порядке: [w_m, w_{m-1}, ..., w_0]
        // Нам нужен восходящий порядок: [w_0, w_1, ..., w_m]
        std::vector<double> result(coeffs.rbegin(), coeffs.rend());
        return result;
    }
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_WEIGHT_MULTIPLIER_H
