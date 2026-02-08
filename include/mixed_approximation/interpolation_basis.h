#ifndef MIXED_APPROXIMATION_INTERPOLATION_BASIS_H
#define MIXED_APPROXIMATION_INTERPOLATION_BASIS_H

#include "types.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

/**
 * @brief Перечисление методов интерполяции для построения P_int(x)
 */
enum class InterpolationMethod {
    LAGRANGE,       ///< Интерполяция по формуле Лагранжа
    BARYCENTRIC,    ///< Барицентрическая интерполяция
    NEWTON          ///< Интерполяция по формуле Ньютона
};

/**
 * @brief Базисный интерполяционный полином P_int(x)
 */
struct InterpolationBasis {
    // Исходные данные (нормализованные)
    std::vector<double> nodes;           // узлы интерполяции (отсортированные, возможно нормализованные)
    std::vector<double> values;          // значения f(z_e) в узлах
    int m_eff;                           // эффективное число узлов после объединения близких
    
    // Параметры нормализации координат
    double x_center;                     // центр интервала [a, b] для нормализации
    double x_scale;                      // масштаб интервала
    bool is_normalized;                  // флаг: были ли узлы нормализованы
    
    // Метод интерполяции
    InterpolationMethod method;          // метод интерполяции
    
    // Барицентрические веса
    std::vector<double> barycentric_weights;  // веса для барицентрического метода
    double weight_scale;                 // масштабирующий коэффициент для весов
    
    // Для метода Ньютона
    std::vector<double> divided_differences; // разделённые разности
    
    // Кэшированные данные для ускорения
    std::vector<double> weighted_values; // w_e * f(z_e) для быстрого вычисления
    
    // Флаги состояния
    bool is_valid;                       // флаг корректности построения
    std::string error_message;           // сообщение об ошибке
    
    InterpolationBasis()
        : method(InterpolationMethod::BARYCENTRIC)
        , m_eff(0)
        , x_center(0.0)
        , x_scale(1.0)
        , is_normalized(false)
        , weight_scale(1.0)
        , is_valid(false)
        , error_message("Not built") {}
    
    /**
     * @brief Построение интерполяционного полинома с расширенными возможностями
     * @param nodes узлы интерполяции (в исходных координатах)
     * @param values значения функции в узлах
     * @param method метод интерполяции
     * @param interval_start начало интервала [a, b] для нормализации
     * @param interval_end конец интервала
     * @param enable_normalization включать ли нормализацию координат
     * @param enable_node_merging включать ли объединение близких узлов
     */
    void build(const std::vector<double>& nodes,
               const std::vector<double>& values,
               InterpolationMethod method = InterpolationMethod::BARYCENTRIC,
               double interval_start = 0.0,
               double interval_end = 1.0,
               bool enable_normalization = true,
               bool enable_node_merging = true);
    
    /**
     * @brief Вычисление значения P_int(x) в точке (в исходных координатах)
     * @param x точка вычисления
     * @return значение P_int(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Вычисление первой производной P_int'(x) в точке
     * @param x точка вычисления
     * @param order порядок производной (1 или 2)
     * @return значение производной
     */
    double evaluate_derivative(double x, int order = 1) const;
    
    /**
     * @brief Верификация точности интерполяции
     * Проверяет, что P_int(z_e) = f(z_e) с заданной точностью
     * @param tolerance допуск
     * @return true, если все условия выполняются
     */
    bool verify_interpolation(double tolerance = 1e-10) const;
    
    /**
     * @brief Получение информации о построении
     * @return строка с диагностической информацией
     */
    std::string get_info() const;
    
private:
    // Вспомогательные методы для различных методов интерполяции
    double evaluate_barycentric(double x) const;
    double evaluate_newton(double x) const;
    double evaluate_lagrange(double x) const;
    
    // Нормализация координат
    void normalize_nodes(double a, double b);
    
    // Обработка близких узлов
    struct MergedNode {
        double x;           // нормализованная координата
        double value;       // усреднённое значение
        int count;          // сколько исходных узлов объединено
    };
    std::vector<MergedNode> merge_close_nodes(const std::vector<double>& nodes_norm,
                                               const std::vector<double>& values,
                                               double interval_length);
    
    // Вычисление барицентрических весов с логарифмической стабилизацией
    void compute_barycentric_weights();
    void compute_barycentric_weights_standard();
    void compute_barycentric_weights_logarithmic();
    
    // Кэширование
    void precompute_weighted_values();
    
    // Вычисление разделённых разностей (для метода Ньютона)
    void compute_divided_differences();
    
    // Вычисление производной для барицентрической формы
    double evaluate_barycentric_derivative(double x, int order) const;
    
    // Вычисление производной через разделённые разности (для метода Ньютона)
    double evaluate_newton_derivative(double x) const;
    
    // Специальные случаи
    void compute_chebyshev_weights();
    
public:
    /**
     * @brief Проверка, что узлы уникальны (с заданной точностью)
     * @param tolerance допуск
     * @return true, если все узлы различны
     */
    static bool are_nodes_unique(const std::vector<double>& nodes, double tolerance);
    
    /**
     * @brief Сортировка узлов и соответствующих значений
     * @param nodes узлы (будут отсортированы)
     * @param values значения (будут переставлены соответственно)
     */
    static void sort_nodes_and_values(std::vector<double>& nodes, std::vector<double>& values);
    
    /**
     * @brief Проверка равномерной сетки узлов
     * @param tolerance допуск для проверки постоянства шага
     * @return true, если узлы равномерно распределены
     */
    bool detect_equally_spaced_nodes(double tolerance = 1e-10) const;
    
    /**
     * @brief Проверка узлов Чебышёва
     * @param tolerance допуск для сравнения с узлами Чебышёва
     * @return true, если узлы соответствуют узлам Чебышёва
     */
    bool detect_chebyshev_nodes(double tolerance = 1e-10) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_INTERPOLATION_BASIS_H
