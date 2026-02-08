#ifndef MIXED_APPROXIMATION_CORRECTION_POLYNOMIAL_H
#define MIXED_APPROXIMATION_CORRECTION_POLYNOMIAL_H

#include "types.h"
#include "weight_multiplier.h"
#include "interpolation_basis.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

/**
 * @brief Тип базиса для корректирующего полинома Q(x)
 */
enum class BasisType {
    MONOMIAL,       ///< Мономиальный базис: 1, x, x^2, ...
    CHEBYSHEV       ///< Ортогональный базис Чебышёва: T_0(t), T_1(t), ...
};

/**
 * @brief Метод инициализации коэффициентов корректирующего полинома
 */
enum class InitializationMethod {
    ZERO,           ///< Нулевая инициализация: все q_k = 0
    LEAST_SQUARES,  ///< Инициализация через взвешенный МНК по аппроксимирующим точкам
    RANDOM          ///< Случайная инициализация в небольшом диапазоне
};

/**
 * @brief Корректирующий полином Q(x) с параметризацией через свободные коэффициенты
 *
 * Представление: F(x) = P_int(x) + Q(x)·W(x), где Q(x) = Σ_{k=0..deg_Q} q_k · φ_k(x)
 * φ_k(x) - базисные функции (мономы или полиномы Чебышёва)
 */
struct CorrectionPolynomial {
    // Параметры базиса
    BasisType basis_type;                // MONOMIAL или CHEBYSHEV
    int degree;                          // deg_Q = n - m
    int n_free;                          // n_free = deg_Q + 1
    
    // Коэффициенты (вектор свободных параметров для оптимизатора)
    std::vector<double> coeffs;          // q_0, q_1, ..., q_{deg_Q} (в порядке возрастания степеней)
    
    // Параметры нормализации (для базиса Чебышёва)
    double x_center;                     // центр интервала [a, b]
    double x_scale;                      // масштаб интервала (b - a) / 2
    
    // Кэшированные данные для ускорения вычислений
    std::vector<std::vector<double>> basis_cache_x;   // basis_cache_x[i][k] = φ_k(x_i)
    std::vector<std::vector<double>> basis_cache_y;   // basis_cache_y[j][k] = φ_k(y_j)
    std::vector<std::vector<double>> basis2_cache_x;  // basis2_cache_x[i][k] = φ_k''(x_i)
    std::vector<std::vector<double>> basis2_cache_y;  // basis2_cache_y[j][k] = φ_k''(y_j)
    
    // Матрица жёсткости для регуляризации: K[k][l] = ∫ φ_k''(x) φ_l''(x) W(x)^2 dx
    std::vector<std::vector<double>> stiffness_matrix;
    bool stiffness_matrix_computed;
    
    // Параметр регуляризации
    double regularization_lambda;
    
    // Метаданные состояния
    bool is_initialized;
    InitializationMethod init_method;
    std::string validation_message;
    
    CorrectionPolynomial()
        : basis_type(BasisType::MONOMIAL)
        , degree(0)
        , n_free(0)
        , x_center(0.0)
        , x_scale(1.0)
        , stiffness_matrix_computed(false)
        , regularization_lambda(0.0)
        , is_initialized(false)
        , init_method(InitializationMethod::ZERO)
        , validation_message("Not initialized") {}
    
    /**
     * @brief Инициализация структуры
     * @param deg степень корректирующего полинома
     * @param basis выбранный тип базиса
     * @param interval_center центр интервала [a, b]
     * @param interval_scale масштаб интервала (b-a)/2
     */
    void initialize(int deg, BasisType basis, double interval_center = 0.0, double interval_scale = 1.0);
    
    /**
     * @brief Выбор базиса на основе степени
     * @param deg степень полинома
     * @return выбранный тип базиса
     */
    static BasisType choose_basis_type(int deg);
    
    /**
     * @brief Инициализация коэффициентов выбранным методом
     */
    void initialize_coefficients(InitializationMethod method,
                                 const std::vector<WeightedPoint>& approx_points,
                                 const std::vector<RepulsionPoint>& repel_points,
                                 const InterpolationBasis& p_int,
                                 const WeightMultiplier& W,
                                 double interval_start,
                                 double interval_end);
    
    /**
     * @brief Нулевая инициализация: q_k = 0
     */
    void initialize_zero();
    
    /**
     * @brief Инициализация через взвешенный МНК
     */
    void initialize_least_squares(const std::vector<WeightedPoint>& approx_points,
                                   const InterpolationBasis& p_int,
                                   const WeightMultiplier& W);
    
    /**
     * @brief Случайная инициализация в малом диапазоне [-0.01, 0.01]
     */
    void initialize_random();
    
    /**
     * @brief Гибридная стратегия с защитой от барьеров
     */
    void apply_barrier_protection(const std::vector<RepulsionPoint>& repel_points,
                                   double safe_distance_factor = 0.1);
    
    /**
     * @brief Вычисление значения Q(x) в точке
     * @param x точка вычисления (в исходных координатах)
     * @return значение Q(x)
     */
    double evaluate_Q(double x) const;
    
    /**
     * @brief Вычисление производной Q'(x) или Q''(x) в точке
     * @param x точка вычисления
     * @param order порядок производной (1 или 2)
     * @return значение производной
     */
    double evaluate_Q_derivative(double x, int order) const;
    
    /**
     * @brief Построение кэшей значений базисных функций
     */
    void build_caches(const std::vector<double>& points_x,
                      const std::vector<double>& points_y);
    
    /**
     * @brief Очистка кэшей
     */
    void clear_caches();
    
    /**
     * @brief Вычисление матрицы жёсткости K для регуляризационного члена
     */
    void compute_stiffness_matrix(double a, double b, const WeightMultiplier& W, int gauss_points = 20);
    
    /**
     * @brief Верификация корректности параметризации
     */
    bool verify_initialization(const std::vector<WeightedPoint>& approx_points,
                              const std::vector<RepulsionPoint>& repel_points,
                              const InterpolationBasis& p_int,
                              const WeightMultiplier& W);
    
    /**
     * @brief Установка параметра регуляризации λ
     */
    void set_regularization_lambda(double lambda) { regularization_lambda = lambda; }
    
    /**
     * @brief Вычисление значения функционала качества с регуляризацией
     */
    double compute_objective(const std::vector<WeightedPoint>& approx_points,
                            const InterpolationBasis& p_int,
                            const WeightMultiplier& W) const;
    
    /**
     * @brief Вычисление градиента функционала по коэффициентам q_k
     */
    std::vector<double> compute_gradient(const std::vector<WeightedPoint>& approx_points,
                                        const InterpolationBasis& p_int,
                                        const WeightMultiplier& W) const;
    
    /**
     * @brief Получение диагностической информации
     */
    std::string get_diagnostic_info() const;
    
    // ============== Методы для FunctionalEvaluator (с внешними коэффициентами) ==============
    
    /**
     * @brief Вычисление Q(x) с переданными коэффициентами
     * @param x точка вычисления
     * @param q внешние коэффициенты
     * @return значение Q(x)
     */
    double evaluate_Q_with_coeffs(double x, const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление производной Q'(x) или Q''(x) с переданными коэффициентами
     * @param x точка вычисления
     * @param q внешние коэффициенты
     * @param order порядок производной (1 или 2)
     * @return значение производной
     */
    double evaluate_Q_derivative_with_coeffs(double x, const std::vector<double>& q, int order) const;
    
    /**
     * @brief Вычисление k-й базисной функции φ_k(x)
     * @param x точка вычисления
     * @param q внешние коэффициенты (не используются для вычисления базисной функции, только для валидации размера)
     * @param k индекс базисной функции
     * @return значение φ_k(x)
     */
    double compute_basis_function_with_coeffs(double x, const std::vector<double>& q, int k) const;
    
    /**
     * @brief Вычисление k-й базисной функции и её производных
     * @param x точка вычисления
     * @param q внешние коэффициенты
     * @param k индекс базисной функции
     * @param order порядок производной (0 - значение, 1 - первая, 2 - вторая)
     * @return значение базисной функции или её производной
     */
    double compute_basis_derivative_with_coeffs(double x, const std::vector<double>& q, int k, int order) const;
    
private:
    // Вспомогательные методы для базисных функций
    double compute_basis_function(double x_norm, int k) const;
    double compute_basis_derivative(double x_norm, int k, int order) const;
    
    // Нормализация координат для базиса Чебышёва
    double normalize_x(double x) const { return (x - x_center) / x_scale; }
    
    // Полиномы Чебышёва (рекуррентно)
    static void chebyshev_polynomials(double t, int max_k, std::vector<double>& T);
    static void chebyshev_derivatives(double t, int max_k, std::vector<double>& T, 
                                       std::vector<double>& T1, std::vector<double>& T2);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_CORRECTION_POLYNOMIAL_H
