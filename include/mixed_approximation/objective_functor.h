#ifndef MIXED_APPROXIMATION_OBJECTIVE_FUNCTOR_H
#define MIXED_APPROXIMATION_OBJECTIVE_FUNCTOR_H

#include "types.h"
#include "composite_polynomial.h"
#include "correction_polynomial.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <memory>

namespace mixed_approx {

// ============== Шаг 2.1.9.1: Структуры данных для оптимизации ==============

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

/**
 * @brief Структура для журналирования итерации оптимизации
 */
struct OptimizationIterationLog {
    int iteration;
    double objective_value;
    double approx_term;
    double repel_term;
    double reg_term;
    double gradient_norm;
    double step_size;
    bool barrier_proximity;      // близость к барьеру отталкивания
    bool numerical_anomaly;      // численные аномалии (Inf/NaN)
    std::vector<double> coefficients_snapshot; // опционально для отладки
    
    OptimizationIterationLog()
        : iteration(0), objective_value(0.0), approx_term(0.0)
        , repel_term(0.0), reg_term(0.0), gradient_norm(0.0)
        , step_size(0.0), barrier_proximity(false), numerical_anomaly(false) {}
};

/**
 * @brief Монитор сходимости для диагностики проблем
 */
class ConvergenceMonitor {
public:
    // Параметры мониторинга
    double tol_gradient;           // допуск по градиенту
    double tol_objective;          // допуск по изменению функционала
    double tol_step;               // минимальный шаг
    int max_oscillation_count;     // число итераций для определения осцилляций
    int max_plateau_count;         // число итераций для определения плато
    
    // Конструктор
    ConvergenceMonitor(double tol_grad = 1e-6, double tol_obj = 1e-8)
        : tol_gradient(tol_grad), tol_objective(tol_obj), tol_step(1e-12)
        , max_oscillation_count(5), max_plateau_count(20)
        , current_iteration_(0), oscillation_count_(0)
        , plateau_count_(0), barrier_proximity_count_(0)
        , numerical_anomaly_count_(0), is_diverging_(false) {}
    
    // Проверка критериев сходимости
    bool is_converged(double gradient_norm, double objective_value,
                      double objective_change, double step_size);
    
    // Проверка на осцилляции
    bool detect_oscillation(const std::vector<double>& history);
    
    // Проверка на плато
    bool detect_plateau(double current_objective);
    
    // Проверка на расходимость
    bool detect_divergence(double current_objective);
    
    // Обновление счётчиков проблем
    void update_barrier_proximity(bool active) {
        if (active) barrier_proximity_count_++;
        else barrier_proximity_count_ = 0;
    }
    
    void update_numerical_anomaly(bool active) {
        if (active) numerical_anomaly_count_++;
        else numerical_anomaly_count_ = 0;
    }
    
    // Сброс монитора
    void reset();
    
    // Получение диагностической информации
    std::string get_diagnostic_info() const;
    
    // Получение истории для анализа
    const std::vector<double>& objective_history() const { return objective_history_; }

private:
    int current_iteration_;
    int oscillation_count_;
    int plateau_count_;
    int barrier_proximity_count_;
    int numerical_anomaly_count_;
    bool is_diverging_;
    
    std::vector<double> objective_history_;
    std::vector<double> gradient_history_;
};

// ============== Шаг 2.1.9.1: Функтор стоимости для оптимизатора ==============

/**
 * @brief Функтор для вычисления функционала и его градиента
 * 
 * Архитектура:
 * - Хранит константную ссылку на CompositePolynomial (параметризацию)
 * - Хранит константную ссылку на OptimizationProblemData (данные задачи)
 * - Предоставляет три ключевых метода для оптимизатора:
 *   * value(q) — вычисление значения функционала
 *   * gradient(q, grad) — вычисление градиента
 *   * value_and_gradient(q, f, grad) — комбинированное вычисление
 */
class ObjectiveFunctor {
public:
    /**
     * @brief Конструктор
     * @param parametrisation композитный полином F(x) = P_int(x) + Q(x)·W(x)
     * @param data данные задачи оптимизации
     */
    ObjectiveFunctor(const CompositePolynomial& parametrisation,
                     const OptimizationProblemData& data);
    
    /**
     * @brief Вычисление значения функционала J(q)
     * @param q вектор коэффициентов корректирующего полинома Q(x)
     * @return значение функционала
     */
    double value(const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление градиента ∇J(q)
     * @param q вектор коэффициентов
     * @param grad вектор градиента (будет заполнен)
     */
    void gradient(const std::vector<double>& q, std::vector<double>& grad) const;
    
    /**
     * @brief Комбинированное вычисление значения и градиента
     * @param q вектор коэффициентов
     * @param f значение функционала (будет заполнено)
     * @param grad вектор градиента (будет заполнен)
     */
    void value_and_gradient(const std::vector<double>& q,
                            double& f,
                            std::vector<double>& grad) const;
    
    /**
     * @brief Вычисление компонент функционала для диагностики
     * @param q вектор коэффициентов
     * @return структура с компонентами
     */
    struct Components {
        double approx;
        double repel;
        double reg;
        double total;
    };
    
    Components compute_components(const std::vector<double>& q) const;
    
    /**
     * @brief Построение кэшей для ускорения вычислений
     */
    void build_caches();
    
    /**
     * @brief Обновление кэшей при изменении базиса
     * @param n_free число свободных параметров
     */
    void update_basis_cache(int n_free);
    
    /**
     * @brief Проверка валидности функтора
     */
    bool is_valid() const;
    
    /**
     * @brief Получение метаданных
     */
    int num_free_parameters() const { return param_.num_free_parameters(); }
    const CompositePolynomial& parameterization() const { return param_; }
    const OptimizationProblemData& data() const { return problem_data_; }

private:
    // ============== Компоненты функционала ==============
    
    /**
     * @brief Аппроксимирующий член: J_approx = Σ_i weight_i · (F(x_i) - target_i)²
     */
    double compute_approx_term(const std::vector<double>& q) const;
    
    /**
     * @brief Отталкивающий член: J_repel = Σ_j B_j / distance_j²
     */
    double compute_repel_term(const std::vector<double>& q) const;
    
    /**
     * @brief Регуляризационный член: J_reg = γ · ∫(F''(x))² dx
     */
    double compute_reg_term(const std::vector<double>& q) const;
    
    // ============== Градиенты компонентов ==============
    
    /**
     * @brief Градиент аппроксимирующего члена
     * ∂J_approx/∂q_k = 2 · Σ_i weight_i · residual_i · φ_k(x_i) · W_i
     */
    void compute_approx_gradient(const std::vector<double>& q,
                                std::vector<double>& grad) const;
    
    /**
     * @brief Градиент отталкивающего члена
     * ∂J_repel/∂q_k = -2 · Σ_j [B_j / distance_j³] · sign(...) · φ_k(y_j) · W_j
     */
    void compute_repel_gradient(const std::vector<double>& q,
                                std::vector<double>& grad) const;
    
    /**
     * @brief Градиент регуляризационного члена
     * ∂J_reg/∂q_k = 2γ · Σ_l K[k][l] · q_l
     */
    void compute_reg_gradient(const std::vector<double>& q,
                              std::vector<double>& grad) const;
    
    // ============== Защита от численных аномалий ==============
    
    /**
     * @brief Безопасное вычисление расстояния до барьера
     */
    double safe_barrier_distance(double poly_value, double forbidden_value) const;
    
    /**
     * @brief Проверка на численные аномалии
     */
    bool has_numerical_anomaly(double value) const;
    
    // Константные ссылки на внешние данные
    const CompositePolynomial& param_;
    const OptimizationProblemData& problem_data_;
    
    // Кэш для ускорения вычислений
    mutable OptimizationCache cache_;
    
    // Флаг готовности кэшей
    bool caches_built_;
};

// ============== Шаг 2.1.9.6: Стратегии инициализации ==============

/**
 * @brief Стратегии инициализации коэффициентов перед оптимизацией
 */
enum class InitializationStrategy {
    ZERO,              ///< Нулевая инициализация
    LEAST_SQUARES,     ///< Инициализация через взвешенный МНК
    MULTI_START        ///< Многостартовая инициализация (для n_free ≤ 10)
};

/**
 * @brief Результат инициализации
 */
struct InitializationResult {
    std::vector<double> initial_coeffs;
    double initial_objective;
    bool success;
    std::string message;
    InitializationStrategy strategy_used;
    
    InitializationResult()
        : initial_objective(0.0), success(false), strategy_used(InitializationStrategy::ZERO) {}
};

/**
 * @brief Класс для автоматического выбора стратегии инициализации
 */
class InitializationStrategySelector {
public:
    /**
     * @brief Автоматический выбор стратегии инициализации
     * @param param композитный полином
     * @param data данные задачи
     * @return выбранная стратегия
     */
    static InitializationStrategy select(const CompositePolynomial& param,
                                          const OptimizationProblemData& data);
    
    /**
     * @brief Выполнить инициализацию с выбранной стратегией
     * @param param композитный полином
     * @param data данные задачи
     * @param functor функтор для вычисления функционала
     * @return результат инициализации
     */
    static InitializationResult initialize(const CompositePolynomial& param,
                                            const OptimizationProblemData& data,
                                            ObjectiveFunctor& functor);
    
private:
    // Инициализация нулями
    static InitializationResult zero_initialization(const CompositePolynomial& param);
    
    // Инициализация через МНК
    static InitializationResult least_squares_initialization(const CompositePolynomial& param,
                                                              const OptimizationProblemData& data,
                                                              ObjectiveFunctor& functor);
    
    // Многостартовая инициализация
    static InitializationResult multi_start_initialization(const CompositePolynomial& param,
                                                            const OptimizationProblemData& data,
                                                            ObjectiveFunctor& functor);
};

// ============== Шаг 2.1.9.8: Обработка результатов оптимизации ==============

/**
 * @brief Результат пост-обработки оптимизации
 */
struct PostOptimizationReport {
    // Качество решения
    double max_interpolation_error;
    double min_barrier_distance;
    double final_gradient_norm;
    
    // Баланс компонентов
    double approx_percentage;
    double repel_percentage;
    double reg_percentage;
    
    // Статус
    bool interpolation_satisfied;
    bool barrier_constraints_satisfied;
    bool converged;
    
    // Рекомендации
    std::vector<std::string> recommendations;
    
    PostOptimizationReport()
        : max_interpolation_error(0.0), min_barrier_distance(0.0)
        , final_gradient_norm(0.0), approx_percentage(0.0)
        , repel_percentage(0.0), reg_percentage(0.0)
        , interpolation_satisfied(false)
        , barrier_constraints_satisfied(false), converged(false) {}
};

/**
 * @brief Класс для пост-обработки результатов оптимизации
 */
class OptimizationPostProcessor {
public:
    /**
     * @brief Конструктор
     * @param param композитный полином
     * @param data данные задачи
     */
    OptimizationPostProcessor(const CompositePolynomial& param,
                               const OptimizationProblemData& data);
    
    /**
     * @brief Генерация отчёта о результатах оптимизации
     * @param final_coeffs финальные коэффициенты
     * @param final_objective финальное значение функционала
     * @return отчёт
     */
    PostOptimizationReport generate_report(const std::vector<double>& final_coeffs,
                                            double final_objective);
    
    /**
     * @brief Генерация текстового отчёта
     */
    std::string generate_text_report(const PostOptimizationReport& report);
    
    /**
     * @brief Адаптивная коррекция параметров при дисбалансе
     * @param report текущий отчёт
     * @param config конфигурация для модификации
     */
    void suggest_parameter_corrections(const PostOptimizationReport& report,
                                        ApproximationConfig& config);

private:
    const CompositePolynomial& param_;
    const OptimizationProblemData& data_;
};

// ============== Шаг 2.1.9.9: Интеграция с внешними библиотеками ==============

#ifdef USE_NLOPT

/**
 * @brief Адаптер для библиотеки NLopt
 */
class NloptAdapter {
public:
    /**
     * @brief Конструктор
     * @param functor функтор для вычисления функционала
     */
    explicit NloptAdapter(ObjectiveFunctor* functor);
    
    /**
     * @brief Функция обратного вызова для NLopt
     */
    static double nlopt_objective(unsigned n, const double* q, double* grad, void* data);
    
    /**
     * @brief Создание оптимизатора NLopt
     * @param algorithm алгоритм (NLOPT_LD_LBFGS и т.д.)
     * @return указатель на оптимизатор
     */
    void* create_optimizer(int algorithm);
    
    /**
     * @brief Установка параметров
     */
    void set_parameters(double ftol_rel, double xtol_rel, int max_eval);

};

#endif // USE_NLOPT

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_OBJECTIVE_FUNCTOR_H
