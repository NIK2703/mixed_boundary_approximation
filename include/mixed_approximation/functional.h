#ifndef MIXED_APPROXIMATION_FUNCTIONAL_H
#define MIXED_APPROXIMATION_FUNCTIONAL_H

#include "types.h"
#include "polynomial.h"
#include "composite_polynomial.h"
#include <vector>
#include <string>
#include <limits>

namespace mixed_approx {

// ============== Шаг 3.2: Параметры барьерной защиты ==============

/**
 * @brief Параметры для смягчённой барьерной защиты
 */
struct BarrierParams {
    double epsilon_safe;           ///< Минимальный безопасный порог расстояния
    double smoothing_factor;        ///< Коэффициент сглаживания k (по умолчанию 10)
    double adaptive_gain;          ///< Коэффициент усиления α (по умолчанию 5)
    double warning_zone_factor;    ///< Множитель для определения зоны предупреждения (по умолчанию 10)
    
    BarrierParams()
        : epsilon_safe(1e-8)
        , smoothing_factor(10.0)
        , adaptive_gain(5.0)
        , warning_zone_factor(10.0) {}
    
    /**
     * @brief Вычисление epsilon_safe на основе диапазона значений
     */
    static double compute_epsilon_safe(double max_range) {
        return std::max(1e-8, 1e-6 * max_range);
    }
};

/**
 * @brief Результат вычисления отталкивающего члена с диагностикой
 */
struct RepulsionResult {
    double total;                          ///< Суммарное значение J_repel
    double min_distance;                  ///< Минимальное расстояние до запрещённых точек
    double max_distance;                  ///< Максимальное расстояние
    int critical_count;                   ///< Количество точек в критической зоне
    int warning_count;                    ///< Количество точек в зоне предупреждения
    std::vector<double> distances;        ///< Расстояния для каждой точки
    bool barrier_activated;              ///< Флаг активации барьерной защиты
};

// ============== Шаг 3.2: Параметры нормализации ==============

/**
 * @brief Параметры нормализации компонент функционала
 */
struct NormalizationParams {
    double scale_approx;                  ///< Масштаб аппроксимирующего члена
    double scale_repel;                   ///< Масштаб отталкивающего члена
    double scale_reg;                     ///< Масштаб регуляризационного члена
    
    double weight_approx;                 ///< Нормализованный вес аппроксимации
    double weight_repel;                   ///< Нормализованный вес отталкивания
    double weight_reg;                    ///< Нормализованный вес регуляризации
    
    bool auto_scaling_enabled;           ///< Флаг автоматического масштабирования
    
    NormalizationParams()
        : scale_approx(1.0)
        , scale_repel(1.0)
        , scale_reg(1.0)
        , weight_approx(1.0)
        , weight_repel(1.0)
        , weight_reg(1.0)
        , auto_scaling_enabled(true) {}
};

// ============== Шаг 3.2: Диагностика функционала ==============

/**
 * @brief Структура для хранения диагностической информации о функционале
 */
struct FunctionalDiagnostics {
    // Компоненты функционала
    double raw_approx;                    ///< Сырое значение аппроксимирующего члена
    double raw_repel;                     ///< Сырое значение отталкивающего члена
    double raw_reg;                       ///< Сырое значение регуляризационного члена
    
    double normalized_approx;             ///< Нормализованное значение
    double normalized_repel;              ///< Нормализованное значение
    double normalized_reg;                ///< Нормализованное значение
    
    // Доли компонент в суммарном функционале
    double share_approx;                   ///< Доля аппроксимации (в процентах)
    double share_repel;                   ///< Доля отталкивания (в процентах)
    double share_reg;                     ///< Доля регуляризации (в процентах)
    
    // Диагностика аппроксимации
    double max_residual;                  ///< Максимальный остаток аппроксимации
    double min_residual;                  ///< Минимальный остаток
    double mean_residual;                 ///< Средний остаток
    
    // Диагностика отталкивания
    double min_repulsion_distance;        ///< Минимальное расстояние до запрещённых точек
    double max_repulsion_distance;        ///< Максимальное расстояние
    bool repulsion_barrier_active;        ///< Активен ли барьер
    int repulsion_violations;             ///< Количество нарушений
    
    // Диагностика регуляризации
    double second_deriv_norm;             ///< Норма второй производной
    
    // Общая информация
    double total_functional;              ///< Суммарный функционал
    bool has_numerical_anomaly;           ///< Флаг численной аномалии
    std::string anomaly_description;       ///< Описание аномалии (если есть)
    
    // Веса компонент (для отчёта)
    double weight_approx;                 ///< Вес аппроксимации
    double weight_repel;                  ///< Вес отталкивания
    double weight_reg;                    ///< Вес регуляризации
    
    FunctionalDiagnostics()
        : raw_approx(0.0)
        , raw_repel(0.0)
        , raw_reg(0.0)
        , normalized_approx(0.0)
        , normalized_repel(0.0)
        , normalized_reg(0.0)
        , share_approx(0.0)
        , share_repel(0.0)
        , share_reg(0.0)
        , max_residual(0.0)
        , min_residual(0.0)
        , mean_residual(0.0)
        , min_repulsion_distance(std::numeric_limits<double>::infinity())
        , max_repulsion_distance(0.0)
        , repulsion_barrier_active(false)
        , repulsion_violations(0)
        , second_deriv_norm(0.0)
        , total_functional(0.0)
        , has_numerical_anomaly(false)
        , weight_approx(1.0)
        , weight_repel(1.0)
        , weight_reg(1.0) {}
    
    /**
     * @brief Форматирование диагностики в строку
     */
    std::string format_report() const;
    
    /**
     * @brief Проверка доминирования одной компоненты
     */
    bool is_dominant_component() const {
        return share_approx > 95.0 || share_repel > 95.0 || share_reg > 95.0;
    }
    
    /**
     * @brief Получение рекомендации по корректировке весов
     */
    std::string get_weight_recommendation() const;
};

/**
 * @brief Код завершения вычисления функционала
 */
enum class FunctionalStatus {
    OK,                         ///< Успешное вычисление
    NAN_DETECTED,               ///< Обнаружен NaN
    INF_DETECTED,               ///< Обнаружен Inf
    OVERFLOW,                   ///< Переполнение
    BARRIER_COLLAPSE,           ///< Барьерный коллапс
    EMPTY_APPROX_POINTS,        ///< Пустой набор аппроксимирующих точек
    EMPTY_REPEL_POINTS,         ///< Пустой набор отталкивающих точек
    CRITERIA_CONFLICT          ///< Конфликт критериев
};

/**
 * @brief Результат вычисления функционала с расширенной информацией
 */
struct FunctionalResult {
    double value;                       ///< Значение функционала
    FunctionalStatus status;            ///< Статус вычисления
    FunctionalDiagnostics diagnostics; ///< Диагностическая информация
    
    FunctionalResult()
        : value(0.0)
        , status(FunctionalStatus::OK) {}
    
    bool is_ok() const { return status == FunctionalStatus::OK; }
};

/**
 * @brief Класс для вычисления функционала смешанной аппроксимации и его градиента
 * 
 * Функционал: J = Σ_i |f(x_i) - F(x_i)|^2 / σ_i + Σ_j B_j / |y_j^* - F(y_j)|^2 + γ ∫_a^b (F''(x))^2 dx
 * 
 * Реализует шаг 3.2 с улучшенной барьерной защитой, нормализацией и диагностикой.
 */
class Functional {
private:
    ApproximationConfig config_;
    
public:
    /**
     * @brief Конструктор
     * @param config конфигурация метода
     */
    explicit Functional(const ApproximationConfig& config);
    
    /**
     * @brief Вычисление значения функционала
     * @param poly полином F(x)
     * @return значение J
     */
    double evaluate(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление градиента функционала по коэффициентам полинома
     * @param poly полином F(x)
     * @return градиент (вектор той же размерности, что и коэффициенты poly)
     */
    std::vector<double> gradient(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление компонент функционала для анализа
     * @param poly полином F(x)
     * @return кортеж: (J_approx, J_repel, J_reg, total)
     */
    struct Components {
        double approx_component;
        double repel_component;
        double reg_component;
        double total;
    };
    
    Components get_components(const Polynomial& poly) const;
    
private:
    /**
     * @brief Вычисление аппроксимирующего компонента
     * J_approx = Σ_i |f(x_i) - F(x_i)|^2 / σ_i
     */
    double compute_approx_component(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление отталкивающего компонента
     * J_repel = Σ_j B_j / |y_j^* - F(y_j)|^2
     * Использует защиту от деления на ноль через epsilon
     */
    double compute_repel_component(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление регуляризационного компонента
     * J_reg = γ ∫_a^b (F''(x))^2 dx
     */
    double compute_reg_component(const Polynomial& poly) const;
    
    /**
     * @brief Градиент аппроксимирующего компонента по коэффициентам полинома
     */
    std::vector<double> compute_approx_gradient(const Polynomial& poly) const;
    
    /**
     * @brief Градиент отталкивающего компонента по коэффициентам полинома
     */
    std::vector<double> compute_repel_gradient(const Polynomial& poly) const;
    
    /**
     * @brief Градиент регуляризационного компонента по коэффициентам полинома
     */
    std::vector<double> compute_reg_gradient(const Polynomial& poly) const;
    
    /**
     * @brief Вычисление защищенного расстояния до отталкивающей точки
     * Использует epsilon для предотвращения деления на ноль
     */
    double safe_repel_distance(double poly_value, double target_value) const;
};

// ============== Шаг 2.1.7.10: FunctionalEvaluator для CompositePolynomial ==============

/**
 * @brief Класс для вычисления функционала и его градиента по коэффициентам Q(x)
 * 
 * Предназначен для работы с параметризацией F(x) = P_int(x) + Q(x)·W(x)
 * Оптимизация выполняется по коэффициентам корректирующего полинома Q(x)
 */
class FunctionalEvaluator {
private:
    const ApproximationConfig& config_;
    
    // Параметры барьерной защиты
    BarrierParams barrier_params_;
    
    // Параметры нормализации
    mutable NormalizationParams normalization_params_;
    
public:
    /**
     * @brief Конструктор с параметрами барьерной защиты
     * @param config конфигурация метода
     * @param barrier_params параметры барьерной защиты
     */
    explicit FunctionalEvaluator(const ApproximationConfig& config,
                                const BarrierParams& barrier_params = BarrierParams());
    
    /**
     * @brief Установка параметров барьерной защиты
     */
    void set_barrier_params(const BarrierParams& params);
    
    /**
     * @brief Установка параметров нормализации
     */
    void set_normalization_params(const NormalizationParams& params);
    
    /**
     * @brief Вычисление значения функционала J(q)
     * @param param параметризация (композитный полином)
     * @param q коэффициенты корректирующего полинома Q(x)
     * @return значение функционала
     */
    double evaluate_objective(const CompositePolynomial& param, 
                             const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление градиента ∇J(q)
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @param grad вектор градиента (будет заполнен)
     */
    void evaluate_gradient(const CompositePolynomial& param,
                          const std::vector<double>& q,
                          std::vector<double>& grad) const;
    
    /**
     * @brief Комбинированное вычисление функционала и градиента
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @param f значение функционала (будет заполнено)
     * @param grad вектор градиента (будет заполнен)
     */
    void evaluate_objective_and_gradient(const CompositePolynomial& param,
                                        const std::vector<double>& q,
                                        double& f,
                                        std::vector<double>& grad) const;
    
    /**
     * @brief Вычисление компонент функционала для диагностики
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @return структура с компонентами
     */
    struct Components {
        double approx_component;
        double repel_component;
        double reg_component;
        double total;
    };
    
    Components evaluate_components(const CompositePolynomial& param,
                                   const std::vector<double>& q) const;
    
    /**
     * @brief Расширенное вычисление с полной диагностикой
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @return результат с диагностикой
     */
    FunctionalResult evaluate_with_diagnostics(const CompositePolynomial& param,
                                               const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление отталкивающего члена с расширенной диагностикой
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @return результат с расстояниями и статистикой
     */
    RepulsionResult compute_repel_withDiagnostics(const CompositePolynomial& param,
                                                   const std::vector<double>& q) const;
    
    /**
     * @brief Инициализация параметров нормализации на основе начального приближения
     * @param param параметризация
     * @param q начальные коэффициенты Q(x)
     */
    void initialize_normalization(const CompositePolynomial& param,
                                   const std::vector<double>& q);
    
    /**
     * @brief Получение параметров нормализации
     */
    const NormalizationParams& get_normalization_params() const;
    
    // ============== Шаг 3.3: Градиент с улучшенной барьерной защитой ==============
    
    /**
     * @brief Структура для хранения диагностики градиента
     */
    struct GradientDiagnostics {
        double norm_approx;              ///< Норма градиента аппроксимации
        double norm_repel;               ///< Норма градиента отталкивания
        double norm_reg;                 ///< Норма градиента регуляризации
        double norm_total;               ///< Общая норма градиента
        
        int critical_zone_points;        ///< Количество точек в критической зоне при вычислении градиента отталкивания
        int warning_zone_points;         ///< Количество точек в предупредительной зоне
        
        double max_grad_component;       ///< Максимальная компонента градиента
        double min_grad_component;       ///< Минимальная компонента градиента
        
        std::vector<double> grad_approx; ///< Градиент аппроксимации (для отладки)
        std::vector<double> grad_repel;  ///< Градиент отталкивания (для отладки)
        std::vector<double> grad_reg;    ///< Градиент регуляризации (для отладки)
        
        GradientDiagnostics()
            : norm_approx(0.0), norm_repel(0.0), norm_reg(0.0), norm_total(0.0)
            , critical_zone_points(0), warning_zone_points(0)
            , max_grad_component(0.0), min_grad_component(0.0) {}
    };
    
    /**
     * @brief Результат численной верификации градиента
     */
    struct GradientVerificationResult {
        bool success;                    ///< Флаг успешной верификации
        double relative_error;           ///< Относительная ошибка
        int failed_component;            ///< Индекс компоненты с ошибкой (-1 если все OK)
        std::string message;             ///< Описание результата
        
        GradientVerificationResult()
            : success(false), relative_error(0.0), failed_component(-1) {}
    };
    
    /**
     * @brief Вычисление градиента с устойчивой барьерной защитой (шаг 3.3)
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @param grad вектор градиента (будет заполнен)
     * @param diag диагностика градиента (опционально, может быть nullptr)
     */
    void compute_gradient_robust(const CompositePolynomial& param,
                                 const std::vector<double>& q,
                                 std::vector<double>& grad,
                                 GradientDiagnostics* diag = nullptr) const;
    
    /**
     * @brief Нормализация градиента с адаптивными весами (шаг 3.3, раздел 5)
     * @param grad_approx градиент аппроксимации
     * @param grad_repel градиент отталкивания
     * @param grad_reg градиент регуляризации
     * @param normalized_grad выходной нормализованный градиент
     * @param scaling_factors выходные коэффициенты нормализации
     */
    void normalize_gradient(const std::vector<double>& grad_approx,
                           const std::vector<double>& grad_repel,
                           const std::vector<double>& grad_reg,
                           std::vector<double>& normalized_grad,
                           std::vector<double>& scaling_factors) const;
    
    /**
     * @brief Численная верификация градиента методом конечных разностей (шаг 3.3, раздел 6)
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @param h шаг конечных разностей (по умолчанию 1e-6)
     * @param test_component компонента для тестирования (-1 = все)
     * @return результат верификации
     */
    GradientVerificationResult verify_gradient_numerical(
        const CompositePolynomial& param,
        const std::vector<double>& q,
        double h = 1e-6,
        int test_component = -1) const;
    
    /**
     * @brief Построение кэшей для ускорения вычисления градиента (шаг 3.3, раздел 7)
     * @param param параметризация (будет модифицирована для добавления кэшей)
     * @param points_x аппроксимирующие точки
     * @param points_y отталкивающие точки
     */
    void build_gradient_caches(CompositePolynomial& param,
                              const std::vector<WeightedPoint>& points_x,
                              const std::vector<RepulsionPoint>& points_y) const;
    
    /**
     * @brief Вычисление градиента с использованием кэшей (оптимизированная версия)
     * @param param параметризация с построенными кэшами
     * @param q коэффициенты Q(x)
     * @param grad выходной градиент
     * @param use_normalization использовать ли нормализацию
     */
    void compute_gradient_cached(const CompositePolynomial& param,
                                const std::vector<double>& q,
                                std::vector<double>& grad,
                                bool use_normalization = true) const;
    
    /**
     * @brief Получение диагностики градиента
     * @param param параметризация
     * @param q коэффициенты Q(x)
     * @return структура с диагностикой
     */
    GradientDiagnostics get_gradient_diagnostics(const CompositePolynomial& param,
                                                 const std::vector<double>& q) const;
    
private:
    /**
     * @brief Вычисление аппроксимирующего компонента J_approx = Σ_i |f(x_i) - F(x_i)|^2 / σ_i
     */
    double compute_approx(const CompositePolynomial& param,
                         const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление отталкивающего компонента J_repel = Σ_j B_j / |y_j^* - F(y_j)|^2
     */
    double compute_repel(const CompositePolynomial& param,
                         const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление регуляризационного компонента J_reg = γ ∫ (F''(x))^2 dx
     */
    double compute_regularization(const CompositePolynomial& param,
                                  const std::vector<double>& q) const;
    
    /**
     * @brief Градиент аппроксимирующего компонента по q_k
     */
    void compute_approx_gradient(const CompositePolynomial& param,
                                 const std::vector<double>& q,
                                 std::vector<double>& grad) const;
    
    /**
     * @brief Градиент отталкивающего компонента по q_k
     */
    void compute_repel_gradient(const CompositePolynomial& param,
                                const std::vector<double>& q,
                                std::vector<double>& grad) const;
    
    /**
     * @brief Градиент регуляризационного компонента по q_k
     */
    void compute_reg_gradient(const CompositePolynomial& param,
                               const std::vector<double>& q,
                               std::vector<double>& grad) const;
    
    /**
     * @brief Вычисление смягчённого барьерного члена
     * @param dist расстояние до запрещённой точки
     * @param weight вес B_j
     * @param zone классификация зоны (0 - нормальная, 1 - предупреждение, 2 - критическая)
     * @return значение барьерного члена
     */
    double compute_barrier_term(double dist, double weight, int& zone) const;
    
    /**
     * @brief Вычисление эффективного веса с адаптивным усилением
     */
    double compute_effective_weight(double base_weight, double dist, int zone) const;
    
    /**
     * @brief Проверка на численные аномалии
     */
    bool check_numerical_anomaly(double value, const std::string& component) const;
    
    /**
     * @brief Заполнение диагностики аппроксимации
     */
    void fill_approx_diagnostics(const CompositePolynomial& param,
                                  const std::vector<double>& q,
                                  FunctionalDiagnostics& diag) const;
    
    /**
     * @brief Заполнение диагностики отталкивания
     */
    void fill_repel_diagnostics(const CompositePolynomial& param,
                                 const std::vector<double>& q,
                                 FunctionalDiagnostics& diag,
                                 const RepulsionResult& repel_result) const;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_FUNCTIONAL_H
