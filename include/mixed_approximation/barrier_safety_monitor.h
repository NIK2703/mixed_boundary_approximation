#ifndef MIXED_APPROXIMATION_BARRIER_SAFETY_MONITOR_H
#define MIXED_APPROXIMATION_BARRIER_SAFETY_MONITOR_H

#include "types.h"
#include <vector>
#include <string>
#include <deque>
#include <memory>

namespace mixed_approx {

// Forward declarations
class CompositePolynomial;

/**
 * @brief Зона приближения к барьеру
 */
enum class BarrierZone {
    SAFE,       // Безопасная зона: |δ| > ε_warning
    WARNING,    // Предупредительная зона: ε_critical < |δ| ≤ ε_warning
    CRITICAL    // Критическая зона: |δ| ≤ ε_critical
};

/**
 * @brief Тип барьерной защиты
 */
enum class BarrierProtectionType {
    NONE,               // Без защиты
    SMOOTHING,          // Сглаживание (quadratic smoothing)
    LOGARITHMIC,        // Логарифмический барьер
    HYBRID,             // Гибридный барьер
    GRADIENT_CLIPPING,  // Ограничение градиента
    STEP_DAMPING        // Демпфирование шага
};

/**
 * @brief Состояние одного барьера (отталкивающей точки)
 */
struct BarrierState {
    double current_distance;      // текущее |δ|
    double min_distance_10it;     // минимум за 10 итераций
    BarrierZone zone;              // классификация зоны
    bool approaching;              // признак сближения (δ уменьшается)
    int iterations_in_zone;       // итераций в текущей зоне подряд
    double weight;                // вес B_j
    
    BarrierState()
        : current_distance(0.0)
        , min_distance_10it(0.0)
        , zone(BarrierZone::SAFE)
        , approaching(false)
        , iterations_in_zone(0)
        , weight(0.0) {}
};

/**
 * @brief Событие барьерной защиты
 */
struct BarrierEvent {
    int iteration;                    // номер итерации
    int barrier_index;                // индекс отталкивающей точки
    double distance;                  // расстояние до барьера
    BarrierZone zone;                 // зона барьера
    BarrierProtectionType protection; // применённый тип защиты
    double damping_factor;            // коэффициент демпфирования
    double functional_value;          // значение функционала
    bool step_rejected;               // был ли отклонён шаг оптимизатора
    
    BarrierEvent()
        : iteration(0)
        , barrier_index(0)
        , distance(0.0)
        , zone(BarrierZone::SAFE)
        , protection(BarrierProtectionType::NONE)
        , damping_factor(1.0)
        , functional_value(0.0)
        , step_rejected(false) {}
};

/**
 * @brief Результат проверки барьерной безопасности
 */
struct BarrierSafetyResult {
    bool is_safe;                     // безопасна ли текущая итерация
    double danger_score;              // оценка опасности [0, 1]
    double step_damping_factor;       // коэффициент демпфирования шага
    bool step_rejection_recommended;  // рекомендация отклонить шаг
    bool recovery_required;           // требуется ли восстановление
    int critical_count;              // количество критических барьеров
    int warning_count;                // количество предупредительных барьеров
    std::vector<BarrierState> barrier_states; // состояния всех барьеров
    
    BarrierSafetyResult()
        : is_safe(true)
        , danger_score(0.0)
        , step_damping_factor(1.0)
        , step_rejection_recommended(false)
        , recovery_required(false)
        , critical_count(0)
        , warning_count(0) {}
};

/**
 * @brief Параметры барьерной защиты
 */
struct BarrierSafetyConfig {
    // Базовые пороги (статические)
    double epsilon_critical_base;     // абсолютный минимум (1e-8)
    double epsilon_warning_base;     // порог начала осторожного режима (1e-4)
    
    // Коэффициенты адаптации
    double scale_factor_small;        // множитель для масштабирования (1e-10)
    double scale_factor_large;       // множитель для предупредительной зоны (1e-6)
    
    // Параметры сглаживания
    double smoothing_k;              // коэффициент сглаживания k (5.0)
    bool use_quadratic_smoothing;    // использовать квадратичное сглаживание
    
    // Параметры ограничения градиента
    double gradient_max_per_component;   // максимум на компоненту (1e8 / n_free)
    double gradient_max_norm;           // максимум нормы (1e6)
    double gradient_adaptive_factor;     // адаптивный фактор (10.0)
    
    // Параметры демпфирования шага
    double danger_threshold_warning;    // порог опасности для предупреждения (0.5)
    double danger_threshold_critical;   // порог опасности для критической ситуации (0.9)
    double step_damping_warning;         // демпфирование в предупредительной зоне (0.5)
    double step_damping_critical;        // демпфирование в критической зоне (0.1)
    
    // Параметры восстановления после коллапса
    double collapse_functional_ratio;    // порог для определения коллапса (100.0)
    double collapse_distance_ratio;      // порог расстояния для коллапса (0.1)
    double temporary_weight_reduction;   // временное уменьшение веса (0.1)
    double temporary_gamma_multiplier;    // временное увеличение γ (2.0)
    
    // История и диагностика
    int history_size;                    // размер истории для адаптации (10)
    int max_recovery_attempts;           // максимум попыток восстановления (3)
    bool enable_logging;                 // включить логирование событий
    
    BarrierSafetyConfig()
        : epsilon_critical_base(1e-8)
        , epsilon_warning_base(1e-4)
        , scale_factor_small(1e-10)
        , scale_factor_large(1e-6)
        , smoothing_k(5.0)
        , use_quadratic_smoothing(true)
        , gradient_max_per_component(1e8)
        , gradient_max_norm(1e6)
        , gradient_adaptive_factor(10.0)
        , danger_threshold_warning(0.5)
        , danger_threshold_critical(0.9)
        , step_damping_warning(0.5)
        , step_damping_critical(0.1)
        , collapse_functional_ratio(100.0)
        , collapse_distance_ratio(0.1)
        , temporary_weight_reduction(0.1)
        , temporary_gamma_multiplier(2.0)
        , history_size(10)
        , max_recovery_attempts(3)
        , enable_logging(true) {}
    
    /**
     * @brief Создание конфигурации с адаптивными порогами
     */
    static BarrierSafetyConfig create_adaptive(const ApproximationConfig& config);
};

/**
 * @brief Результат барьерного восстановления
 */
struct BarrierRecoveryResult {
    bool success;                       // успешность восстановления
    bool rollback_performed;            // был ли выполнён откат
    double escaped_distance;            // расстояние после восстановления
    std::string message;                // сообщение о результате
    std::vector<double> corrected_params; // скорректированные параметры
    
    BarrierRecoveryResult()
        : success(false)
        , rollback_performed(false)
        , escaped_distance(0.0) {}
};

/**
 * @brief Класс для мониторинга и защиты от барьерных особенностей
 * 
 * Реализует шаг 5.1: защита от особенностей отталкивающего члена
 * 
 * Функциональность:
 * - Многоуровневая система порогов безопасности (critical, warning, safe)
 * - Сглаживающие функции для критической зоны
 * - Защита градиента в критической зоне
 * - Динамическая адаптация шага оптимизатора
 * - Обнаружение и восстановление после барьерного коллапса
 * - Диагностическая система мониторинга барьеров
 */
class BarrierSafetyMonitor {
public:
    /**
     * @brief Конструктор
     * @param config конфигурация аппроксимации
     * @param safety_config параметры барьерной защиты
     */
    BarrierSafetyMonitor(const ApproximationConfig& config,
                         const BarrierSafetyConfig& safety_config = BarrierSafetyConfig());
    
    /**
     * @brief Конструктор с отталкивающими точками
     * @param repel_points отталкивающие точки
     * @param safety_config параметры барьерной защиты
     */
    BarrierSafetyMonitor(const std::vector<RepulsionPoint>& repel_points,
                         const BarrierSafetyConfig& safety_config = BarrierSafetyConfig());
    
    /**
     * @brief Сброс состояния монитора
     */
    void reset();
    
    // ============== Основные операции ==============
    
    /**
     * @brief Проверка безопасности и получение коэффициента демпфирования
     * @param distances текущие расстояния до барьеров
     * @param weights веса барьеров B_j
     * @param functional_value текущее значение функционала
     * @return результат проверки безопасности
     */
    BarrierSafetyResult check_safety(const std::vector<double>& distances,
                                     const std::vector<double>& weights,
                                     double functional_value);
    
    /**
     * @brief Вычисление сглаженного барьерного члена
     * @param distance расстояние до барьера
     * @param weight вес барьера B_j
     * @param zone зона барьера (выходной параметр)
     * @return значение барьерного члена
     */
    double compute_smoothed_barrier(double distance, double weight, BarrierZone& zone);
    
    /**
     * @brief Вычисление защищённого градиента отталкивания
     * @param raw_gradient "сырой" градиент отталкивания
     * @param distances текущие расстояния до барьеров
     * @return защищённый градиент
     */
    std::vector<double> protect_gradient(const std::vector<double>& raw_gradient,
                                         const std::vector<double>& distances);
    
    /**
     * @brief Получение коэффициента демпфирования шага
     * @return коэффициент ∈ [0.01, 1.0]
     */
    double get_step_damping_factor() const;
    
    /**
     * @brief Проверка на барьерный коллапс
     * @param current_functional текущее значение функционала
     * @param previous_functional предыдущее значение функционала
     * @param distances текущие расстояния
     * @return true если обнаружен коллапс
     */
    bool detect_collapse(double current_functional,
                         double previous_functional,
                         const std::vector<double>& distances);
    
    /**
     * @brief Восстановление после коллапса
     * @param current_params текущие параметры
     * @param safe_params безопасные параметры (из истории)
     * @param distances текущие расстояния до барьеров
     * @param weights веса барьеров
     * @return результат восстановления
     */
    BarrierRecoveryResult recover_from_collapse(const std::vector<double>& current_params,
                                                const std::vector<double>& safe_params,
                                                const std::vector<double>& distances,
                                                const std::vector<double>& weights);
    
    /**
     * @brief Предотвращающая коррекция начального приближения
     * @param distances расстояния до барьеров
     * @param weights веса барьеров
     * @param scale_y характерный масштаб значений
     * @return true если коррекция применена
     */
    bool apply_preventive_correction(std::vector<double>& distances,
                                      const std::vector<double>& weights,
                                      double scale_y);
    
    // ============== Диагностика и мониторинг ==============
    
    /**
     * @brief Обновление истории для адаптивной коррекции порогов
     * @param distances текущие расстояния
     */
    void update_history(const std::vector<double>& distances);
    
    /**
     * @brief Получение текущих адаптивных порогов
     * @return пара (epsilon_critical, epsilon_warning)
     */
    std::pair<double, double> get_adaptive_thresholds() const;
    
    /**
     * @brief Получение истории событий
     */
    const std::vector<BarrierEvent>& get_events() const { return events_; }
    
    /**
     * @brief Очистка истории событий
     */
    void clear_events() { events_.clear(); }
    
    /**
     * @brief Получение диагностического отчёта
     * @return строка с отчётом
     */
    std::string generate_report() const;
    
    /**
     * @brief Проверка критического предупреждения (устойчивые проблемы)
     * @return true если есть критические проблемы
     */
    bool has_critical_warning() const;
    
    /**
     * @brief Получение рекомендаций по параметрам
     * @return вектор рекомендаций
     */
    std::vector<std::string> get_recommendations() const;
    
    /**
     * @brief Установка безопасных параметров для потенциального отката
     * @param params безопасные параметры
     */
    void set_safe_parameters(const std::vector<double>& params);
    
    /**
     * @brief Получение последних безопасных параметров
     */
    const std::vector<double>& get_safe_parameters() const { return safe_parameters_; }
    
private:
    // Конфигурация
    const ApproximationConfig& config_;
    BarrierSafetyConfig safety_config_;
    
    // Отталкивающие точки
    std::vector<RepulsionPoint> repel_points_;
    
    // Адаптивные пороги
    double epsilon_critical_;
    double epsilon_warning_;
    
    // История для адаптации
    std::deque<std::vector<double>> distance_history_;
    
    // Безопасные параметры для отката
    std::vector<double> safe_parameters_;
    
    // История значений функционала
    std::deque<double> functional_history_;
    
    // События барьерной защиты
    std::vector<BarrierEvent> events_;
    
    // Состояния барьеров
    std::vector<BarrierState> barrier_states_;
    
    // Счётчики
    int collapse_count_;
    int recovery_attempts_;
    
    /**
     * @brief Инициализация состояний барьеров
     */
    void initialize_barrier_states();
    
    /**
     * @brief Вычисление характерного масштаба
     */
    double compute_characteristic_scale() const;
    
    /**
     * @brief Обновление адаптивных порогов
     */
    void update_adaptive_thresholds();
    
    /**
     * @brief Классификация зоны по расстоянию
     */
    BarrierZone classify_zone(double distance) const;
    
    /**
     * @brief Логирование события
     */
    void log_event(const BarrierEvent& event);
    
    /**
     * @brief Вычисление оценки опасности
     */
    double compute_danger_score(const std::vector<double>& distances,
                                const std::vector<double>& weights) const;
    
    /**
     * @brief Вычисление направления "от барьера"
     */
    std::vector<double> compute_escape_direction(const std::vector<double>& distances,
                                                  const std::vector<double>& weights);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_BARRIER_SAFETY_MONITOR_H
