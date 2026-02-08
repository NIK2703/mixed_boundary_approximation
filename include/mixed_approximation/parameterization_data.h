#ifndef MIXED_APPROXIMATION_PARAMETERIZATION_DATA_H
#define MIXED_APPROXIMATION_PARAMETERIZATION_DATA_H

#include "types.h"
#include "interpolation_basis.h"
#include "weight_multiplier.h"
#include "correction_polynomial.h"
#include "composite_polynomial.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <cstdlib>

namespace mixed_approx {

// ============== Шаг 2.1.7.2: Структура для хранения узлов интерполяции и значений ==============

/**
 * @brief Структура для хранения исходных данных об интерполяционных узлах
 * 
 * Инкапсулирует:
 * - Абсциссы узлов z_e в исходных координатах
 * - Ординаты значений f(z_e)
 * - Нормализованные координаты для численной устойчивости
 * - Метаданные качества данных (минимальное расстояние, диапазон значений)
 * - Статус валидации набора узлов
 */
struct InterpolationNodeSet {
    // Основные данные
    std::vector<double> x_coords;      // Абсциссы узлов z_e в исходных координатах
    std::vector<double> y_values;      // Ординаты значений f(z_e)
    int count;                          // Эффективное число узлов (после объединения близких)
    
    // Нормализованные координаты (для численной устойчивости)
    std::vector<double> x_norm;        // z_e в интервале [-1, 1]
    double norm_center;                 // Центр нормализации: (a + b) / 2
    double norm_scale;                  // Масштаб нормализации: (b - a) / 2
    
    // Метаданные качества данных
    double min_distance;                // Минимальное расстояние между узлами
    double value_range;                 // Диапазон значений: max(f) - min(f)
    bool has_close_nodes;               // Флаг наличия близких узлов (< 1% от (b-a))
    
    // Статус валидации
    bool is_valid;                      // Корректность набора узлов
    std::string validation_error;       // Сообщение об ошибке при некорректности
    
    // Предупреждения (для внутреннего использования)
    std::vector<std::string> warnings_;
    
    InterpolationNodeSet()
        : count(0)
        , norm_center(0.0)
        , norm_scale(1.0)
        , min_distance(0.0)
        , value_range(0.0)
        , has_close_nodes(false)
        , is_valid(false)
        , validation_error("Not initialized") {}
    
    /**
     * @brief Построение набора узлов из исходных данных
     * @param x координаты узлов
     * @param y значения функции в узлах
     * @param interval_start левая граница интервала [a, b]
     * @param interval_end правая граница интервала [a, b]
     * @param merge_threshold порог для объединения близких узлов
     */
    void build(const std::vector<double>& x,
               const std::vector<double>& y,
               double interval_start = 0.0,
               double interval_end = 1.0,
               double merge_threshold = 1e-4);
    
    /**
     * @brief Объединение близких узлов
     * @param epsilon порог близости
     */
    void detect_and_merge_close_nodes(double epsilon);
    
    /**
     * @brief Получение количества узлов
     * @return количество узлов
     */
    int size() const { return static_cast<int>(x_coords.size()); }
    
    /**
     * @brief Получение узла в исходных координатах
     * @param idx индекс узла
     * @return пара (x, y) в исходных координатах
     */
    std::pair<double, double> get_original(int idx) const {
        return {x_coords[idx], y_values[idx]};
    }
    
    /**
     * @brief Получение узла в нормализованных координатах
     * @param idx индекс узла
     * @return пара (x_norm, y) в нормализованных координатах
     */
    std::pair<double, double> get_normalized(int idx) const {
        return {x_norm[idx], y_values[idx]};
    }
    
    /**
     * @brief Преобразование точки из исходных координат в нормализованные
     * @param x точка в исходных координатах
     * @return точка в нормализованных координатах [-1, 1]
     */
    double normalize(double x) const {
        return (x - norm_center) / norm_scale;
    }
    
    /**
     * @brief Преобразование точки из нормализованных координат в исходные
     * @param x_norm точка в нормализованных координатах
     * @return точка в исходных координатах
     */
    double denormalize(double x_norm) const {
        return x_norm * norm_scale + norm_center;
    }
    
    /**
     * @brief Получение диагностической информации
     * @return строка с информацией о наборе узлов
     */
    std::string get_info() const;
    
    /**
     * @brief Добавление предупреждения
     */
    void add_warning(const std::string& msg) {
        warnings_.push_back(msg);
    }
};

// ============== Шаг 2.1.7.7: ParameterizationBuilder ==============

/**
 * @brief Фасад для пошагового построения параметризации с валидацией
 * 
 * Обеспечивает контроль последовательности операций и накопление диагностики
 */
class ParameterizationBuilder {
private:
    // Компоненты параметризации
    InterpolationNodeSet nodes_;
    InterpolationBasis basis_;
    WeightMultiplier weight_;
    CorrectionPolynomial correction_;
    CompositePolynomial composite_;
    
    // Состояние построения
    bool nodes_validated_;
    bool basis_built_;
    bool weight_built_;
    bool correction_built_;
    bool composite_assembled_;
    bool verified_;
    
    // Журнал построения
    std::vector<std::string> build_log_;
    std::vector<std::string> warnings_;
    std::vector<std::string> errors_;
    
public:
    ParameterizationBuilder()
        : nodes_validated_(false)
        , basis_built_(false)
        , weight_built_(false)
        , correction_built_(false)
        , composite_assembled_(false)
        , verified_(false) {}
    
    /**
     * @brief Шаг 1.1: Валидация узлов интерполяции
     * @param config конфигурация задачи
     * @return true, если валидация успешна
     */
    bool validate_nodes(const ApproximationConfig& config);
    
    /**
     * @brief Шаг 1.2: Коррекция формулировки (при необходимости)
     * @param config конфигурация (может быть модифицирована)
     * @return true, если коррекция выполнена успешно
     */
    bool correct_formulation(ApproximationConfig& config);
    
    /**
     * @brief Шаг 2.1.2: Построение базисного интерполяционного полинома
     * @return true, если построение успешно
     */
    bool build_basis();
    
    /**
     * @brief Шаг 2.1.3: Построение весового множителя
     * @return true, если построение успешно
     */
    bool build_weight_multiplier();
    
    /**
     * @brief Шаг 2.1.4: Построение корректирующего полинома
     * @return true, если построение успешно
     */
    bool build_correction_poly();
    
    /**
     * @brief Шаг 2.1.5: Сборка составного полинома
     * @return true, если сборка успешна
     */
    bool assemble_composite();
    
    /**
     * @brief Шаг 2.1.6: Верификация параметризации
     * @return true, если верификация успешна
     */
    bool verify_parameterization();
    
    /**
     * @brief Получение журнала построения
     * @return вектор строк с хронологией построения
     */
    const std::vector<std::string>& get_build_log() const { return build_log_; }
    
    /**
     * @brief Получение предупреждений
     * @return вектор предупреждений
     */
    const std::vector<std::string>& get_warnings() const { return warnings_; }
    
    /**
     * @brief Получение ошибок
     * @return вектор ошибок
     */
    const std::vector<std::string>& get_errors() const { return errors_; }
    
    /**
     * @brief Получение построенной параметризации (move semantics)
     * @return составной полином
     */
    CompositePolynomial take_result() {
        return std::move(composite_);
    }
    
    /**
     * @brief Получение ссылки на параметризацию (const)
     * @return const ссылка на составной полином
     */
    const CompositePolynomial& get_parameterization() const {
        return composite_;
    }
    
    /**
     * @brief Проверка готовности к оптимизации
     * @return true, если параметризация готова
     */
    bool is_ready_for_optimization() const {
        return verified_ && composite_.is_valid();
    }
    
    /**
     * @brief Очистка состояния для повторного построения
     */
    void reset();
    
private:
    /**
     * @brief Добавление записи в журнал
     * @param message сообщение
     * @param is_warning флаг предупреждения
     */
    void log(const std::string& message, bool is_warning = false);
    
    /**
     * @brief Добавление ошибки
     * @param error сообщение об ошибке
     */
    void add_error(const std::string& error) {
        errors_.push_back(error);
        log("ERROR: " + error);
    }
    
    /**
     * @brief Добавление предупреждения
     * @param warning сообщение о предупреждении
     */
    void add_warning(const std::string& warning) {
        warnings_.push_back(warning);
        log("WARNING: " + warning, true);
    }
};

// ============== Шаг 2.1.7.7: ParameterizationWorkspace ==============

/**
 * @brief Управление памятью и временными данными во время оптимизации
 * 
 * Обеспечивает:
 * - Временные буферы для вычислений
 * - Пул памяти для частых аллокаций
 * - Вспомогательные методы для пакетных операций
 */
class ParameterizationWorkspace {
private:
    // Пул памяти для временных векторов
    std::vector<std::vector<double>*> memory_pool_;
    
public:
    ParameterizationWorkspace() = default;
    
    ~ParameterizationWorkspace() {
        // Освобождение всех выделенных векторов из пула
        for (auto ptr : memory_pool_) {
            delete ptr;
        }
    }
    
    // Запрет копирования
    ParameterizationWorkspace(const ParameterizationWorkspace&) = delete;
    ParameterizationWorkspace& operator=(const ParameterizationWorkspace&) = delete;
    
    // Разрешение перемещения
    ParameterizationWorkspace(ParameterizationWorkspace&&) noexcept = default;
    ParameterizationWorkspace& operator=(ParameterizationWorkspace&&) noexcept = default;
    
    /**
     * @brief Выделение памяти для вектора значений
     * @param size размер вектора
     * @return указатель на выделенный вектор
     */
    std::vector<double>* allocate_values(size_t size) {
        auto* ptr = new std::vector<double>(size);
        memory_pool_.push_back(ptr);
        return ptr;
    }
    
    /**
     * @brief Освобождение памяти вектора
     * @param ptr указатель на вектор
     */
    void release_values(std::vector<double>* ptr) {
        if (ptr) {
            auto it = std::find(memory_pool_.begin(), memory_pool_.end(), ptr);
            if (it != memory_pool_.end()) {
                memory_pool_.erase(it);
            }
            delete ptr;
        }
    }
    
    /**
     * @brief Пакетное умножение векторов
     * @param a первый вектор
     * @param b второй вектор
     * @param result результирующий вектор
     * @param size размер векторов
     */
    void compute_batch_product(const double* a, const double* b, double* result, size_t size);
    
    /**
     * @brief Очистка пула памяти
     */
    void clear_pool() {
        for (auto ptr : memory_pool_) {
            delete ptr;
        }
        memory_pool_.clear();
    }
    
    /**
     * @brief Получение размера пула
     * @return количество выделенных векторов
     */
    size_t pool_size() const { return memory_pool_.size(); }
};

// ============== Шаг 2.1.7.10: Интерфейс для оптимизатора ==============

/**
 * @brief Функтор для оптимизатора L-BFGS
 * 
 * Инкапсулирует вычисление функционала и его градиента
 */
class ObjectiveFunctor {
private:
    const CompositePolynomial& param_;   // Ссылка на параметризацию (без изменения)
    const ApproximationConfig& config_;   // Данные задачи
    
public:
    /**
     * @brief Конструктор
     * @param param параметризация
     * @param config конфигурация задачи
     */
    ObjectiveFunctor(const CompositePolynomial& param, const ApproximationConfig& config)
        : param_(param), config_(config) {}
    
    /**
     * @brief Вычисление значения функционала
     * @param q вектор коэффициентов Q(x)
     * @return значение функционала J(q)
     */
    double operator()(const std::vector<double>& q) const;
    
    /**
     * @brief Вычисление градиента
     * @param q вектор коэффициентов
     * @param grad вектор градиента (будет заполнен)
     */
    void gradient(const std::vector<double>& q, std::vector<double>& grad) const;
    
    /**
     * @brief Комбинированное вычисление (для оптимизаторов, требующих одновременного расчёта)
     * @param q вектор коэффициентов
     * @param f значение функционала (будет заполнено)
     * @param grad вектор градиента (будет заполнен)
     */
    void evaluate_with_gradient(const std::vector<double>& q,
                                double& f, std::vector<double>& grad) const;
};

// ============== Шаг 2.1.7.9: Механизмы инвалидации кэшей ==============

/**
 * @brief Статус валидации параметризации
 */
enum class ValidationStatus {
    UNVALIDATED,   // Не валидировано
    PASSED,        // Все тесты пройдены
    WARNING,       // Есть предупреждения
    FAILED         // Есть ошибки
};

/**
 * @brief Результат валидации параметризации
 */
struct ValidationResult {
    ValidationStatus status;
    std::string message;
    double max_interpolation_error;
    double condition_number;
    bool numerically_stable;
    
    ValidationResult()
        : status(ValidationStatus::UNVALIDATED)
        , message("Not validated")
        , max_interpolation_error(0.0)
        , condition_number(0.0)
        , numerically_stable(false) {}
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_PARAMETERIZATION_DATA_H
