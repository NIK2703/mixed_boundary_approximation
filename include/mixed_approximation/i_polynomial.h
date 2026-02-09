#ifndef MIXED_APPROXIMATION_IPOLYNOMIAL_H
#define MIXED_APPROXIMATION_IPOLYNOMIAL_H

#include <vector>
#include <string>
#include <functional>
#include <array>

namespace mixed_approx {

/**
 * @brief Базовый тип полинома (мономиальный или Чебышёва)
 */
enum class BasisType {
    MONOMIAL,     // Стандартный мономиальный базис {1, x, x², ...}
    CHEBYSHEV     // Базис полиномов Чебышёва {T₀, T₁, T₂, ...}
};

/**
 * @brief Результат оценки полинома с производными
 */
struct EvaluationResult {
    double value;        // F(x)
    double first_deriv;  // F'(x)
    double second_deriv; // F''(x)
    
    EvaluationResult() : value(0.0), first_deriv(0.0), second_deriv(0.0) {}
    EvaluationResult(double v, double f, double s) : value(v), first_deriv(f), second_deriv(s) {}
};

/**
 * @brief Абстрактный интерфейс полинома
 * 
 * Принцип разделения интерфейса (ISP):
 * - Методы оценки функции отделены от методов оптимизации
 * - Методы базисных функций выделены для градиента
 * 
 * Все методы noexcept где возможно для оптимизации.
 */
class IPolynomial {
public:
    virtual ~IPolynomial() = default;
    
    // ============== Методы оценки функции ==============
    
    /**
     * @brief Вычисление значения полинома в точке
     * @param x точка вычисления
     * @return значение P(x)
     */
    virtual double evaluate(double x) const noexcept = 0;
    
    /**
     * @brief Безопасная версия evaluate с проверкой на NaN/Inf
     * @param x точка вычисления
     * @param result выходной результат
     * @return true если оценка успешна, false при ошибке
     */
    virtual bool evaluate_safe(double x, double& result) const noexcept = 0;
    
    /**
     * @brief Пакетная оценка значения и производных
     * @param x точка вычисления
     * @param f значение F(x)
     * @param f1 первая производная F'(x)
     * @param f2 вторая производная F''(x)
     */
    virtual void derivatives(double x, double& f, double& f1, double& f2) const noexcept = 0;
    
    /**
     * @brief Пакетная оценка в структурированном виде
     * @param x точка вычисления
     * @return структура с F, F', F''
     */
    virtual EvaluationResult evaluate_with_derivatives(double x) const noexcept = 0;
    
    // ============== Методы вычисления производных ==============
    
    /**
     * @brief Первая производная в точке
     * @param x точка вычисления
     * @return значение P'(x)
     */
    virtual double first_derivative(double x) const noexcept = 0;
    
    /**
     * @brief Вторая производная в точке
     * @param x точка вычисления
     * @return значение P''(x)
     */
    virtual double second_derivative(double x) const noexcept = 0;
    
    // ============== Методы для оптимизации ==============
    
    /**
     * @brief Степень полинома
     * @return степень n
     */
    virtual std::size_t degree() const noexcept = 0;
    
    /**
     * @brief Число свободных параметров
     * @return количество параметров для оптимизации
     */
    virtual std::size_t num_parameters() const noexcept = 0;
    
    /**
     * @brief Получение параметра по индексу
     * @param index индекс параметра (0..num_parameters-1)
     * @return значение параметра
     * @throws std::out_of_range при неверном индексе
     */
    virtual double parameter(std::size_t index) const = 0;
    
    /**
     * @brief Установка параметра по индексу
     * @param index индекс параметра
     * @param value новое значение
     * @throws std::out_of_range при неверном индексе
     * @throws std::domain_error при NaN/Inf
     */
    virtual void set_parameter(std::size_t index, double value) = 0;
    
    /**
     * @brief Получение всех параметров
     * @return вектор параметров
     */
    virtual std::vector<double> parameters() const = 0;
    
    /**
     * @brief Установка всех параметров
     * @param params вектор параметров
     */
    virtual void set_parameters(const std::vector<double>& params) = 0;
    
    // ============== Методы базисных функций ==============
    
    /**
     * @brief Оценка k-й базисной функции φₖ(x)
     * @param k индекс базисной функции
     * @param x точка вычисления
     * @return значение φₖ(x)
     */
    virtual double basis_function(std::size_t k, double x) const = 0;
    
    /**
     * @brief Производная k-й базисной функции
     * @param k индекс базисной функции
     * @param x точка вычисления
     * @param order порядок производной (1 или 2)
     * @return значение d^order φₖ(x) / dx^order
     */
    virtual double basis_derivative(std::size_t k, double x, int order) const = 0;
    
    /**
     * @brief Градиент по параметрам в точке x
     * @param x точка вычисления
     * @return вектор градиента [∂F/∂a₀, ∂F/∂a₁, ...]
     */
    virtual std::vector<double> gradient(double x) const = 0;
    
    // ============== Вспомогательные методы ==============
    
    /**
     * @brief Текстовое представление для отладки
     * @return строковое представление полинома
     */
    virtual std::string to_string() const = 0;
    
    /**
     * @brief Внутренняя проверка корректности состояния
     * @return true если состояние корректно
     */
    virtual bool validate() const = 0;
    
    /**
     * @brief Сброс кэшей при изменении параметров
     */
    virtual void reset_cache() = 0;
    
    /**
     * @brief Тип базиса
     * @return MONOMIAL или CHEBYSHEV
     */
    virtual BasisType basis_type() const noexcept = 0;
    
    /**
     * @brief Границы интервала определения
     * @return пара (a, b)
     */
    virtual std::array<double, 2> interval() const noexcept = 0;
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_IPOLYNOMIAL_H
