#ifndef MIXED_APPROXIMATION_POLYNOMIAL_H
#define MIXED_APPROXIMATION_POLYNOMIAL_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include "types.h"

namespace mixed_approx {

/**
 * @brief Класс для работы с алгебраическим полиномом
 * 
 * Полином представляется в виде: P(x) = a_n * x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0
 * Коэффициенты хранятся в порядке убывания степеней: [a_n, a_{n-1}, ..., a_0]
 */
class Polynomial {
private:
    std::vector<double> coeffs_;  // коэффициенты [a_n, a_{n-1}, ..., a_0]
    int degree_;                   // степень полинома
    
public:
    /**
     * @brief Конструктор по коэффициентам
     * @param coeffs вектор коэффициентов в порядке убывания степеней
     */
    Polynomial(const std::vector<double>& coeffs);
    
    /**
     * @brief Конструктор нулевого полинома заданной степени
     * @param degree степень полинома
     */
    explicit Polynomial(int degree);
    
    /**
     * @brief Вычисление значения полинома в точке x (схема Горнера)
     * @param x точка вычисления
     * @return значение P(x)
     */
    double evaluate(double x) const;
    
    /**
     * @brief Вычисление первой производной в точке x
     * @param x точка вычисления
     * @return значение P'(x)
     */
    double derivative(double x) const;
    
    /**
     * @brief Вычисление второй производной в точке x
     * @param x точка вычисления
     * @return значение P''(x)
     */
    double second_derivative(double x) const;
    
    /**
     * @brief Получение вектора коэффициентов
     * @return константная ссылка на коэффициенты
     */
    const std::vector<double>& coefficients() const { return coeffs_; }
    
    /**
     * @brief Получение степени полинома
     * @return степень
     */
    int degree() const { return degree_; }
    
    /**
     * @brief Установка коэффициентов
     * @param coeffs новый вектор коэффициентов
     */
    void setCoefficients(const std::vector<double>& coeffs);
    
    /**
     * @brief Сложение двух полиномов
     * @param other второй полином
     * @return результат сложения
     */
    Polynomial operator+(const Polynomial& other) const;
    
    /**
     * @brief Вычитание полиномов
     * @param other второй полином
     * @return результат вычитания
     */
    Polynomial operator-(const Polynomial& other) const;
    
    /**
     * @brief Умножение полинома на скаляр
     * @param scalar множитель
     * @return результат умножения
     */
    Polynomial operator*(double scalar) const;
    
    /**
     * @brief Вычисление значения квадрата отклонения |P(x) - target|^2
     * @param x точка
     * @param target целевое значение
     * @return квадрат отклонения
     */
    double squared_error(double x, double target) const;
    
    /**
     * @brief Вычисление градиента квадрата отклонения по коэффициентам
     * ∂/∂a_k |P(x) - target|^2 = 2 * (P(x) - target) * x^{n-k}
     * @param x точка
     * @param target целевое значение
     * @return градиент (вектор той же размерности, что и коэффициенты)
     */
    std::vector<double> gradient_squared_error(double x, double target) const;
};

// ============== Вспомогательные функции ==============

/**
 * @brief Вычисление интеграла от (P''(x))^2 на [a, b] аналитически
 * Для полинома P(x) = Σ a_k x^k, P''(x) = Σ k*(k-1)*a_k x^{k-2}
 * Интеграл вычисляется как Σ_i Σ_j a_i * a_j * (b^{i+j-3} - a^{i+j-3}) / (i+j-3)
 * @param poly полином
 * @param a начало интервала
 * @param b конец интервала
 * @return значение интеграла
 */
double integrate_second_derivative_squared(const Polynomial& poly, double a, double b);

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_POLYNOMIAL_H
