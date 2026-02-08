#ifndef MIXED_APPROXIMATION_GAUSS_QUADRATURE_H
#define MIXED_APPROXIMATION_GAUSS_QUADRATURE_H

#include <vector>

namespace mixed_approx {

/**
 * @brief Получение узлов и весов квадратуры Гаусса-Лежандра
 * @param n требуемое число узлов
 * @param nodes выходной вектор узлов
 * @param weights выходной вектор весов
 */
void get_gauss_legendre_quadrature(int n, std::vector<double>& nodes, std::vector<double>& weights);

/**
 * @brief Преобразование координат из [-1, 1] в [a, b]
 * @param t координата в стандартном интервале [-1, 1]
 * @param a левая граница целевого интервала
 * @param b правая граница целевого интервала
 * @return координата в интервале [a, b]
 */
double transform_to_standard_interval(double t, double a, double b);

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_GAUSS_QUADRATURE_H
