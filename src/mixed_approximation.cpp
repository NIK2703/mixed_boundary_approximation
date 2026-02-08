#include "mixed_approximation/mixed_approximation.h"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cmath>
#include <vector>
#include <algorithm>

namespace mixed_approx {

// Статические вспомогательные функции (не являются частью публичного интерфейса)

/**
 * @brief Построение интерполяционного полинома Лагранжа
 * Возвращает полином степени m-1, удовлетворяющий F(z_e) = f(z_e) для всех узлов.
 */
static Polynomial build_lagrange_polynomial(const std::vector<InterpolationNode>& nodes) {
    int m = nodes.size();
    if (m == 0) return Polynomial(0);
    
    // Результат: полином степени m-1
    std::vector<double> result_coeffs(m, 0.0);
    
    for (int i = 0; i < m; ++i) {
        double xi = nodes[i].x;
        double fi = nodes[i].value;
        
        // Базисный полином Лагранжа L_i(x)
        std::vector<double> Li_coeffs(1, 1.0); // степень 0
        
        for (int j = 0; j < m; ++j) {
            if (j == i) continue;
            double xj = nodes[j].x;
            // Умножаем на (x - xj)
            std::vector<double> new_coeffs(Li_coeffs.size() + 1, 0.0);
            for (size_t k = 0; k < Li_coeffs.size(); ++k) {
                new_coeffs[k] += -xj * Li_coeffs[k];   // коэффициент при x^k
                new_coeffs[k+1] += Li_coeffs[k];      // коэффициент при x^{k+1}
            }
            Li_coeffs = new_coeffs;
        }
        
        // Делим на знаменатель: prod_{j≠i} (xi - xj)
        double denom = 1.0;
        for (int j = 0; j < m; ++j) {
            if (j == i) continue;
            denom *= (xi - nodes[j].x);
        }
        if (std::abs(denom) < 1e-12) {
            throw std::runtime_error("Duplicate interpolation nodes detected");
        }
        double scale = fi / denom;
        for (double& c : Li_coeffs) {
            c *= scale;
        }
        
        // Добавляем к общему полиному
        if (Li_coeffs.size() > result_coeffs.size()) {
            result_coeffs.resize(Li_coeffs.size(), 0.0);
        }
        for (size_t k = 0; k < Li_coeffs.size(); ++k) {
            result_coeffs[k] += Li_coeffs[k];
        }
    }
    
    return Polynomial(result_coeffs);
}

/**
 * @brief Построение весового множителя W(x) = Π_{e} (x - z_e)
 * Возвращает полином степени m.
 */
static Polynomial build_interpolation_multiplier(const std::vector<InterpolationNode>& nodes) {
    std::vector<double> coeffs(1, 1.0); // начинаем с 1
    for (const auto& node : nodes) {
        double z = node.x;
        std::vector<double> new_coeffs(coeffs.size() + 1, 0.0);
        for (size_t i = 0; i < coeffs.size(); ++i) {
            new_coeffs[i] += -z * coeffs[i];
            new_coeffs[i+1] += coeffs[i];
        }
        coeffs = new_coeffs;
    }
    return Polynomial(coeffs);
}

// ============== Реализация методов класса MixedApproximation ==============

MixedApproximation::MixedApproximation(const ApproximationConfig& config) 
    : config_(config), polynomial_(build_initial_approximation()), functional_(config) {
    // Валидация конфигурации
    std::string validation_error = Validator::validate(config);
    if (!validation_error.empty()) {
        throw std::invalid_argument("Invalid configuration: " + validation_error);
    }
}

OptimizationResult MixedApproximation::solve(std::unique_ptr<Optimizer> optimizer) {
    if (!optimizer) {
        // Используем адаптивный градиентный спуск по умолчанию
        optimizer = std::make_unique<AdaptiveGradientDescentOptimizer>();
    }
    
    // Начальные коэффициенты
    std::vector<double> initial_coeffs = polynomial_.coefficients();
    
    // Оптимизация
    OptimizationResult result = optimizer->optimize(functional_, initial_coeffs);
    
    // Обновляем полином
    polynomial_ = Polynomial(result.coefficients);
    
    // Проверяем интерполяционные условия
    if (!check_interpolation_conditions(config_.interpolation_tolerance)) {
        result.success = false;
        result.message = "Interpolation conditions not satisfied";
    }
    
    return result;
}

Polynomial MixedApproximation::build_initial_approximation() const {
    int n = config_.polynomial_degree;
    int m = static_cast<int>(config_.interp_nodes.size());
    
    if (m == 0) {
        // Нет интерполяционных узлов, возвращаем нулевой полином
        return Polynomial(n);
    }
    
    if (m == n + 1) {
        // Количество узлов равно степени + 1, интерполяционный полином Лагранжа единственный
        return build_lagrange_polynomial(config_.interp_nodes);
    }
    
    // Строим интерполяционный полином P_int(x) через все узлы (степени m-1)
    Polynomial P_int = build_lagrange_polynomial(config_.interp_nodes);
    
    // Проверка начального приближения на близость к запрещённым значениям (шаг 1.2.6)
    const double epsilon_init = 1e-4;
    bool need_perturbation = false;
    for (const auto& point : config_.repel_points) {
        double poly_value = P_int.evaluate(point.x);
        if (std::abs(point.y_forbidden - poly_value) < epsilon_init) {
            need_perturbation = true;
            break;
        }
    }
    
    Polynomial poly = P_int;
    
    if (need_perturbation && m < n + 1) {
        // Строим множитель Π(x - z_e)
        Polynomial multiplier = build_interpolation_multiplier(config_.interp_nodes);
        
        // Возмущаем, добавляя R(x) = perturb * x^{n-m} * Π(x)
        double perturb = 1e-6;
        const auto& pi_coeffs = multiplier.coefficients();  // [c_m, c_{m-1}, ..., c_0]
        std::vector<double> R_coeffs(n + 1, 0.0);
        // Располагаем коэффициенты Π(x) в R(x) со сдвигом на n-m позиций.
        // pi_coeffs[0] соответствует степени m, после умножения на x^{n-m} получаем степень n.
        for (size_t i = 0; i < pi_coeffs.size() && i < R_coeffs.size(); ++i) {
            R_coeffs[i] = perturb * pi_coeffs[i];
        }
        Polynomial R(R_coeffs);
        poly = poly + R;
    }
    
    // Дополняем коэффициенты нулями до степени n, если необходимо
    auto coeffs = poly.coefficients();
    if (static_cast<int>(coeffs.size()) < n + 1) {
        coeffs.resize(n + 1, 0.0);
    }
    return Polynomial(coeffs);
}

Polynomial MixedApproximation::apply_interpolation_constraints(const Polynomial& poly) const {
    // В данной реализации интерполяционные условия обеспечиваются в build_initial_approximation,
    // поэтому дополнительная коррекция не требуется.
    return poly;
}

bool MixedApproximation::check_interpolation_conditions(double tolerance) const {
    for (const auto& node : config_.interp_nodes) {
        double computed = polynomial_.evaluate(node.x);
        if (std::abs(computed - node.value) > tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<double> MixedApproximation::compute_repel_distances() const {
    std::vector<double> distances;
    for (const auto& point : config_.repel_points) {
        double poly_value = polynomial_.evaluate(point.x);
        distances.push_back(std::abs(point.y_forbidden - poly_value));
    }
    return distances;
}

Functional::Components MixedApproximation::get_functional_components() const {
    return functional_.get_components(polynomial_);
}

Polynomial MixedApproximation::get_polynomial() const {
    return polynomial_;
}

} // namespace mixed_approx
