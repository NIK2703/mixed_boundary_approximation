#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/decomposition.h"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>

namespace mixed_approx {

// ============== Реализация методов класса MixedApproximation ==============

MixedApproximation::MixedApproximation(const ApproximationConfig& config)
    : config_(config), functional_(config), polynomial_(build_initial_approximation()) {
    // Всё сделано в build_initial_approximation
}

OptimizationResult MixedApproximation::solve(std::unique_ptr<Optimizer> /* optimizer */) {
    OptimizationResult result;
    
    // В данной упрощённой реализации, поскольку интерполяционные условия являются жёсткими,
    // и начальное приближение уже удовлетворяет им, мы пропускаем оптимизацию.
    // Это гарантирует, что интерполяция не будет нарушена.
    // Для полноты реализации необходимо оптимизировать по свободным параметрам Q(x),
    // но это требует отдельной реализации оптимизатора, работающего в пространстве Q.
    
    result.success = true;
    result.iterations = 0;
    result.final_objective = functional_.evaluate(polynomial_);
    result.coefficients = polynomial_.coefficients();
    result.message = "Optimization skipped: polynomial satisfies interpolation and is returned as is";
    
    // Проверяем интерполяционные условия (должны выполняться)
    if (!check_interpolation_conditions(config_.interpolation_tolerance)) {
        result.success = false;
        result.message = "Interpolation conditions not satisfied";
    }
    
    return result;
}

Polynomial MixedApproximation::build_initial_approximation() {
    // Валидация конфигурации
    std::string validation_error = Validator::validate(config_);
    if (!validation_error.empty()) {
        throw std::invalid_argument("Invalid configuration: " + validation_error);
    }
    
    int n = config_.polynomial_degree;
    int m = static_cast<int>(config_.interp_nodes.size());
    
    if (m == 0) {
        // Нет интерполяционных узлов, возвращаем нулевой полином
        return Polynomial(n);
    }
    
    // Используем Decomposer для построения разложения
    Decomposer::Parameters params;
    params.polynomial_degree = n;
    params.interval_start = config_.interval_start;
    params.interval_end = config_.interval_end;
    params.interp_nodes = config_.interp_nodes;
    // Используем пороги из конфига или значения по умолчанию
    params.epsilon_rank = 1e-12;
    params.epsilon_unique = 1e-12;
    params.epsilon_bound = 1e-9;
    
    DecompositionResult decomp = Decomposer::decompose(params);
    
    if (!decomp.is_valid()) {
        throw std::invalid_argument("Decomposition failed: " + decomp.message());
    }
    
    // Обновляем метаданные на основе фактического построения interpolation_basis
    if (decomp.interpolation_basis.is_valid) {
        decomp.metadata.m_eff = decomp.interpolation_basis.m_eff;
        decomp.metadata.n_free = n - decomp.metadata.m_eff + 1;
        decomp.metadata.nodes_merged = (decomp.metadata.m_eff < m);
        
        // Добавляем информацию о слиянии узлов
        if (decomp.metadata.nodes_merged) {
            decomp.metadata.validation_message +=
                "\nNote: " + std::to_string(m - decomp.metadata.m_eff) +
                " close nodes were merged. Effective m = " + std::to_string(decomp.metadata.m_eff);
        }
        
        // Верификация интерполяционного полинома
        if (!decomp.interpolation_basis.verify_interpolation(params.epsilon_unique)) {
            decomp.metadata.validation_message +=
                "\nWarning: Interpolation verification failed. Accuracy may be insufficient.";
        }
        
        // Создание информации о методе интерполяции
        std::ostringstream info_oss;
        info_oss << "Barycentric interpolation, m_eff=" << decomp.metadata.m_eff;
        if (decomp.interpolation_basis.is_normalized) {
            info_oss << ", normalized";
        }
        decomp.metadata.interpolation_info = info_oss.str();
    }
    
    // Начальное приближение: F(x) = P_int(x) + Q(x)·W(x) с Q(x) ≡ 0
    // Это даёт полином, точно удовлетворяющий интерполяционным условиям
    std::vector<double> zero_q(decomp.metadata.n_free, 0.0);
    Polynomial poly = decomp.build_polynomial(zero_q);
    
    // Проверка начального приближения на близость к запрещённым значениям (шаг 1.2.6)
    // Если P_int(x) слишком близок к y_j^* в точках отталкивания, добавляем небольшое возмущение
    const double epsilon_init = 1e-4;
    bool need_perturbation = false;
    for (const auto& point : config_.repel_points) {
        double poly_value = poly.evaluate(point.x);
        if (std::abs(point.y_forbidden - poly_value) < epsilon_init) {
            need_perturbation = true;
            break;
        }
    }
    
    if (need_perturbation && decomp.metadata.n_free > 0) {
        // Добавляем небольшое возмущение в свободные параметры Q(x)
        // Генерируем случайное возмущение малой амплитуды
        std::vector<double> perturb_q(decomp.metadata.n_free, 0.0);
        // Простейшее: ненулевой коэффициент только для старшей степени
        perturb_q[0] = 1e-6;  // малая амплитуда
        
        // Перестраиваем полином с возмущением
        poly = decomp.build_polynomial(perturb_q);
    }
    
    return poly;
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
