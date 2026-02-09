#include "mixed_approximation/edge_case_handler.h"
#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

namespace mixed_approx {

// ============== Шаг 2.1.8.2: Обработка случая m = 0 ==============

ZeroNodesResult EdgeCaseHandler::handle_zero_nodes(int n) {
    ZeroNodesResult result;
    
    result.use_trivial_parameterization = true;
    result.degree = n;
    
    std::ostringstream oss;
    oss << "INFO: No interpolation nodes (m = 0).\n";
    oss << "      Task reduces to regularized approximation without constraints.\n";
    oss << "      Parameterization simplified: F(x) = Q(x) with degree " << n << ".\n";
    result.info_message = oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.3: Обработка случая m = n + 1 ==============

FullInterpolationResult EdgeCaseHandler::handle_full_interpolation(int /*n*/, int m) {
    FullInterpolationResult result;
    
    result.is_degenerate = true;
    result.n_free = 0;
    
    std::ostringstream oss;
    oss << "WARNING: Full interpolation (m = n + 1 = " << m << ").\n";
    oss << "         Solution is uniquely defined by interpolation conditions.\n";
    oss << "         Approximation and repulsion criteria are inactive.\n";
    oss << "         Correction polynomial degenerates: Q(x) = 0.\n";
    result.info_message = oss.str();
    
    std::ostringstream rec_oss;
    rec_oss << "Recommendations:\n";
    rec_oss << "  1. To include approximation data, increase polynomial degree to n >= " << m << "\n";
    rec_oss << "  2. To include repulsion, increase polynomial degree to n >= " << (m + 1) << "\n";
    rec_oss << "  3. Current solution: pure Lagrange interpolation of degree " << (m - 1) << "\n";
    result.recommendations = rec_oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.4: Обработка избыточных ограничений ==============

OverconstrainedResult EdgeCaseHandler::handle_overconstrained(int n, int m,
                                                             const std::vector<double>& interp_coords,
                                                             const std::vector<double>& /*interp_values*/,
                                                             OverconstrainedStrategy strategy) {
    OverconstrainedResult result;
    
    result.original_m = m;
    result.strategy = strategy;
    
    if (strategy == OverconstrainedStrategy::STRICT) {
        result.resolved = false;
        std::ostringstream oss;
        oss << "ERROR: Overconstrained system (m = " << m << ", n = " << n << ").\n";
        oss << "       Maximum allowed nodes for degree " << n << " is n+1 = " << (n + 1) << ".\n";
        oss << "       Recommended actions:\n";
        oss << "         1. Reduce number of nodes to " << (n + 1) << " or fewer\n";
        oss << "         2. Increase polynomial degree to " << (m - 1) << " or more\n";
        oss << "         3. Convert some nodes to approximation points with high weights\n";
        result.info_message = oss.str();
        return result;
    }
    
    // Мягкая стратегия: выбор подмножества узлов
    int max_nodes = n + 1;
    result.adapted_m = std::min(m, max_nodes);
    
    if (m <= max_nodes) {
        result.resolved = true;
        result.info_message = "No overconstraint detected";
        return result;
    }
    
    // Выбор подмножества узлов для точной интерполяции
    std::vector<int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return interp_coords[a] < interp_coords[b];
    });
    
    std::vector<int> selected;
    double step = static_cast<double>(m - 1) / max_nodes;
    for (int i = 0; i < max_nodes; ++i) {
        int idx = static_cast<int>(i * step + 0.5);
        selected.push_back(indices[std::min(idx, m - 1)]);
    }
    
    bool has_left = false, has_right = false;
    for (int idx : selected) {
        if (idx == 0) has_left = true;
        if (idx == m - 1) has_right = true;
    }
    if (!has_left && m > 0) selected.push_back(0);
    if (!has_right && m > 0 && selected.size() < static_cast<size_t>(max_nodes)) {
        selected.push_back(m - 1);
    }
    
    if (selected.size() > static_cast<size_t>(max_nodes)) {
        selected.resize(max_nodes);
    }
    
    result.selected_indices = selected;
    result.adapted_m = static_cast<int>(selected.size());
    result.resolved = true;
    
    std::ostringstream oss;
    oss << "INFO: Overconstrained system adapted (m = " << m << " -> " << result.adapted_m << ").\n";
    oss << "       Selected " << result.adapted_m << " nodes for exact interpolation.\n";
    oss << "       Remaining " << (m - result.adapted_m) << " nodes converted to approximation points.\n";
    result.info_message = oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.5: Обработка кратных/близких корней ==============

CloseNodesResult EdgeCaseHandler::handle_close_nodes(const std::vector<double>& coords,
                                                      const std::vector<double>& values,
                                                      double interval_length) {
    CloseNodesResult result;
    
    result.original_m = static_cast<int>(coords.size());
    
    if (coords.empty()) {
        result.has_close_nodes = false;
        result.effective_m = 0;
        result.info_message = "No nodes to process";
        return result;
    }
    
    // Вычисляем минимальное расстояние между узлами
    result.min_distance = std::numeric_limits<double>::max();
    for (int i = 1; i < result.original_m; ++i) {
        double dist = std::abs(coords[i] - coords[i-1]);
        result.min_distance = std::min(result.min_distance, dist);
    }
    
    // Определяем порог близости
    double close_threshold = epsilon_close_nodes * interval_length;
    
    if (result.min_distance >= close_threshold) {
        result.has_close_nodes = false;
        result.effective_m = result.original_m;
        result.info_message = "No close nodes detected";
        return result;
    }
    
    result.has_close_nodes = true;
    
    // Объединяем близкие узлы в кластеры
    result.clusters = cluster_close_nodes(coords, values, close_threshold);
    
    double value_threshold = epsilon_value_spread * std::max(1.0, 
        *std::max_element(values.begin(), values.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); }));
    
    int merged_count = 0;
    for (auto& cluster : result.clusters) {
        if (should_merge_cluster(cluster, value_threshold)) {
            cluster.should_merge = true;
            merged_count++;
        }
    }
    
    result.effective_m = result.original_m;
    for (const auto& cluster : result.clusters) {
        if (cluster.should_merge) {
            result.effective_m -= (static_cast<int>(cluster.indices.size()) - 1);
        }
    }
    
    std::ostringstream oss;
    oss << "INFO: Close nodes detected (min distance = " << std::scientific 
        << std::setprecision(2) << (result.min_distance / interval_length) << " of interval).\n";
    oss << "      " << merged_count << " cluster(s) with value spread below threshold.\n";
    oss << "      Effective node count reduced from " << result.original_m 
        << " to " << result.effective_m << ".\n";
    result.info_message = oss.str();
    
    return result;
}

// ============== Шаг 2.1.8.6: Обработка высокой степени ==============

HighDegreeResult EdgeCaseHandler::handle_high_degree(int n) {
    HighDegreeResult result;
    
    result.original_degree = n;
    
    if (n <= high_degree_threshold) {
        result.requires_adaptation = false;
        result.switch_to_chebyshev = false;
        result.use_long_double = false;
        result.suggest_splines = false;
        result.recommended_degree = n;
        return result;
    }
    
    result.requires_adaptation = true;
    
    if (n > 15) {
        result.switch_to_chebyshev = true;
    }
    
    if (n > 40) {
        result.suggest_splines = true;
    }
    
    result.recommended_degree = std::min(n, 30);
    result.recommendations = generate_high_degree_recommendations(n);
    
    return result;
}

// ============== Шаг 2.1.8.7: Анализ вырожденных данных ==============

DegeneracyResult EdgeCaseHandler::analyze_degeneracy(const std::vector<double>& values) {
    DegeneracyResult result;
    
    if (values.empty()) {
        result.is_degenerate = false;
        result.type = DegeneracyType::NONE;
        result.info_message = "Empty values - no degeneracy";
        return result;
    }
    
    // Проверка на константные значения
    if (is_constant(values, epsilon_machine)) {
        result.is_degenerate = true;
        result.type = DegeneracyType::CONSTANT;
        result.constant_value = values[0];
        result.effective_degree = 0;
        
        std::ostringstream oss;
        oss << "INFO: Constant interpolation values detected (value = " << values[0] << ").\n";
        oss << "      Basis polynomial simplifies to constant.\n";
        oss << "      First and second derivatives are zero.\n";
        result.info_message = oss.str();
        return result;
    }
    
    // Проверка на линейную зависимость
    if (values.size() >= 3) {
        double slope1 = (values[1] - values[0]);
        double slope2 = (values[2] - values[1]);
        
        if (std::abs(slope1 - slope2) < 1e-6) {
            result.is_degenerate = true;
            result.type = DegeneracyType::LINEAR;
            result.effective_degree = 1;
            
            std::ostringstream oss;
            oss << "INFO: Linear dependence detected in interpolation values.\n";
            oss << "      Effective degree reduced to 1.\n";
            result.info_message = oss.str();
            return result;
        }
    }
    
    result.is_degenerate = false;
    result.type = DegeneracyType::NONE;
    result.info_message = "No degeneracy detected";
    return result;
}

} // namespace mixed_approx
