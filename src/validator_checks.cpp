#include "mixed_approximation/validator.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace mixed_approx {

// ==================== Вспомогательные функции ====================

/**
 * @brief Вычисление адаптивной погрешности для интервала
 */
static double compute_interval_epsilon(double a, double b) {
    double max_abs = std::max(std::abs(a), std::max(std::abs(b), 1.0));
    return 1e-12 * max_abs;
}

/**
 * @brief Вычисление адаптивной погрешности для точек
 */
static double compute_point_epsilon(double a, double b) {
    return 1e-9 * (b - a);
}

/**
 * @brief Вычисление адаптивного порога для сравнения координат
 */
static double compute_overlap_epsilon(double x, double y) {
    double max_abs = std::max(std::abs(x), std::max(std::abs(y), 1.0));
    return std::max(1e-12, 1e-9 * max_abs);
}

/**
 * @brief Проверка, является ли число конечным (не NaN, не Inf)
 */
static bool is_finite(double x) {
    return std::isfinite(x);
}

// ==================== Индивидуальные проверки ====================

std::string Validator::check_interval(const ApproximationConfig& config) {
    double a = config.interval_start;
    double b = config.interval_end;
    
    // Проверка конечности
    if (!is_finite(a)) {
        return "Interval start is not a finite number (found: " + std::to_string(a) + ")";
    }
    if (!is_finite(b)) {
        return "Interval end is not a finite number (found: " + std::to_string(b) + ")";
    }
    
    // Проверка a < b с адаптивной погрешностью
    double epsilon = compute_interval_epsilon(a, b);
    if (!(b - a > epsilon)) {
        return "Interval is degenerate or invalid: start (" + std::to_string(a) + 
               ") must be less than end (" + std::to_string(b) + 
               ") by at least " + std::to_string(epsilon);
    }
    
    return "";
}

std::string Validator::check_points_in_interval(const ApproximationConfig& config) {
    double a = config.interval_start;
    double b = config.interval_end;
    double epsilon = compute_point_epsilon(a, b);
    
    // Проверка аппроксимирующих точек
    for (const auto& point : config.approx_points) {
        if (!is_finite(point.x)) {
            return "Approximation point has non-finite x-coordinate: " + std::to_string(point.x);
        }
        if (point.x < a - epsilon) {
            return "Approximation point x = " + std::to_string(point.x) + 
                   " is below interval start (" + std::to_string(a) + ")";
        }
        if (point.x > b + epsilon) {
            return "Approximation point x = " + std::to_string(point.x) + 
                   " is above interval end (" + std::to_string(b) + ")";
        }
    }
    
    // Проверка отталкивающих точек
    for (const auto& point : config.repel_points) {
        if (!is_finite(point.x)) {
            return "Repel point has non-finite x-coordinate: " + std::to_string(point.x);
        }
        if (point.x < a - epsilon) {
            return "Repel point x = " + std::to_string(point.x) + 
                   " is below interval start (" + std::to_string(a) + ")";
        }
        if (point.x > b + epsilon) {
            return "Repel point x = " + std::to_string(point.x) + 
                   " is above interval end (" + std::to_string(b) + ")";
        }
    }
    
    // Проверка интерполяционных узлов
    for (const auto& node : config.interp_nodes) {
        if (!is_finite(node.x)) {
            return "Interpolation node has non-finite x-coordinate: " + std::to_string(node.x);
        }
        if (node.x < a - epsilon) {
            return "Interpolation node x = " + std::to_string(node.x) + 
                   " is below interval start (" + std::to_string(a) + ")";
        }
        if (node.x > b + epsilon) {
            return "Interpolation node x = " + std::to_string(node.x) + 
                   " is above interval end (" + std::to_string(b) + ")";
        }
    }
    
    return "";
}

std::string Validator::check_disjoint_sets(const ApproximationConfig& config) {
    // Собираем все точки с метками
    struct PointInfo {
        double x;
        int type;  // 0 = approx, 1 = repel, 2 = interp
        size_t index;
    };
    
    std::vector<PointInfo> all_points;
    
    for (size_t i = 0; i < config.approx_points.size(); ++i) {
        all_points.push_back({config.approx_points[i].x, 0, i});
    }
    for (size_t i = 0; i < config.repel_points.size(); ++i) {
        all_points.push_back({config.repel_points[i].x, 1, i});
    }
    for (size_t i = 0; i < config.interp_nodes.size(); ++i) {
        all_points.push_back({config.interp_nodes[i].x, 2, i});
    }
    
    if (all_points.empty()) {
        return "";  // Пустое множество не проверяем здесь
    }
    
    // Сортируем по координате (O(N log N))
    std::sort(all_points.begin(), all_points.end(), 
              [](const PointInfo& a, const PointInfo& b) { return a.x < b.x; });
    
    // Проверяем соседние элементы
    std::vector<std::string> conflicts;
    
    for (size_t i = 1; i < all_points.size(); ++i) {
        const auto& p1 = all_points[i-1];
        const auto& p2 = all_points[i];
        
        double epsilon = compute_overlap_epsilon(p1.x, p2.x);
        
        if (std::abs(p1.x - p2.x) < epsilon) {
            // Найдено пересечение
            std::ostringstream oss;
            oss << "Points overlap at approximately x = " << p1.x << ": ";
            
            bool fatal = false;
            
            // Определяем типы точек
            if ((p1.type == 0 && p2.type == 2) || (p1.type == 2 && p2.type == 0)) {
                oss << "approximation point and interpolation node";
                fatal = true;
            } else if ((p1.type == 1 && p2.type == 2) || (p1.type == 2 && p2.type == 1)) {
                oss << "repel point and interpolation node";
                fatal = true;
            } else if (p1.type != p2.type) {
                oss << "approximation point and repel point";
                // Не фатально, но предупреждение
            } else {
                // Две точки одного типа (должны были быть обработаны в check_unique_*)
                continue;
            }
            
            oss << " (indices: " << p1.index << " and " << p2.index << ")";
            
            if (fatal) {
                oss << " [FATAL]";
            }
            
            conflicts.push_back(oss.str());
        }
    }
    
    if (!conflicts.empty()) {
        // Проверяем, есть ли фатальные конфликты
        bool has_fatal = false;
        for (const auto& c : conflicts) {
            if (c.find("[FATAL]") != std::string::npos) {
                has_fatal = true;
                break;
            }
        }
        
        std::ostringstream result;
        result << "Point set conflicts detected:\n";
        for (const auto& c : conflicts) {
            result << "  - " << c << "\n";
        }
        
        if (has_fatal) {
            result << "FATAL: Interpolation nodes must not coincide with approximation or repel points.";
        }
        
        return result.str();
    }
    
    return "";
}

std::string Validator::check_positive_weights(const ApproximationConfig& config) {
    const double epsilon_weight = 1e-15;
    std::ostringstream errors;
    bool has_error = false;
    
    // Проверка весов аппроксимирующих точек
    for (const auto& point : config.approx_points) {
        if (!is_finite(point.weight)) {
            errors << "Approximation point weight is not finite (x=" << point.x << ": " << point.weight << ")\n";
            has_error = true;
        } else if (point.weight <= epsilon_weight) {
            errors << "Approximation point weight must be positive (x=" << point.x << 
                      ", found: " << point.weight << ")\n";
            has_error = true;
        }
    }
    
    // Проверка весов отталкивающих точек
    for (const auto& point : config.repel_points) {
        if (!is_finite(point.weight)) {
            errors << "Repel point weight is not finite (x=" << point.x << ": " << point.weight << ")\n";
            has_error = true;
        } else if (point.weight <= epsilon_weight) {
            errors << "Repel point weight must be positive (x=" << point.x << 
                      ", found: " << point.weight << ")\n";
            has_error = true;
        }
    }
    
    // Проверка gamma
    if (!is_finite(config.gamma)) {
        errors << "Gamma is not a finite number (found: " << config.gamma << ")\n";
        has_error = true;
    } else if (config.gamma < 0.0) {
        errors << "Gamma must be non-negative (found: " << config.gamma << ")\n";
        has_error = true;
    }
    
    if (has_error) {
        return errors.str();
    }
    
    return "";
}

std::string Validator::check_interpolation_nodes_count(const ApproximationConfig& config) {
    int m = static_cast<int>(config.interp_nodes.size());
    int n = config.polynomial_degree;
    
    if (m > n + 1) {
        return "Number of interpolation nodes (" + std::to_string(m) + 
               ") exceeds polynomial degree + 1 (" + std::to_string(n + 1) + ")";
    }
    
    // Предупреждение при m == n+1
    if (m == n + 1 && m > 0) {
        return "WARNING: Number of interpolation nodes equals polynomial degree + 1. "
               "The polynomial is fully determined by interpolation conditions, "
               "making approximation and repulsion criteria ineffective.";
    }
    
    return "";
}

std::string Validator::check_unique_interpolation_nodes(const ApproximationConfig& config) {
    if (config.interp_nodes.size() <= 1) {
        return "";  // 0 или 1 узел - дубликатов быть не может
    }
    
    std::vector<double> x_values;
    for (const auto& node : config.interp_nodes) {
        if (!is_finite(node.x)) {
            return "Interpolation node has non-finite x-coordinate: " + std::to_string(node.x);
        }
        x_values.push_back(node.x);
    }
    
    std::sort(x_values.begin(), x_values.end());
    
    for (size_t i = 1; i < x_values.size(); ++i) {
        double epsilon = compute_overlap_epsilon(x_values[i-1], x_values[i]);
        if (std::abs(x_values[i] - x_values[i-1]) < epsilon) {
            return "Interpolation nodes must be distinct (duplicate near x = " + 
                   std::to_string(x_values[i]) + ")";
        }
    }
    
    return "";
}

std::string Validator::check_nonempty_sets(const ApproximationConfig& config) {
    bool has_approx = !config.approx_points.empty();
    bool has_interp = !config.interp_nodes.empty();
    
    if (!has_approx && !has_interp) {
        return "At least one of approximation points or interpolation nodes must be non-empty. "
               "Both sets are empty.";
    }
    
    return "";
}

std::string Validator::check_numerical_anomalies(const ApproximationConfig& config) {
    std::ostringstream warnings;
    bool has_warning = false;
    
    // Проверка весов аппроксимирующих точек на разброс
    if (config.approx_points.size() >= 2) {
        double min_w = config.approx_points[0].weight;
        double max_w = config.approx_points[0].weight;
        for (const auto& p : config.approx_points) {
            min_w = std::min(min_w, p.weight);
            max_w = std::max(max_w, p.weight);
        }
        
        if (max_w / min_w >= 1e12) {
            warnings << "Approximation weights have extreme ratio (max/min = " << 
                        std::scientific << std::setprecision(3) << (max_w / min_w) << 
                        std::defaultfloat << "). This may cause numerical instability.\n";
            has_warning = true;
        }
    }
    
    // Проверка весов отталкивающих точек
    for (const auto& point : config.repel_points) {
        if (point.weight > 1e8) {
            warnings << "Repel point at x=" << point.x << 
                         " has very large weight (" << point.weight << 
                         "). This may create a very stiff barrier and cause numerical difficulties.\n";
            has_warning = true;
        } else if (point.weight < 1e-6) {
            warnings << "Repel point at x=" << point.x << 
                         " has very small weight (" << point.weight << 
                         "). The repulsion effect may be negligible.\n";
            has_warning = true;
        }
    }
    
    // Проверка gamma
    if (config.gamma == 0.0) {
        warnings << "Gamma is zero. This may lead to oscillations for high-degree polynomials (Runge phenomenon).\n";
        has_warning = true;
    } else if (config.gamma > 1e6) {
        warnings << "Gamma is very large (" << config.gamma << 
                         "). This may cause excessive smoothing and loss of approximation accuracy.\n";
        has_warning = true;
    }
    
    // Проверка на очень большие/малые координаты
    double a = config.interval_start;
    double b = config.interval_end;
    double interval_size = b - a;
    
    if (interval_size > 1e10 || interval_size < 1e-10) {
        warnings << "Interval size is " << interval_size << 
                         ", which may cause numerical issues. Consider rescaling.\n";
        has_warning = true;
    }
    
    if (has_warning) {
        return warnings.str();
    }
    
    return "";
}

std::string Validator::check_repel_interp_value_conflict(const ApproximationConfig& config) {
    if (config.repel_points.empty() || config.interp_nodes.empty()) {
        return "";
    }
    
    double a = config.interval_start;
    double b = config.interval_end;
    double epsilon_coord = compute_point_epsilon(a, b);  // 1e-9 * (b - a)
    
    for (const auto& repel : config.repel_points) {
        for (const auto& interp : config.interp_nodes) {
            if (std::abs(repel.x - interp.x) < epsilon_coord) {
                // Координаты близки, проверяем значения
                double epsilon_value = 1e-9 * std::max(std::abs(interp.value), std::max(std::abs(repel.y_forbidden), 1.0));
                if (std::abs(repel.y_forbidden - interp.value) < epsilon_value) {
                    std::ostringstream oss;
                    oss << "FATAL: Repel point at x=" << repel.x << 
                           " has y_forbidden=" << repel.y_forbidden <<
                           " which conflicts with interpolation node (x=" << interp.x <<
                           ", f=" << interp.value << "). " <<
                           "Interpolation requires F(x) = " << interp.value <<
                           " but repulsion forbids this value.";
                    return oss.str();
                }
            }
        }
    }
    
    return "";
}

bool Validator::contains_point(const std::vector<double>& points, double x, double tolerance) {
    for (double p : points) {
        if (std::abs(p - x) < tolerance) {
            return true;
        }
    }
    return false;
}

} // namespace mixed_approx
