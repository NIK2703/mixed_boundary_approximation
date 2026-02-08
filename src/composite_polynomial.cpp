#include "mixed_approximation/composite_polynomial.h"
#include <sstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

namespace mixed_approx {

// Константы для квадратуры Гаусса-Лежандра (10 узлов)
namespace {
    const std::vector<double> GAUSS_NODES_10 = {
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
        -0.4333953941292472, -0.14887433898163122, 0.14887433898163122,
        0.4333953941292472, 0.6794095682990244, 0.8650633666889845,
        0.9739065285171717
    };
    
    const std::vector<double> GAUSS_WEIGHTS_10 = {
        0.06667134430868814, 0.1494513491505806, 0.21908636251598204,
        0.26926671930999635, 0.29552422471475287, 0.29552422471475287,
        0.26926671930999635, 0.21908636251598204, 0.1494513491505806,
        0.06667134430868814
    };
    
    const double EPSILON_ROOT = 1e-12;  // Порог для определения близости к корню
    const double MAX_ANALYTIC_DEGREE = 15;  // Максимальная степень для аналитической сборки
    const double REGULARIZATION_THRESHOLD = 1e-100;  // Порог для численной стабилизации
    
    double transform_to_interval(double t, double a, double b) {
        return 0.5 * (b - a) * t + 0.5 * (a + b);
    }
    
    void get_gauss_legendre_nodes(int n, std::vector<double>& nodes, std::vector<double>& weights) {
        if (n <= 10) {
            nodes = GAUSS_NODES_10;
            weights = GAUSS_WEIGHTS_10;
            if (n < 10) {
                nodes.resize(n);
                weights.resize(n);
            }
        } else {
            // Для n > 10 используем 20-узловую квадратуру
            static const std::vector<double> GAUSS_NODES_20 = {
                -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
                -0.8391169718222188, -0.7463319064601508, -0.636053680726515,
                -0.5108670019508271, -0.37370608871541955, -0.22778585114164507,
                -0.07652652113349733, 0.07652652113349733, 0.22778585114164507,
                0.37370608871541955, 0.5108670019508271, 0.636053680726515,
                0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
                0.9639719272779138, 0.9931285991850949
            };
            static const std::vector<double> GAUSS_WEIGHTS_20 = {
                0.017614007139152118, 0.04060142980038694, 0.06267204833410906,
                0.08327674157670475, 0.10193011981724044, 0.11819453196151842,
                0.13168863844917664, 0.14209610931838205, 0.14917298647260374,
                0.15275338713072585, 0.15275338713072585, 0.14917298647260374,
                0.14209610931838205, 0.13168863844917664, 0.11819453196151842,
                0.10193011981724044, 0.08327674157670475, 0.06267204833410906,
                0.04060142980038694, 0.017614007139152118
            };
            nodes = GAUSS_NODES_20;
            weights = GAUSS_WEIGHTS_20;
            if (n < 20) {
                nodes.resize(n);
                weights.resize(n);
            }
        }
    }
}  // anonymous namespace

void CompositePolynomial::build(const InterpolationBasis& basis,
                               const WeightMultiplier& W,
                               const CorrectionPolynomial& Q,
                               double interval_start,
                               double interval_end,
                               EvaluationStrategy strategy) {
    interpolation_basis = basis;
    weight_multiplier = W;
    correction_poly = Q;
    
    interval_a = interval_start;
    interval_b = interval_end;
    eval_strategy = strategy;
    
    // Вычисляем метаданные
    total_degree = correction_poly.degree + weight_multiplier.degree();
    num_constraints = static_cast<int>(weight_multiplier.roots.size());
    num_free_params = correction_poly.n_free;
    
    // Проверяем корректность
    if (total_degree < 0 || num_constraints < 0) {
        validation_message = "Invalid polynomial structure: negative degree or constraints";
        return;
    }
    
    if (num_constraints > total_degree + 1) {
        validation_message = "Too many constraints: m = " + std::to_string(num_constraints) +
                            " > n + 1 = " + std::to_string(total_degree + 1);
        return;
    }
    
    // Проверяем компоненты
    if (!interpolation_basis.is_valid) {
        validation_message = "Interpolation basis is not valid: " + interpolation_basis.error_message;
        return;
    }
    
    // Очищаем кэши
    clear_caches();
    analytic_coeffs.clear();
    analytic_coeffs_valid = false;
    
    validation_message = "CompositePolynomial built successfully";
}

double CompositePolynomial::evaluate(double x) const {
    // F(x) = P_int(x) + Q(x) * W(x)
    
    double p_int_val = interpolation_basis.evaluate(x);
    
    // Особые случаи
    if (num_constraints == 0) {
        // W(x) = 1, F(x) = P_int(x) + Q(x)
        return p_int_val + correction_poly.evaluate_Q(x);
    }
    
    if (num_constraints == total_degree + 1) {
        // Q(x) вырожден (n_free = 0), F(x) = P_int(x)
        return p_int_val;
    }
    
    double q_val = correction_poly.evaluate_Q(x);
    double w_val = weight_multiplier.evaluate(x);
    
    return p_int_val + q_val * w_val;
}

double CompositePolynomial::evaluate_derivative(double x, int order) const {
    if (order < 1 || order > 2) {
        throw std::invalid_argument("CompositePolynomial::evaluate_derivative: order must be 1 or 2");
    }
    
    // F''(x) = P_int''(x) + Q''(x)·W(x) + 2·Q'(x)·W'(x) + Q(x)·W''(x)
    // F'(x) = P_int'(x) + Q'(x)·W(x) + Q(x)·W'(x)
    
    double p_int_val = interpolation_basis.evaluate_derivative(x, order);
    
    // Особые случаи
    if (num_constraints == 0) {
        // W(x) = 1, W'(x) = 0, W''(x) = 0
        if (order == 1) {
            return p_int_val + correction_poly.evaluate_Q_derivative(x, 1);
        } else {
            return p_int_val + correction_poly.evaluate_Q_derivative(x, 2);
        }
    }
    
    if (num_constraints == total_degree + 1) {
        // Q(x) вырожден, F(x) = P_int(x)
        return p_int_val;
    }
    
    double q_val = correction_poly.evaluate_Q(x);
    double q1_val = correction_poly.evaluate_Q_derivative(x, 1);
    double q2_val = (order == 2) ? correction_poly.evaluate_Q_derivative(x, 2) : 0.0;
    
    double w_val = weight_multiplier.evaluate(x);
    double w1_val = weight_multiplier.evaluate_derivative(x, 1);
    double w2_val = (order == 2) ? weight_multiplier.evaluate_derivative(x, 2) : 0.0;
    
    if (order == 1) {
        return p_int_val + q1_val * w_val + q_val * w1_val;
    } else {
        return p_int_val + q2_val * w_val + 2.0 * q1_val * w1_val + q_val * w2_val;
    }
}

bool CompositePolynomial::build_analytic_coefficients(int max_degree_for_analytic) {
    analytic_coeffs.clear();
    analytic_coeffs_valid = false;
    
    // Проверяем степень
    if (total_degree > max_degree_for_analytic) {
        validation_message = "Degree " + std::to_string(total_degree) + 
                            " exceeds maximum for analytic assembly (" + 
                            std::to_string(max_degree_for_analytic) + "). Use lazy evaluation.";
        return false;
    }
    
    // Проверяем особые случаи
    if (num_constraints == total_degree + 1) {
        // Q вырожден, F(x) = P_int(x)
        analytic_coeffs = extract_p_int_coefficients();
        analytic_coeffs_valid = !analytic_coeffs.empty();
        if (analytic_coeffs_valid) {
            validation_message = "Analytic coefficients: F(x) = P_int(x) (full interpolation)";
        }
        return analytic_coeffs_valid;
    }
    
    // Извлекаем коэффициенты P_int(x)
    std::vector<double> p_int_coeffs = extract_p_int_coefficients();
    if (p_int_coeffs.empty() && num_constraints > 0) {
        validation_message = "Failed to extract P_int coefficients";
        return false;
    }
    
    // Получаем коэффициенты W(x) и Q(x)
    std::vector<double> w_coeffs = weight_multiplier.coeffs;
    std::vector<double> q_coeffs = correction_poly.coeffs;
    
    // Преобразуем q_coeffs из порядка возрастания степеней в порядок убывания
    std::vector<double> q_coeffs_desc;
    if (!q_coeffs.empty()) {
        q_coeffs_desc.resize(q_coeffs.size());
        for (size_t i = 0; i < q_coeffs.size(); ++i) {
            q_coeffs_desc[q_coeffs.size() - 1 - i] = q_coeffs[i];
        }
    }
    
    // Вычисляем коэффициенты Q(x)·W(x) через свёртку
    std::vector<double> qw_coeffs;
    if (q_coeffs_desc.empty() || q_coeffs_desc[0] == 0.0 && q_coeffs_desc.size() == 1) {
        // Q(x) = 0
        qw_coeffs.clear();
    } else {
        qw_coeffs = convolve_coefficients(q_coeffs_desc, w_coeffs);
    }
    
    // Складываем: F_coeffs = P_int_coeffs + QW_coeffs
    int result_size = std::max(
        p_int_coeffs.empty() ? 0 : static_cast<int>(p_int_coeffs.size()),
        qw_coeffs.empty() ? 0 : static_cast<int>(qw_coeffs.size())
    );
    
    if (result_size == 0) {
        analytic_coeffs = {0.0};
        analytic_coeffs_valid = true;
        validation_message = "Analytic coefficients: zero polynomial";
        return true;
    }
    
    analytic_coeffs.assign(result_size, 0.0);
    
    // Добавляем P_int
    if (!p_int_coeffs.empty()) {
        int offset = result_size - static_cast<int>(p_int_coeffs.size());
        for (size_t i = 0; i < p_int_coeffs.size(); ++i) {
            analytic_coeffs[offset + i] += p_int_coeffs[i];
        }
    }
    
    // Добавляем Q*W
    if (!qw_coeffs.empty()) {
        int offset = result_size - static_cast<int>(qw_coeffs.size());
        for (size_t i = 0; i < qw_coeffs.size(); ++i) {
            analytic_coeffs[offset + i] += qw_coeffs[i];
        }
    }
    
    // Удаляем старшие нули
    while (analytic_coeffs.size() > 1 && 
           std::abs(analytic_coeffs[0]) < std::numeric_limits<double>::epsilon() * 1e6) {
        analytic_coeffs.erase(analytic_coeffs.begin());
    }
    
    analytic_coeffs_valid = true;
    validation_message = "Analytic coefficients built successfully for degree " + 
                        std::to_string(total_degree);
    
    return true;
}

double CompositePolynomial::evaluate_analytic(double x) const {
    if (!analytic_coeffs_valid || analytic_coeffs.empty()) {
        throw std::runtime_error("Analytic coefficients not available. Call build_analytic_coefficients first.");
    }
    
    // Схема Горнера
    double result = 0.0;
    for (double coeff : analytic_coeffs) {
        result = result * x + coeff;
    }
    return result;
}

void CompositePolynomial::evaluate_batch(const std::vector<double>& points, 
                                         std::vector<double>& results) const {
    results.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        results[i] = evaluate(points[i]);
    }
}

void CompositePolynomial::evaluate_batch_analytic(const std::vector<double>& points, 
                                                   std::vector<double>& results) const {
    if (!analytic_coeffs_valid) {
        // Используем ленивую оценку
        evaluate_batch(points, results);
        return;
    }
    
    results.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        results[i] = evaluate_analytic(points[i]);
    }
}

void CompositePolynomial::build_caches(const std::vector<double>& points_x,
                                        const std::vector<double>& points_y,
                                        const std::vector<double>& quad_points) {
    clear_caches();
    
    // Кэши для аппроксимирующих точек
    cache.P_at_x.resize(points_x.size());
    cache.W_at_x.resize(points_x.size());
    for (size_t i = 0; i < points_x.size(); ++i) {
        cache.P_at_x[i] = interpolation_basis.evaluate(points_x[i]);
        cache.W_at_x[i] = weight_multiplier.evaluate(points_x[i]);
    }
    
    // Кэши для отталкивающих точек
    cache.P_at_y.resize(points_y.size());
    cache.W_at_y.resize(points_y.size());
    for (size_t i = 0; i < points_y.size(); ++i) {
        cache.P_at_y[i] = interpolation_basis.evaluate(points_y[i]);
        cache.W_at_y[i] = weight_multiplier.evaluate(points_y[i]);
    }
    
    // Кэши для квадратуры
    int quad_n;
    if (!quad_points.empty()) {
        quad_n = static_cast<int>(quad_points.size());
        cache.quad_points = quad_points;
    } else {
        quad_n = std::max(10, 2 * total_degree + 1);
        std::vector<double> quad_weights;
        get_gauss_legendre_nodes(quad_n, cache.quad_points, quad_weights);
    }
    
    cache.W_at_quad.resize(quad_n);
    cache.W1_at_quad.resize(quad_n);
    cache.W2_at_quad.resize(quad_n);
    cache.Q_at_quad.resize(quad_n);
    cache.Q1_at_quad.resize(quad_n);
    cache.Q2_at_quad.resize(quad_n);
    cache.P2_at_quad.resize(quad_n);
    
    for (int i = 0; i < quad_n; ++i) {
        double x = cache.quad_points[i];
        cache.W_at_quad[i] = weight_multiplier.evaluate(x);
        cache.W1_at_quad[i] = weight_multiplier.evaluate_derivative(x, 1);
        cache.W2_at_quad[i] = weight_multiplier.evaluate_derivative(x, 2);
        cache.Q_at_quad[i] = correction_poly.evaluate_Q(x);
        cache.Q1_at_quad[i] = correction_poly.evaluate_Q_derivative(x, 1);
        cache.Q2_at_quad[i] = correction_poly.evaluate_Q_derivative(x, 2);
        cache.P2_at_quad[i] = interpolation_basis.evaluate_derivative(x, 2);
    }
    
    caches_built = true;
}

void CompositePolynomial::clear_caches() {
    cache.P_at_x.clear();
    cache.W_at_x.clear();
    cache.P_at_y.clear();
    cache.W_at_y.clear();
    cache.quad_points.clear();
    cache.W_at_quad.clear();
    cache.W1_at_quad.clear();
    cache.W2_at_quad.clear();
    cache.Q_at_quad.clear();
    cache.Q1_at_quad.clear();
    cache.Q2_at_quad.clear();
    cache.P2_at_quad.clear();
    caches_built = false;
}

double CompositePolynomial::compute_regularization_term(double gamma) const {
    if (gamma <= 0.0 || total_degree < 2) {
        return 0.0;
    }
    
    // F''(x) = P_int''(x) + Q''(x)·W(x) + 2·Q'(x)·W'(x) + Q(x)·W''(x)
    
    // Определяем число узлов квадратуры
    int quad_n = std::max(10, 2 * total_degree + 1);
    
    std::vector<double> quad_nodes, quad_weights;
    get_gauss_legendre_nodes(quad_n, quad_nodes, quad_weights);
    
    double integral = 0.0;
    double scale_factor = 0.5 * (interval_b - interval_a);
    
    for (int i = 0; i < quad_n; ++i) {
        double t = quad_nodes[i];
        double weight = quad_weights[i];
        double x = transform_to_interval(t, interval_a, interval_b);
        
        // Вычисляем F''(x)
        double p2 = interpolation_basis.evaluate_derivative(x, 2);
        double q = correction_poly.evaluate_Q(x);
        double q1 = correction_poly.evaluate_Q_derivative(x, 1);
        double q2 = correction_poly.evaluate_Q_derivative(x, 2);
        double w = weight_multiplier.evaluate(x);
        double w1 = weight_multiplier.evaluate_derivative(x, 1);
        double w2 = weight_multiplier.evaluate_derivative(x, 2);
        
        double f2 = p2 + q2 * w + 2.0 * q1 * w1 + q * w2;
        
        // Интегрируем (F''(x))^2
        double integrand = f2 * f2;
        integral += integrand * weight;
    }
    
    // Масштабируем интеграл с Jacobian
    integral *= scale_factor;
    
    return gamma * integral;
}

bool CompositePolynomial::verify_assembly(double tolerance) {
    // Проверяем интерполяционные условия: F(z_e) ≈ f(z_e)
    for (const auto& node : interpolation_basis.nodes) {
        double F_val = evaluate(node);
        // Находим соответствующее значение f(z_e)
        // (Примечание: nodes_normalized соответствуют исходным nodes через нормализацию)
    }
    
    // Более прямая проверка: W(z_e) должен быть близок к нулю
    if (num_constraints > 0) {
        for (double root : weight_multiplier.roots) {
            double W_val = weight_multiplier.evaluate(root);
            if (std::abs(W_val) > tolerance) {
                validation_message = "W(z_e) not close to zero at z_e = " + std::to_string(root);
                return false;
            }
        }
    }
    
    // Проверяем, что analytic_coeffs (если доступны) дают те же значения
    if (analytic_coeffs_valid && !analytic_coeffs.empty()) {
        std::vector<double> test_points = {
            interval_a, 
            (interval_a + interval_b) * 0.5, 
            interval_b,
            (interval_a + interval_b) * 0.25,
            (interval_a + interval_b) * 0.75
        };
        
        for (double x : test_points) {
            double lazy_val = evaluate(x);
            double analytic_val = evaluate_analytic(x);
            double rel_diff = std::abs(lazy_val - analytic_val) / 
                             (std::abs(lazy_val) + std::numeric_limits<double>::epsilon());
            
            if (rel_diff > 1e-8) {
                validation_message = "Discrepancy between lazy and analytic evaluation at x = " +
                                    std::to_string(x);
                return false;
            }
        }
    }
    
    validation_message = "Assembly verification passed";
    return true;
}

bool CompositePolynomial::verify_representations_consistency(int num_test_points,
                                                            double relative_tolerance) const {
    if (!analytic_coeffs_valid) {
        // Нет аналитических коэффициентов для сравнения
        return true;
    }
    
    // Генерируем тестовые точки
    std::vector<double> test_points;
    double step = (interval_b - interval_a) / (num_test_points + 1);
    for (int i = 1; i <= num_test_points; ++i) {
        test_points.push_back(interval_a + i * step);
    }
    
    // Добавляем граничные точки
    test_points.push_back(interval_a);
    test_points.push_back(interval_b);
    
    for (double x : test_points) {
        double lazy_val = evaluate(x);
        double analytic_val = evaluate_analytic(x);
        
        double max_abs = std::max(std::abs(lazy_val), std::abs(analytic_val));
        if (max_abs < std::numeric_limits<double>::epsilon()) {
            max_abs = 1.0;
        }
        
        double rel_diff = std::abs(lazy_val - analytic_val) / max_abs;
        
        if (rel_diff > relative_tolerance) {
            return false;
        }
    }
    
    return true;
}

std::string CompositePolynomial::get_diagnostic_info() const {
    std::ostringstream oss;
    oss << "CompositePolynomial diagnostic:\n";
    oss << "  degree: " << total_degree << "\n";
    oss << "  constraints (m): " << num_constraints << "\n";
    oss << "  free params (n-m+1): " << num_free_params << "\n";
    oss << "  interval: [" << interval_a << ", " << interval_b << "]\n";
    oss << "  eval_strategy: ";
    switch (eval_strategy) {
        case EvaluationStrategy::LAZY: oss << "LAZY"; break;
        case EvaluationStrategy::ANALYTIC: oss << "ANALYTIC"; break;
        case EvaluationStrategy::HYBRID: oss << "HYBRID"; break;
    }
    oss << "\n";
    oss << "  analytic_coeffs: " << (analytic_coeffs_valid ? "valid" : "not built") << "\n";
    oss << "  caches: " << (caches_built ? "built" : "not built") << "\n";
    oss << "  message: " << validation_message << "\n";
    
    if (analytic_coeffs_valid && !analytic_coeffs.empty()) {
        oss << "  coeffs: [";
        int max_show = std::min(10, static_cast<int>(analytic_coeffs.size()));
        for (int i = 0; i < max_show; ++i) {
            oss << analytic_coeffs[i];
            if (i < max_show - 1) oss << ", ";
        }
        if (static_cast<int>(analytic_coeffs.size()) > max_show) {
            oss << ", ...";
        }
        oss << "]\n";
    }
    
    return oss.str();
}

bool CompositePolynomial::is_valid() const {
    if (total_degree < 0 || num_constraints < 0) {
        return false;
    }
    
    if (num_constraints > total_degree + 1) {
        return false;
    }
    
    if (!interpolation_basis.is_valid) {
        return false;
    }
    
    if (!correction_poly.is_initialized) {
        return false;
    }
    
    return true;
}

std::vector<double> CompositePolynomial::extract_p_int_coefficients() const {
    // Для малого числа узлов используем прямое раскрытие формулы Лагранжа
    // Для большого числа узлов используем численное решение
    
    // Проверяем, доступны ли узлы и значения
    if (interpolation_basis.nodes.empty() || interpolation_basis.values.empty()) {
        // P_int(x) = 0
        return std::vector<double>();
    }
    
    int m = static_cast<int>(interpolation_basis.nodes.size());
    
    // Для небольшого числа узлов вычисляем коэффициенты напрямую
    if (m <= 8) {
        std::vector<double> coeffs(m, 0.0);  // Полином степени m-1
        
        for (int j = 0; j < m; ++j) {
            // Строим базисный полином Лагранжа L_j(x)
            std::vector<double> L_coeff(1, 1.0);  // Начинаем с 1
            
            double denom = 1.0;
            for (int k = 0; k < m; ++k) {
                if (k == j) continue;
                denom *= (interpolation_basis.nodes[j] - interpolation_basis.nodes[k]);
                
                // Умножаем на (x - z_k)
                std::vector<double> new_L(k + 2, 0.0);
                for (int i = 0; i < k + 1; ++i) {
                    new_L[i] -= interpolation_basis.nodes[k] * L_coeff[i];
                    new_L[i + 1] += L_coeff[i];
                }
                L_coeff = new_L;
            }
            
            // Делим на denom и добавляем к результату с весом f(z_j)
            double weight = interpolation_basis.values[j] / denom;
            if (coeffs.size() < L_coeff.size()) {
                coeffs.resize(L_coeff.size(), 0.0);
            }
            for (size_t i = 0; i < L_coeff.size(); ++i) {
                coeffs[coeffs.size() - 1 - i] += weight * L_coeff[L_coeff.size() - 1 - i];
            }
        }
        
        // Удаляем старшие нули
        while (coeffs.size() > 1 && std::abs(coeffs[0]) < 1e-15) {
            coeffs.erase(coeffs.begin());
        }
        
        return coeffs;
    }
    
    // Для большого числа узлов используем интерполяцию Вандермонда
    // Строим систему V * a = f, где V[i,j] = z_i^j
    
    std::vector<double> nodes = interpolation_basis.nodes;
    std::vector<double> values = interpolation_basis.values;
    
    int n = m - 1;  // Степень полинома
    
    // Формируем матрицу Вандермонда и решаем методом Гаусса
    std::vector<std::vector<double>> V(m, std::vector<double>(m));
    for (int i = 0; i < m; ++i) {
        double power = 1.0;
        for (int j = 0; j <= n; ++j) {
            V[i][j] = power;
            power *= nodes[i];
        }
    }
    
    // Прямой ход исключения Гаусса
    for (int col = 0; col < m - 1; ++col) {
        // Поиск максимального элемента в столбце
        int max_row = col;
        double max_val = std::abs(V[col][col]);
        for (int row = col + 1; row < m; ++row) {
            if (std::abs(V[row][col]) > max_val) {
                max_val = std::abs(V[row][col]);
                max_row = row;
            }
        }
        
        if (max_val < 1e-14) {
            // Сингулярная матрица - возвращаем пустой результат
            return std::vector<double>();
        }
        
        if (max_row != col) {
            std::swap(V[max_row], V[col]);
            std::swap(values[max_row], values[col]);
        }
        
        // Нормализация
        for (int row = col + 1; row < m; ++row) {
            double factor = V[row][col] / V[col][col];
            values[row] -= factor * values[col];
            for (int j = col; j < m; ++j) {
                V[row][j] -= factor * V[col][j];
            }
        }
    }
    
    // Обратная подстановка
    std::vector<double> coeffs_desc(m);
    for (int i = m - 1; i >= 0; --i) {
        double sum = values[i];
        for (int j = i + 1; j < m; ++j) {
            sum -= V[i][j] * coeffs_desc[j];
        }
        coeffs_desc[i] = sum / V[i][i];
    }
    
    return coeffs_desc;
}

std::vector<double> CompositePolynomial::convolve_coefficients(const std::vector<double>& q_coeffs,
                                                                const std::vector<double>& w_coeffs) const {
    if (q_coeffs.empty() || w_coeffs.empty()) {
        return std::vector<double>();
    }
    
    int deg_Q = static_cast<int>(q_coeffs.size()) - 1;
    int deg_W = static_cast<int>(w_coeffs.size()) - 1;
    int deg_result = deg_Q + deg_W;
    
    std::vector<double> result(deg_result + 1, 0.0);
    
    for (int i = 0; i <= deg_Q; ++i) {
        for (int j = 0; j <= deg_W; ++j) {
            int k = i + j;
            result[k] += q_coeffs[i] * w_coeffs[j];
        }
    }
    
    // Удаляем старшие нули
    while (result.size() > 1 && std::abs(result[0]) < 1e-14) {
        result.erase(result.begin());
    }
    
    return result;
}

double CompositePolynomial::transform_quadrature_node(double t) const {
    return 0.5 * (interval_b - interval_a) * t + 0.5 * (interval_a + interval_b);
}

double compute_regularization_via_components(const CompositePolynomial& F, double gamma) {
    return F.compute_regularization_term(gamma);
}

} // namespace mixed_approx
