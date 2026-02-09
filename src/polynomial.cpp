#include "mixed_approximation/polynomial.h"
#include "mixed_approximation/types.h"
#include <stdexcept>
#include <algorithm>

namespace mixed_approx {

Polynomial::Polynomial()
    : coeffs_(), degree_(-1) {
}

Polynomial::Polynomial(const std::vector<double>& coeffs) 
    : coeffs_(coeffs) {
    // Удаляем ведущие нули
    while (coeffs_.size() > 1 && std::abs(coeffs_.front()) < 1e-14) {
        coeffs_.erase(coeffs_.begin());
    }
    degree_ = static_cast<int>(coeffs_.size()) - 1;
}

Polynomial::Polynomial(int degree) 
    : degree_(degree) {
    coeffs_.assign(degree + 1, 0.0);
}

double Polynomial::evaluate(double x) const {
    // Схема Горнера для численной устойчивости
    double result = 0.0;
    for (double coeff : coeffs_) {
        result = result * x + coeff;
    }
    return result;
}

double Polynomial::derivative(double x) const {
    if (degree_ == 0) return 0.0;
    
    // P'(x) = Σ_{k=0}^{n-1} (k+1) * a_{n-k} * x^k
    // Но у нас coeffs_[i] соответствует a_{n-i}
    double result = 0.0;
    int n = degree_;
    for (int i = 0; i < n; ++i) {
        int power = n - i;  // степень для coeffs_[i]
        result = result * x + power * coeffs_[i];
    }
    return result;
}

double Polynomial::second_derivative(double x) const {
    if (degree_ < 2) return 0.0;
    
    // P''(x) = Σ_{k=0}^{n-2} (k+2)*(k+1) * a_{n-k} * x^k
    double result = 0.0;
    int n = degree_;
    for (int i = 0; i < n - 1; ++i) {
        int power = n - i;  // степень для coeffs_[i]
        double coeff = power * (power - 1) * coeffs_[i];
        result = result * x + coeff;
    }
    return result;
}

void Polynomial::setCoefficients(const std::vector<double>& coeffs) {
    coeffs_ = coeffs;
    // Удаляем ведущие нули
    while (coeffs_.size() > 1 && std::abs(coeffs_.front()) < 1e-14) {
        coeffs_.erase(coeffs_.begin());
    }
    degree_ = static_cast<int>(coeffs_.size()) - 1;
}

Polynomial Polynomial::operator+(const Polynomial& other) const {
    size_t max_size = std::max(coeffs_.size(), other.coeffs_.size());
    std::vector<double> result_coeffs(max_size, 0.0);
    
    // Выравниваем полиномы по младшим коэффициентам (константы в конце)
    // coeffs_ хранятся в порядке [a_n, a_{n-1}, ..., a_0]
    // Чтобы сложить, нужно совместить коэффициенты при одинаковых степенях.
    // Для этого добавляем, начиная с конца: младшие коэффициенты (константы) должны быть выровнены.
    for (size_t i = 0; i < coeffs_.size(); ++i) {
        // Позиция в result: смещение от начала = max_size - coeffs_.size() + i
        result_coeffs[max_size - coeffs_.size() + i] += coeffs_[i];
    }
    for (size_t i = 0; i < other.coeffs_.size(); ++i) {
        result_coeffs[max_size - other.coeffs_.size() + i] += other.coeffs_[i];
    }
    
    return Polynomial(result_coeffs);
}

Polynomial Polynomial::operator-(const Polynomial& other) const {
    size_t max_size = std::max(coeffs_.size(), other.coeffs_.size());
    std::vector<double> result_coeffs(max_size, 0.0);
    
    // Выравнивание по младшим коэффициентам
    for (size_t i = 0; i < coeffs_.size(); ++i) {
        result_coeffs[max_size - coeffs_.size() + i] += coeffs_[i];
    }
    for (size_t i = 0; i < other.coeffs_.size(); ++i) {
        result_coeffs[max_size - other.coeffs_.size() + i] -= other.coeffs_[i];
    }
    
    return Polynomial(result_coeffs);
}

Polynomial Polynomial::operator*(double scalar) const {
    std::vector<double> result_coeffs = coeffs_;
    for (double& coeff : result_coeffs) {
        coeff *= scalar;
    }
    return Polynomial(result_coeffs);
}

double Polynomial::squared_error(double x, double target) const {
    double diff = evaluate(x) - target;
    return diff * diff;
}

std::vector<double> Polynomial::gradient_squared_error(double x, double target) const {
    // ∇_a |P(x) - target|^2 = 2 * (P(x) - target) * ∇_a P(x)
    // ∇_a P(x) = [x^n, x^{n-1}, ..., x, 1]
    std::vector<double> gradient(degree_ + 1, 0.0);
    double error = evaluate(x) - target;
    double x_power = 1.0;
    
    // Заполняем gradient[i] для коэффициента a_{n-i} = coeffs_[i]
    // ∂P/∂a_{n-i} = x^i
    for (int i = degree_; i >= 0; --i) {
        gradient[degree_ - i] = 2.0 * error * x_power;
        x_power *= x;
    }
    
    return gradient;
}

// ============== Вспомогательные функции ==============

Polynomial build_lagrange_polynomial(const std::vector<InterpolationNode>& nodes) {
    if (nodes.empty()) {
        return Polynomial(0);
    }
    
    int m = static_cast<int>(nodes.size());
    std::vector<double> result_coeffs;  // результат в формате [a_n, ..., a_0]
    
    // P(x) = Σ_{e=1}^m f(z_e) * L_e(x)
    // L_e(x) = Π_{j≠e} (x - z_j) / (z_e - z_j)
    
    for (int e = 0; e < m; ++e) {
        double ze = nodes[e].x;
        double fe = nodes[e].value;
        
        // Вычисляем базисный полином Лагранжа L_e(x)
        // Начинаем с полинома 1 (степень 0)
        std::vector<double> L_coeffs = {1.0};
        
        for (int j = 0; j < m; ++j) {
            if (j == e) continue;
            double zj = nodes[j].x;
            
            double denom = ze - zj;
            if (std::abs(denom) < 1e-14) {
                throw std::invalid_argument("Interpolation nodes must be distinct");
            }
            
            // Умножаем L(x) на (x - zj)
            // Если L(x) = Σ_{k=0}^{d} c_k x^{d-k}, то L(x)*(x - zj) = 
            //   Σ c_k x^{d-k+1} - zj Σ c_k x^{d-k}
            std::vector<double> new_coeffs(L_coeffs.size() + 1, 0.0);
            
            // Умножение на x: сдвигаем коэффициенты влево
            for (size_t k = 0; k < L_coeffs.size(); ++k) {
                new_coeffs[k] += L_coeffs[k];  // коэффициент при x^{d+1-k} от умножения на x
            }
            // Вычитаем zj * L(x): прибавляем -zj * L_coeffs со смещением 0
            for (size_t k = 0; k < L_coeffs.size(); ++k) {
                new_coeffs[k + 1] += -zj * L_coeffs[k];
            }
            
            L_coeffs = new_coeffs;
            
            // Делим на (ze - zj)
            for (double& c : L_coeffs) {
                c /= denom;
            }
        }
        
        // Умножаем на fe
        for (double& c : L_coeffs) {
            c *= fe;
        }
        
        // Добавляем L_e * f(z_e) к результату
        if (result_coeffs.empty()) {
            result_coeffs = L_coeffs;
        } else {
            // Выравниваем размеры: младшие коэффициенты должны быть выровнены
            // result_coeffs и L_coeffs хранят коэффициенты от старшего к младшему
            // Для сложения нужно, чтобы степени совпадали. 
            // Если L_coeffs имеет степень d1, result_coeffs - степень d2
            // Тогда коэффициенты при x^k находятся на позиции (degree - k)
            // Проще всего: если размеры разные, дополняем меньший вектор нулями в начале
            
            if (L_coeffs.size() > result_coeffs.size()) {
                result_coeffs.insert(result_coeffs.begin(), 
                                     L_coeffs.size() - result_coeffs.size(), 0.0);
            } else if (L_coeffs.size() < result_coeffs.size()) {
                L_coeffs.insert(L_coeffs.begin(), 
                                result_coeffs.size() - L_coeffs.size(), 0.0);
            }
            
            for (size_t i = 0; i < result_coeffs.size(); ++i) {
                result_coeffs[i] += L_coeffs[i];
            }
        }
    }
    
    return Polynomial(result_coeffs);
}

Polynomial build_interpolation_multiplier(const std::vector<InterpolationNode>& nodes) {
    // Π(x - z_e) = (x - z_1)(x - z_2)...(x - z_m)
    Polynomial result(1);  // начинаем с полинома 1 (x^0)
    result.setCoefficients({1.0});
    
    for (const auto& node : nodes) {
        double z = node.x;
        // Умножаем на (x - z) = x - z
        std::vector<double> new_coeffs(result.coefficients().size() + 1, 0.0);
        auto old_coeffs = result.coefficients();
        for (size_t i = 0; i < old_coeffs.size(); ++i) {
            // умножаем на x: сдвиг влево
            new_coeffs[i + 1] += old_coeffs[i];
            // умножаем на -z
            new_coeffs[i] += -z * old_coeffs[i];
        }
        result = Polynomial(new_coeffs);
    }
    
    return result;
}

double integrate_second_derivative_squared(const Polynomial& poly, double a, double b) {
    // Для полинома P(x) = Σ_{k=0}^n a_k x^k
    // P''(x) = Σ_{k=2}^n k*(k-1)*a_k x^{k-2}
    // (P''(x))^2 = Σ_{i=2}^n Σ_{j=2}^n i*(i-1)*j*(j-1)*a_i*a_j * x^{i+j-4}
    // ∫(P''(x))^2 dx = Σ_{i=2}^n Σ_{j=2}^n i*(i-1)*j*(j-1)*a_i*a_j * (b^{i+j-3} - a^{i+j-3}) / (i+j-3)
    
    const auto& coeffs = poly.coefficients();
    int n = poly.degree();
    double integral = 0.0;
    
    // coeffs[k] соответствует a_{n-k}, где k=0..n
    // Нужно преобразовать: a_i соответствует coeffs[n-i]
    
    for (int i = 2; i <= n; ++i) {
        double ai = coeffs[n - i];  // коэффициент при x^i
        double factor_i = i * (i - 1.0);
        
        for (int j = 2; j <= n; ++j) {
            double aj = coeffs[n - j];
            double factor_j = j * (j - 1.0);
            
            int power = i + j - 3;  // степень после интегрирования x^{i+j-4}
            if (power < 0) continue;  // для i=j=1 не может быть, но на всякий случай
            
            double term = factor_i * factor_j * ai * aj;
            double int_value = (std::pow(b, power + 1) - std::pow(a, power + 1)) / (power + 1);
            integral += term * int_value;
        }
    }
    
    return integral;
}

Polynomial Polynomial::operator*(const Polynomial& other) const {
    // Умножение полиномов: (Σ a_i x^i) * (Σ b_j x^j) = Σ_k (Σ_{i+j=k} a_i * b_j) x^k
    // Но у нас коэффициенты хранятся в порядке [a_n, a_{n-1}, ..., a_0]
    // где a_n соответствует x^n
    int n1 = degree_;
    int n2 = other.degree_;
    int result_degree = n1 + n2;
    
    std::vector<double> result_coeffs(result_degree + 1, 0.0);
    
    // Преобразуем к формату [a_0, a_1, ..., a_n] для удобства умножения
    std::vector<double> c1(n1 + 1, 0.0);
    std::vector<double> c2(n2 + 1, 0.0);
    
    for (int i = 0; i <= n1; ++i) {
        c1[i] = coeffs_[n1 - i];  // coeffs_[0] -> x^n, поэтому c1[n1] = coeffs_[0]
    }
    for (int i = 0; i <= n2; ++i) {
        c2[i] = other.coeffs_[n2 - i];
    }
    
    // Умножаем
    for (int i = 0; i <= n1; ++i) {
        for (int j = 0; j <= n2; ++j) {
            result_coeffs[i + j] += c1[i] * c2[j];
        }
    }
    
    // Преобразуем обратно в формат [a_n, ..., a_0]
    std::vector<double> final_coeffs(result_degree + 1, 0.0);
    for (int i = 0; i <= result_degree; ++i) {
        final_coeffs[result_degree - i] = result_coeffs[i];
    }
    
    return Polynomial(final_coeffs);
}

Polynomial Polynomial::minus_scalar(double scalar) const {
    // P(x) - scalar = (a_n x^n + ... + a_0) - scalar = a_n x^n + ... + (a_0 - scalar)
    std::vector<double> new_coeffs = coeffs_;
    if (!new_coeffs.empty()) {
        new_coeffs.back() -= scalar;
    }
    return Polynomial(new_coeffs);
}

} // namespace mixed_approx
