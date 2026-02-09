#include "mixed_approximation/weight_polynomial.h"
#include <sstream>
#include <iomanip>

namespace mixed_approx {

WeightPolynomial::WeightPolynomial(const std::vector<double>& nodes) 
    : nodes_(nodes), valid_(false), derivatives_computed_(false) {
    build_interpolation_multiplier();
}

WeightPolynomial::WeightPolynomial(const std::vector<InterpolationNode>& nodes) 
    : valid_(false), derivatives_computed_(false) {
    nodes_.reserve(nodes.size());
    for (const auto& node : nodes) {
        nodes_.push_back(node.x);
    }
    build_interpolation_multiplier();
}

WeightPolynomial::WeightPolynomial() 
    : degree_(0), valid_(true), derivatives_computed_(false) {
    coeffs_ = {1.0};
}

void WeightPolynomial::build_interpolation_multiplier() {
    // W(x) = Π(x - z_e)
    // Строим последовательным умножением
    coeffs_ = {1.0};  // начинаем с полинома 1
    degree_ = 0;
    
    for (double z : nodes_) {
        std::vector<double> new_coeffs(coeffs_.size() + 1, 0.0);
        for (std::size_t i = 0; i < coeffs_.size(); ++i) {
            // умножаем на x: сдвиг влево (коэффициент при x^{d+1-i})
            new_coeffs[i] += -z * coeffs_[i];
            // умножаем на z (добавляем константу)
            new_coeffs[i + 1] += coeffs_[i];
        }
        coeffs_ = new_coeffs;
        ++degree_;
    }
    
    // coeffs_ теперь [a_n, a_{n-1}, ..., a_0] для степени n
    valid_ = true;
    derivatives_computed_ = false;
}

double WeightPolynomial::evaluate(double x) const noexcept {
    if (coeffs_.empty()) return 0.0;
    
    // Схема Горнера
    double result = 0.0;
    for (double coeff : coeffs_) {
        result = result * x + coeff;
    }
    return result;
}

double WeightPolynomial::derivative(double x) const noexcept {
    if (degree_ < 1) return 0.0;
    
    // P'(x) через модифицированную схему Горнера
    double result = 0.0;
    for (std::size_t i = 0; i < coeffs_.size() - 1; ++i) {
        int power = static_cast<int>(coeffs_.size() - 1 - i);
        result = result * x + power * coeffs_[i];
    }
    return result;
}

double WeightPolynomial::second_derivative(double x) const noexcept {
    if (degree_ < 2) return 0.0;
    
    double result = 0.0;
    for (std::size_t i = 0; i < coeffs_.size() - 2; ++i) {
        int power = static_cast<int>(coeffs_.size() - 1 - i);
        double coeff = power * (power - 1) * coeffs_[i];
        result = result * x + coeff;
    }
    return result;
}

void WeightPolynomial::evaluate_all(double x, double& w, double& wp, double& wpp) const noexcept {
    w = evaluate(x);
    wp = derivative(x);
    wpp = second_derivative(x);
}

bool WeightPolynomial::validate(double tolerance) const {
    for (double z : nodes_) {
        if (std::abs(evaluate(z)) > tolerance) {
            return false;
        }
    }
    return true;
}

std::string WeightPolynomial::to_string() const {
    std::ostringstream oss;
    oss << "WeightPolynomial(degree=" << degree_ << ", nodes=[";
    for (std::size_t i = 0; i < std::min(nodes_.size(), static_cast<std::size_t>(5)); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(4) << nodes_[i];
    }
    if (nodes_.size() > 5) oss << ", ...";
    oss << "])";
    return oss.str();
}

std::vector<double> WeightPolynomial::roots() const {
    // Корни должны совпадать с исходными узлами
    return nodes_;
}

} // namespace mixed_approx
