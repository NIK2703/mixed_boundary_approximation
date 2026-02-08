#include <iostream>
#include <memory>
#include "mixed_approximation/mixed_approximation.h"

using namespace mixed_approx;

int main() {
    try {
        std::cout << "=== Mixed Approximation Method Example ===\n\n";
        
        // Создаем конфигурацию
        ApproximationConfig config;
        config.polynomial_degree = 3;  // степень полинома
        config.interval_start = 0.0;
        config.interval_end = 1.0;
        config.gamma = 0.1;  // коэффициент регуляризации
        
        // Задаем аппроксимирующие точки (x_i, f(x_i), σ_i)
        // Например, аппроксимируем функцию f(x) = sin(2πx)
        config.approx_points = {
            WeightedPoint(0.1, 0.0, 1.0),
            WeightedPoint(0.25, 1.0, 1.0),
            WeightedPoint(0.5, 0.0, 1.0),
            WeightedPoint(0.75, -1.0, 1.0),
            WeightedPoint(0.9, 0.0, 1.0)
        };
        
        // Задаем отталкивающие точки (y_j, y_j^*, B_j)
        // Полином должен избегать этих точек
        config.repel_points = {
            WeightedPoint(0.6, 10.0, 100.0),  // избегать значения 10 в точке x=0.6
        };
        
        // Задаем интерполяционные узлы (z_e, f(z_e))
        // Полином должен точно проходить через эти точки
        config.interp_nodes = {
            InterpolationNode(0.0, 0.0),
            InterpolationNode(1.0, 0.0)
        };
        
        // Валидация конфигурации
        std::string validation_error = Validator::validate(config);
        if (!validation_error.empty()) {
            std::cerr << "Configuration error:\n" << validation_error << std::endl;
            return 1;
        }
        
        std::cout << "Configuration validated successfully.\n";
        std::cout << "Polynomial degree: " << config.polynomial_degree << "\n";
        std::cout << "Approximation points: " << config.approx_points.size() << "\n";
        std::cout << "Repel points: " << config.repel_points.size() << "\n";
        std::cout << "Interpolation nodes: " << config.interp_nodes.size() << "\n";
        std::cout << "Gamma: " << config.gamma << "\n\n";
        
        // Создаем объект метода смешанной аппроксимации
        MixedApproximation method(config);
        
        // Создаем оптимизатор (адаптивный градиентный спуск)
        auto optimizer = std::make_unique<AdaptiveGradientDescentOptimizer>();
        optimizer->set_parameters(
            1000,    // max iterations
            1e-6,    // gradient tolerance
            1e-8,    // objective tolerance
            0.01     // initial step
        );
        
        std::cout << "Starting optimization...\n";
        
        // Выполняем оптимизацию
        OptimizationResult result = method.solve(std::move(optimizer));
        
        // Выводим результаты
        std::cout << "\n=== Optimization Results ===\n";
        std::cout << "Success: " << (result.success ? "Yes" : "No") << "\n";
        std::cout << "Message: " << result.message << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Final objective value: " << result.final_objective << "\n";
        
        std::cout << "\nPolynomial coefficients (highest degree first):\n";
        const auto& coeffs = result.coefficients;
        for (size_t i = 0; i < coeffs.size(); ++i) {
            int power = static_cast<int>(coeffs.size() - 1 - i);
            std::cout << "  a_" << power << " = " << coeffs[i] << "\n";
        }
        
        // Проверяем интерполяционные условия
        bool interp_ok = method.check_interpolation_conditions();
        std::cout << "\nInterpolation conditions satisfied: " 
                  << (interp_ok ? "Yes" : "No") << "\n";
        
        // Вычисляем расстояния до отталкивающих точек
        auto distances = method.compute_repel_distances();
        std::cout << "\nDistances to repel points:\n";
        for (size_t i = 0; i < distances.size(); ++i) {
            std::cout << "  Point " << i << ": " << distances[i] << "\n";
        }
        
        // Выводим компоненты функционала
        auto components = method.get_functional_components();
        std::cout << "\nFunctional components:\n";
        std::cout << "  Approximation: " << components.approx_component << "\n";
        std::cout << "  Repel: " << components.repel_component << "\n";
        std::cout << "  Regularization: " << components.reg_component << "\n";
        std::cout << "  Total: " << components.total << "\n";
        
        // Выводим значения полинома в нескольких точках
        std::cout << "\nPolynomial evaluation:\n";
        Polynomial poly = method.get_polynomial();
        for (double x = 0.0; x <= 1.0; x += 0.1) {
            std::cout << "  F(" << x << ") = " << poly.evaluate(x) << "\n";
        }
        
        std::cout << "\n=== Done ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
