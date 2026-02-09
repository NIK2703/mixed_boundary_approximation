#include <iostream>
#include <memory>
#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/optimization_problem_data.h"

using namespace mixed_approx;

int main() {
    try {
        std::cout << "=== Mixed Approximation Method Example ===\n\n";
        
        // Создаем данные задачи оптимизации
        OptimizationProblemData data;
        
        // Задаем аппроксимирующие точки (x_i, f(x_i), σ_i)
        // Например, аппроксимируем функцию f(x) = sin(2πx)
        data.approx_x = {0.1, 0.25, 0.5, 0.75, 0.9};
        data.approx_f = {0.0, 1.0, 0.0, -1.0, 0.0};
        data.approx_weight = {1.0, 1.0, 1.0, 1.0, 1.0};
        
        // Задаем отталкивающие точки (y_j, y_j^*, B_j)
        // Полином должен избегать этих точек
        data.repel_y = {0.6};
        data.repel_forbidden = {10.0};
        data.repel_weight = {100.0};
        
        // Задаем интерполяционные узлы (z_e, f(z_e))
        // Полином должен точно проходить через эти точки
        data.interp_z = {0.0, 1.0};
        data.interp_f = {0.0, 0.0};
        
        // Параметры
        data.gamma = 0.1;        // коэффициент регуляризации
        data.interval_a = 0.0;   // левая граница интервала
        data.interval_b = 1.0;   // правая граница интервала
        data.epsilon = 1e-8;     // защита от деления на ноль
        
        // Валидация данных
        if (!data.is_valid()) {
            std::cerr << "Error: Invalid optimization problem data\n";
            return 1;
        }
        
        std::cout << "Data validated successfully.\n";
        std::cout << "Approximation points: " << data.num_approx_points() << "\n";
        std::cout << "Repel points: " << data.num_repel_points() << "\n";
        std::cout << "Interpolation nodes: " << data.num_interp_nodes() << "\n";
        std::cout << "Gamma: " << data.gamma << "\n\n";
        
        // Создаем объект метода смешанной аппроксимации
        MixedApproximation method;
        
        // Число свободных параметров = степень корректирующего полинома + 1
        int n_free = 4;  // полином степени 3 -> 4 коэффициента
        
        // Создаем оптимизатор (L-BFGS-B)
        auto optimizer = std::make_unique<LBFGSOptimizer>();
        optimizer->set_parameters(1000, 1e-6, 1e-8, 0.01);
        
        std::cout << "Starting optimization...\n";
        std::cout << "Number of free parameters: " << n_free << "\n\n";
        
        // Выполняем оптимизацию
        OptimizationResult result = method.solve(data, n_free, std::move(optimizer));
        
        // Выводим результаты
        std::cout << "\n=== Optimization Results ===\n";
        std::cout << "Success: " << (result.success ? "Yes" : "No") << "\n";
        std::cout << "Message: " << result.message << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Final objective value: " << result.final_objective << "\n";
        
        // Получаем и выводим коэффициенты полинома
        std::cout << "\nPolynomial coefficients (highest degree first):\n";
        Polynomial poly = method.get_polynomial();
        const auto& coeffs = poly.coefficients();
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
        
        // Выводим значения полинома в нескольких точках
        std::cout << "\nPolynomial evaluation:\n";
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
