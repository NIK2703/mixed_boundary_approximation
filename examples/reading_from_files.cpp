#include <iostream>
#include <memory>
#include "mixed_approximation/mixed_approximation.h"
#include "mixed_approximation/config_reader.h"

using namespace mixed_approx;

int main() {
    try {
        std::cout << "=== Mixed Approximation: Reading from Files Example ===\n\n";
        
        // Путь к конфигурационному файлу YAML
        std::string config_path = "data/config.yaml";
        
        std::cout << "Reading configuration from: " << config_path << "\n\n";
        
        // Чтение конфигурации из YAML файла
        // YAML файл ссылается на CSV файлы с данными
        ApproximationConfig config = ConfigReader::read_from_yaml(config_path);
        
        std::cout << "Configuration loaded successfully!\n\n";
        std::cout << "=== Configuration Summary ===\n";
        std::cout << "Task ID: " << "example_spectrum_2026" << "\n";
        std::cout << "Polynomial degree: " << config.polynomial_degree << "\n";
        std::cout << "Interval: [" << config.interval_start << ", " << config.interval_end << "]\n";
        std::cout << "Gamma (regularization): " << config.gamma << "\n";
        std::cout << "Epsilon: " << config.epsilon << "\n";
        std::cout << "Interpolation tolerance: " << config.interpolation_tolerance << "\n\n";
        
        std::cout << "Data loaded:\n";
        std::cout << "  Approximation points: " << config.approx_points.size() << "\n";
        std::cout << "  Repulsion points: " << config.repel_points.size() << "\n";
        std::cout << "  Interpolation nodes: " << config.interp_nodes.size() << "\n\n";
        
        // Вывод некоторых аппроксимирующих точек
        std::cout << "First 5 approximation points:\n";
        for (size_t i = 0; i < std::min(config.approx_points.size(), size_t(5)); ++i) {
            const auto& p = config.approx_points[i];
            std::cout << "  x=" << p.x << ", f=" << p.value << ", sigma=" << p.weight << "\n";
        }
        if (config.approx_points.size() > 5) {
            std::cout << "  ... and " << (config.approx_points.size() - 5) << " more\n";
        }
        std::cout << "\n";
        
        // Вывод отталкивающих точек
        std::cout << "Repulsion points:\n";
        for (size_t i = 0; i < config.repel_points.size(); ++i) {
            const auto& p = config.repel_points[i];
            std::cout << "  x=" << p.x << ", forbidden=" << p.y_forbidden << ", B=" << p.weight << "\n";
        }
        std::cout << "\n";
        
        // Вывод интерполяционных узлов
        std::cout << "Interpolation nodes:\n";
        for (size_t i = 0; i < config.interp_nodes.size(); ++i) {
            const auto& n = config.interp_nodes[i];
            std::cout << "  x=" << n.x << ", f=" << n.value << "\n";
        }
        std::cout << "\n";
        
        // Валидация конфигурации
        std::cout << "Validating configuration...\n";
        std::string validation_error = Validator::validate(config);
        if (!validation_error.empty()) {
            std::cerr << "Configuration validation failed:\n" << validation_error << std::endl;
            return 1;
        }
        std::cout << "Configuration is valid!\n\n";
        
        // Создаем данные задачи оптимизации из конфигурации
        OptimizationProblemData data(config);
        
        // Создание и решение задачи смешанной аппроксимации
        std::cout << "Creating MixedApproximation method...\n";
        MixedApproximation method;
        
        // Число свободных параметров = степень корректирующего полинома + 1
        int n_free = config.polynomial_degree + 1;
        
        // Создание оптимизатора (L-BFGS-B)
        auto optimizer = std::make_unique<LBFGSOptimizer>();
        optimizer->set_parameters(1000, 1e-6, 1e-8, 0.01);
        
        std::cout << "Starting optimization...\n";
        std::cout << "Number of free parameters: " << n_free << "\n\n";
        
        // Выполнение оптимизации
        OptimizationResult result = method.solve(data, n_free, std::move(optimizer));
        
        // Вывод результатов
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
        
        // Проверка интерполяционных условий
        bool interp_ok = method.check_interpolation_conditions();
        std::cout << "\nInterpolation conditions satisfied: "
                  << (interp_ok ? "Yes" : "No") << "\n";
        
        // Вычисление расстояний до отталкивающих точек
        auto distances = method.compute_repel_distances();
        std::cout << "\nDistances to repel points:\n";
        for (size_t i = 0; i < distances.size(); ++i) {
            std::cout << "  Point " << i << ": " << distances[i] << "\n";
        }
        
        // Вывод значений полинома в нескольких точках
        std::cout << "\nPolynomial evaluation:\n";
        for (double x = 0.0; x <= 10.0; x += 1.0) {
            std::cout << "  F(" << x << ") = " << poly.evaluate(x) << "\n";
        }
        
        std::cout << "\n=== Done ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
