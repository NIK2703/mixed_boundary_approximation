#include "mixed_approximation/config_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

#ifdef HAS_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

namespace mixed_approx {

std::string ConfigReader::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

bool ConfigReader::is_comment_or_empty(const std::string& line) {
    std::string trimmed = trim(line);
    return trimmed.empty() || trimmed[0] == '#';
}

bool ConfigReader::parse_key_value(const std::string& line, std::string& key, std::string& value) {
    std::string trimmed = trim(line);
    size_t equals_pos = trimmed.find('=');
    if (equals_pos == std::string::npos) {
        return false;
    }
    
    key = trim(trimmed.substr(0, equals_pos));
    value = trim(trimmed.substr(equals_pos + 1));
    return !key.empty();
}

char ConfigReader::detect_csv_delimiter(const std::string& sample_line) {
    // Подсчитываем возможные разделители
    int comma_count = 0, semicolon_count = 0, tab_count = 0;
    for (char c : sample_line) {
        if (c == ',') comma_count++;
        else if (c == ';') semicolon_count++;
        else if (c == '\t') tab_count++;
    }
    
    // Выбираем самый частый
    if (comma_count >= semicolon_count && comma_count >= tab_count) return ',';
    if (semicolon_count >= comma_count && semicolon_count >= tab_count) return ';';
    return '\t';
}

bool ConfigReader::has_csv_header(const std::string& line) {
    std::string trimmed = trim(line);
    // Проверяем, содержит ли первая непустая строка нечисловые токены
    std::istringstream iss(trimmed);
    std::string token;
    std::getline(iss, token, ',');
    token = trim(token);
    
    // Если токен не может быть преобразован в число, считаем что это заголовок
    try {
        std::stod(token);
        return false;  // это число, заголовка нет
    } catch (...) {
        return true;   // не число, вероятно заголовок
    }
}

std::vector<WeightedPoint> ConfigReader::read_approximation_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open approximation CSV file: " + filename);
    }
    
    std::vector<WeightedPoint> points;
    std::string line;
    int line_number = 0;
    bool header_processed = false;
    char delimiter = ',';
    
    while (std::getline(file, line)) {
        line_number++;
        line = trim(line);
        
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Обработка первой значимой строки (заголовка или данных)
        if (!header_processed) {
            delimiter = detect_csv_delimiter(line);
            if (has_csv_header(line)) {
                header_processed = true;
                continue;  // пропускаем заголовок
            }
            header_processed = true;  // заголовка нет, эта строка - данные
        }
        
        std::istringstream iss(line);
        std::string x_str, f_str, sigma_str;
        
        if (!std::getline(iss, x_str, delimiter) ||
            !std::getline(iss, f_str, delimiter) ||
            !std::getline(iss, sigma_str, delimiter)) {
            throw std::runtime_error("Missing required columns at line " + std::to_string(line_number));
        }
        
        try {
            double x = std::stod(trim(x_str));
            double f = std::stod(trim(f_str));
            double sigma = std::stod(trim(sigma_str));
            
            if (sigma <= 0) {
                throw std::runtime_error("Sigma must be positive at line " + std::to_string(line_number));
            }
            
            points.emplace_back(x, f, sigma);
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid numeric value at line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    
    file.close();
    return points;
}

std::vector<RepulsionPoint> ConfigReader::read_repulsion_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open repulsion CSV file: " + filename);
    }
    
    std::vector<RepulsionPoint> points;
    std::string line;
    int line_number = 0;
    bool header_processed = false;
    char delimiter = ',';
    
    while (std::getline(file, line)) {
        line_number++;
        line = trim(line);
        
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Обработка первой значимой строки (заголовка или данных)
        if (!header_processed) {
            delimiter = detect_csv_delimiter(line);
            if (has_csv_header(line)) {
                header_processed = true;
                continue;  // пропускаем заголовок
            }
            header_processed = true;  // заголовка нет, эта строка - данные
        }
        
        std::istringstream iss(line);
        std::string x_str, y_forbidden_str, B_str;
        
        if (!std::getline(iss, x_str, delimiter) ||
            !std::getline(iss, y_forbidden_str, delimiter) ||
            !std::getline(iss, B_str, delimiter)) {
            throw std::runtime_error("Missing required columns at line " + std::to_string(line_number));
        }
        
        try {
            double x = std::stod(trim(x_str));
            double y_forbidden = std::stod(trim(y_forbidden_str));
            double B = std::stod(trim(B_str));
            
            if (B <= 0) {
                throw std::runtime_error("B (barrier weight) must be positive at line " + std::to_string(line_number));
            }
            
            points.emplace_back(x, y_forbidden, B);
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid numeric value at line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    
    file.close();
    return points;
}

std::vector<InterpolationNode> ConfigReader::read_interpolation_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open interpolation CSV file: " + filename);
    }
    
    std::vector<InterpolationNode> nodes;
    std::string line;
    int line_number = 0;
    bool header_processed = false;
    char delimiter = ',';
    
    while (std::getline(file, line)) {
        line_number++;
        line = trim(line);
        
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Обработка первой значимой строки (заголовка или данных)
        if (!header_processed) {
            delimiter = detect_csv_delimiter(line);
            if (has_csv_header(line)) {
                header_processed = true;
                continue;  // пропускаем заголовок
            }
            header_processed = true;  // заголовка нет, эта строка - данные
        }
        
        std::istringstream iss(line);
        std::string x_str, f_str;
        
        if (!std::getline(iss, x_str, delimiter) ||
            !std::getline(iss, f_str, delimiter)) {
            throw std::runtime_error("Missing required columns at line " + std::to_string(line_number));
        }
        
        try {
            double x = std::stod(trim(x_str));
            double f = std::stod(trim(f_str));
            
            nodes.emplace_back(x, f);
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid numeric value at line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    
    file.close();
    return nodes;
}

ApproximationConfig ConfigReader::read_from_yaml(const std::string& yaml_filename) {
#ifndef HAS_YAML_CPP
    throw std::runtime_error("YAML support not compiled. Install yaml-cpp or use simple config format.");
#else
    ApproximationConfig config;
    
    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_filename);
        
        if (!yaml_config["problem"]) {
            throw std::runtime_error("Missing 'problem' section in YAML config");
        }
        
        const YAML::Node& problem = yaml_config["problem"];
        
        // Основные параметры
        config.polynomial_degree = problem["polynomial_degree"].as<int>(config.polynomial_degree);
        
        if (problem["interval"]) {
            config.interval_start = problem["interval"]["a"].as<double>(config.interval_start);
            config.interval_end = problem["interval"]["b"].as<double>(config.interval_end);
        }
        
        config.gamma = problem["weights"]["regularization"].as<double>(config.gamma);
        
        // Параметры численной устойчивости
        if (problem["numerical"]) {
            config.epsilon = problem["numerical"]["epsilon_safe"].as<double>(config.epsilon);
            config.interpolation_tolerance = problem["numerical"]["epsilon_interp"].as<double>(config.interpolation_tolerance);
        }
        
        // Загрузка данных из CSV файлов
        if (yaml_config["data_sources"]) {
            const YAML::Node& data_sources = yaml_config["data_sources"];
            
            if (data_sources["approximation"]) {
                std::string approx_file = data_sources["approximation"].as<std::string>();
                config.approx_points = read_approximation_csv(approx_file);
            }
            
            if (data_sources["repulsion"]) {
                std::string repel_file = data_sources["repulsion"].as<std::string>();
                config.repel_points = read_repulsion_csv(repel_file);
            }
            
            if (data_sources["interpolation"]) {
                std::string interp_file = data_sources["interpolation"].as<std::string>();
                config.interp_nodes = read_interpolation_csv(interp_file);
            }
        }
        
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw;
    }
    
    return config;
#endif
}

ApproximationConfig ConfigReader::read_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    ApproximationConfig config;
    std::string line;
    int line_number = 0;
    
    // Флаги для ожидания данных точек
    bool expecting_approx_points = false;
    bool expecting_repel_points = false;
    bool expecting_interp_nodes = false;
    int remaining_points = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        line = trim(line);
        
        if (is_comment_or_empty(line)) {
            continue;
        }
        
        // Обработка ожидания данных точек
        if (expecting_approx_points) {
            if (remaining_points > 0) {
                std::istringstream iss(line);
                double x, value, weight;
                if (iss >> x >> value >> weight) {
                    config.approx_points.emplace_back(x, value, weight);
                    remaining_points--;
                } else {
                    throw std::runtime_error("Invalid approx point format at line " + std::to_string(line_number));
                }
                continue;
            } else {
                // Все точки прочитаны, сбрасываем флаг
                expecting_approx_points = false;
            }
        }
        
        if (expecting_repel_points) {
            if (remaining_points > 0) {
                std::istringstream iss(line);
                double x, value, weight;
                if (iss >> x >> value >> weight) {
                    config.repel_points.emplace_back(x, value, weight);
                    remaining_points--;
                } else {
                    throw std::runtime_error("Invalid repel point format at line " + std::to_string(line_number));
                }
                continue;
            } else {
                expecting_repel_points = false;
            }
        }
        
        if (expecting_interp_nodes) {
            if (remaining_points > 0) {
                std::istringstream iss(line);
                double x, value;
                if (iss >> x >> value) {
                    config.interp_nodes.emplace_back(x, value);
                    remaining_points--;
                } else {
                    throw std::runtime_error("Invalid interpolation node format at line " + std::to_string(line_number));
                }
                continue;
            } else {
                expecting_interp_nodes = false;
            }
        }
        
        // Парсинг ключ-значение
        std::string key, value;
        if (!parse_key_value(line, key, value)) {
            throw std::runtime_error("Invalid key-value format at line " + std::to_string(line_number));
        }
        
        // Парсинг скалярных параметров
        if (key == "polynomial_degree") {
            config.polynomial_degree = std::stoi(value);
        } else if (key == "interval_start") {
            config.interval_start = std::stod(value);
        } else if (key == "interval_end") {
            config.interval_end = std::stod(value);
        } else if (key == "gamma") {
            config.gamma = std::stod(value);
        } else if (key == "epsilon") {
            config.epsilon = std::stod(value);
        } else if (key == "interpolation_tolerance") {
            config.interpolation_tolerance = std::stod(value);
        } 
        // Обработка счетчиков точек
        else if (key == "approx_points_count") {
            int count = std::stoi(value);
            config.approx_points.reserve(count);
            expecting_approx_points = true;
            remaining_points = count;
        } else if (key == "repel_points_count") {
            int count = std::stoi(value);
            config.repel_points.reserve(count);
            expecting_repel_points = true;
            remaining_points = count;
        } else if (key == "interp_nodes_count") {
            int count = std::stoi(value);
            config.interp_nodes.reserve(count);
            expecting_interp_nodes = true;
            remaining_points = count;
        } else {
            // Игнорируем неизвестные ключи (можно добавить предупреждение)
        }
    }
    
    // Проверяем, что все ожидаемые точки были прочитаны
    if (expecting_approx_points && remaining_points > 0) {
        throw std::runtime_error("Not enough approx_points data");
    }
    if (expecting_repel_points && remaining_points > 0) {
        throw std::runtime_error("Not enough repel_points data");
    }
    if (expecting_interp_nodes && remaining_points > 0) {
        throw std::runtime_error("Not enough interp_nodes data");
    }
    
    file.close();
    return config;
}

void ConfigReader::write_to_file(const ApproximationConfig& config, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    file << "# Configuration for Mixed Approximation Method\n";
    file << "# Generated by ConfigReader\n\n";
    
    file << "polynomial_degree = " << config.polynomial_degree << "\n";
    file << "interval_start = " << config.interval_start << "\n";
    file << "interval_end = " << config.interval_end << "\n";
    file << "gamma = " << config.gamma << "\n";
    file << "epsilon = " << config.epsilon << "\n";
    file << "interpolation_tolerance = " << config.interpolation_tolerance << "\n\n";
    
    file << "# Approximation points: x value weight\n";
    file << "approx_points_count = " << config.approx_points.size() << "\n";
    for (const auto& point : config.approx_points) {
        file << point.x << " " << point.value << " " << point.weight << "\n";
    }
    file << "\n";
    
    file << "# Repel points: x y_forbidden weight\n";
    file << "repel_points_count = " << config.repel_points.size() << "\n";
    for (const auto& point : config.repel_points) {
        file << point.x << " " << point.y_forbidden << " " << point.weight << "\n";
    }
    file << "\n";
    
    file << "# Interpolation nodes: x value\n";
    file << "interp_nodes_count = " << config.interp_nodes.size() << "\n";
    for (const auto& node : config.interp_nodes) {
        file << node.x << " " << node.value << "\n";
    }
    
    file.close();
}

} // namespace mixed_approx
