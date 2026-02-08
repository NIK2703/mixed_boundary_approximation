#ifndef MIXED_APPROXIMATION_CONFIG_READER_H
#define MIXED_APPROXIMATION_CONFIG_READER_H

#include "types.h"
#include <string>
#include <vector>

namespace mixed_approx {

/**
 * @brief Класс для чтения и записи конфигурации метода смешанной аппроксимации
 *
 * Поддерживаемые форматы:
 * 1. Простой текстовый формат (обратная совместимость)
 * 2. YAML конфигурация с CSV файлами данных (рекомендуемый)
 */
class ConfigReader {
public:
    /**
     * @brief Чтение конфигурации из простого текстового файла (обратная совместимость)
     * @param filename путь к файлу конфигурации
     * @return структура ApproximationConfig
     * @throws std::runtime_error при ошибке чтения или парсинга
     */
    static ApproximationConfig read_from_file(const std::string& filename);
    
    /**
     * @brief Чтение конфигурации из YAML файла с CSV данными
     *
     * Формат YAML (см. документацию input_data_format.md):
     * - task: метаданные задачи
     * - problem: параметры задачи (polynomial_degree, interval, weights)
     * - data_sources: пути к CSV файлам с данными
     * - processing_options: опции обработки
     * - advanced: расширенные параметры
     *
     * @param yaml_filename путь к YAML конфигурационному файлу
     * @return структура ApproximationConfig
     * @throws std::runtime_error при ошибке чтения или парсинга
     */
    static ApproximationConfig read_from_yaml(const std::string& yaml_filename);
    
    /**
     * @brief Запись конфигурации в простой текстовый файл
     * @param config конфигурация
     * @param filename путь к файлу для записи
     * @throws std::runtime_error при ошибке записи
     */
    static void write_to_file(const ApproximationConfig& config, const std::string& filename);
    
    /**
     * @brief Чтение данных из CSV файла
     *
     * Поддерживаемые типы CSV:
     * - approximation.csv: колонки x,f,sigma
     * - repulsion.csv: колонки x,y_forbidden,B
     * - interpolation.csv: колонки x,f
     *
     * @param filename путь к CSV файлу
     * @param type тип данных ("approximation", "repulsion", "interpolation")
     * @return вектор соответствующих данных
     */
    static std::vector<WeightedPoint> read_approximation_csv(const std::string& filename);
    static std::vector<RepulsionPoint> read_repulsion_csv(const std::string& filename);
    static std::vector<InterpolationNode> read_interpolation_csv(const std::string& filename);
    
private:
    /**
     * @brief Парсинг строки формата "ключ = значение"
     * @param line строка для парсинга
     * @param key ключ
     * @param value значение (выходной параметр)
     * @return true, если строка распаршена успешно
     */
    static bool parse_key_value(const std::string& line, std::string& key, std::string& value);
    
    /**
     * @brief Пропуск комментариев и пустых строк
     * @param line строка
     * @return true, если строка содержит код (не комментарий и не пустая)
     */
    static bool is_comment_or_empty(const std::string& line);
    
    /**
     * @brief Обрезка пробелов в начале и конце строки
     */
    static std::string trim(const std::string& str);
    
    /**
     * @brief Определение разделителя в CSV файле
     */
    static char detect_csv_delimiter(const std::string& sample_line);
    
    /**
     * @brief Проверка наличия заголовка CSV
     */
    static bool has_csv_header(const std::string& line);
};

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_CONFIG_READER_H
