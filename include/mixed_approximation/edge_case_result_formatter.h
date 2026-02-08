#ifndef MIXED_APPROXIMATION_EDGE_CASE_RESULT_FORMATTER_H
#define MIXED_APPROXIMATION_EDGE_CASE_RESULT_FORMATTER_H

#include "edge_case_handler.h"
#include <string>

namespace mixed_approx {

/**
 * @brief Форматирование результата обработки в строку
 */
std::string format_edge_case_result(const EdgeCaseHandlingResult& result);

/**
 * @brief Форматирование результата адаптации в строку
 */
std::string format_zero_nodes_result(const ZeroNodesResult& result);

/**
 * @brief Форматирование результата полной интерполяции
 */
std::string format_full_interpolation_result(const FullInterpolationResult& result);

/**
 * @brief Форматирование результата избыточных ограничений
 */
std::string format_overconstrained_result(const OverconstrainedResult& result);

/**
 * @brief Форматирование результата близких узлов
 */
std::string format_close_nodes_result(const CloseNodesResult& result);

/**
 * @brief Форматирование результата высокой степени
 */
std::string format_high_degree_result(const HighDegreeResult& result);

/**
 * @brief Форматирование результата вырожденности
 */
std::string format_degeneracy_result(const DegeneracyResult& result);

} // namespace mixed_approx

#endif // MIXED_APPROXIMATION_EDGE_CASE_RESULT_FORMATTER_H
