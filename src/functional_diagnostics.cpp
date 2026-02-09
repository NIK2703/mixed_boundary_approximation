#include "mixed_approximation/functional.h"
#include <sstream>
#include <iomanip>
#include <ios>

namespace mixed_approx {

// ============== Реализация FunctionalDiagnostics ==============

std::string FunctionalDiagnostics::format_report() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    
    oss << "Функционал смешанной аппроксимации:\n";
    oss << "  Аппроксимирующий член:  " << normalized_approx << "   (вес: " << weight_approx << ", доля: " << share_approx << "%)\n";
    oss << "  Отталкивающий член:     " << normalized_repel << "   (вес: " << weight_repel << ", доля: " << share_repel << "%)\n";
    oss << "  Регуляризация:          " << normalized_reg << "   (вес: " << weight_reg << ", доля: " << share_reg << "%)\n";
    oss << "  Итого:                  " << total_functional << "\n\n";
    
    oss << "Диагностика:\n";
    if (std::isfinite(min_repulsion_distance)) {
        oss << "  Минимальное расстояние до запрещённых точек: " << min_repulsion_distance << "\n";
    } else {
        oss << "  Минимальное расстояние до запрещённых точек: N/A (нет точек)\n";
    }
    oss << "  Максимальный остаток аппроксимации: " << max_residual << "\n";
    oss << "  Норма второй производной: " << second_deriv_norm << "\n";
    
    if (has_numerical_anomaly) {
        oss << "\nВНИМАНИЕ: Обнаружена численная аномалия!\n";
        oss << "  " << anomaly_description << "\n";
    }
    
    if (is_dominant_component()) {
        oss << "\nВНИМАНИЕ: Доминирование одной компоненты!\n";
        oss << get_weight_recommendation() << "\n";
    }
    
    return oss.str();
}

std::string FunctionalDiagnostics::get_weight_recommendation() const {
    std::ostringstream oss;
    
    if (share_approx > 95.0) {
        oss << "Рекомендация: Аппроксимация доминирует (>95%). ";
        oss << "Рассмотрите увеличение весов отталкивания (B_j) или уменьшение весов аппроксимации (σ_i).";
    } else if (share_repel > 95.0) {
        oss << "Рекомендация: Отталкивание доминирует (>95%). ";
        oss << "Рассмотрите уменьшение весов отталкивания (B_j) или увеличение γ.";
    } else if (share_reg > 95.0) {
        oss << "Рекомендация: Регуляризация доминирует (>95%). ";
        oss << "Рассмотрите уменьшение γ или увеличение весов других компонент.";
    }
    
    return oss.str();
}

} // namespace mixed_approx
