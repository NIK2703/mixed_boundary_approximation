#include <gtest/gtest.h>
#include "mixed_approximation/parameterization_data.h"
#include "mixed_approximation/functional.h"

namespace mixed_approx {
namespace test {

// Тест для InterpolationNodeSet
class InterpolationNodeSetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Подготовка тестовых данных
        x_nodes_ = {0.0, 0.3, 0.5, 0.7, 1.0};
        y_values_ = {1.0, 0.7, 0.5, 0.7, 1.0};
    }
    
    std::vector<double> x_nodes_;
    std::vector<double> y_values_;
};

TEST_F(InterpolationNodeSetTest, BasicConstruction) {
    InterpolationNodeSet nodes;
    nodes.build(x_nodes_, y_values_, 0.0, 1.0);
    
    EXPECT_TRUE(nodes.is_valid);
    EXPECT_EQ(nodes.size(), 5);
    EXPECT_NEAR(nodes.norm_center, 0.5, 1e-10);
    EXPECT_NEAR(nodes.norm_scale, 0.5, 1e-10);
    EXPECT_FALSE(nodes.has_close_nodes);
}

TEST_F(InterpolationNodeSetTest, Normalization) {
    InterpolationNodeSet nodes;
    nodes.build(x_nodes_, y_values_, 0.0, 1.0);
    
    // Проверка нормализации
    EXPECT_NEAR(nodes.x_norm[0], -1.0, 1e-10);
    EXPECT_NEAR(nodes.x_norm[4], 1.0, 1e-10);
    EXPECT_NEAR(nodes.x_norm[2], 0.0, 1e-10);
}

TEST_F(InterpolationNodeSetTest, MergeCloseNodes) {
    // Добавляем близкие узлы
    std::vector<double> close_x = {0.0, 0.001, 0.002, 0.5, 0.999, 1.0};
    std::vector<double> close_y = {1.0, 0.99, 0.98, 0.5, 0.99, 1.0};
    
    InterpolationNodeSet nodes;
    nodes.build(close_x, close_y, 0.0, 1.0, 0.01);  // 1% порог
    
    EXPECT_TRUE(nodes.has_close_nodes);
    EXPECT_LT(nodes.size(), 6);  // Узлы должны быть объединены
}

TEST_F(InterpolationNodeSetTest, DuplicateHandling) {
    std::vector<double> dup_x = {0.0, 0.0, 0.5, 1.0};
    std::vector<double> dup_y = {1.0, 0.9, 0.5, 1.0};
    
    InterpolationNodeSet nodes;
    nodes.build(dup_x, dup_y, 0.0, 1.0);
    
    EXPECT_TRUE(nodes.is_valid);
    EXPECT_EQ(nodes.size(), 3);  // Дубликат должен быть объединён
}

// Тест для ParameterizationBuilder
class ParameterizationBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.polynomial_degree = 5;
        config_.interval_start = 0.0;
        config_.interval_end = 1.0;
        config_.gamma = 0.01;
        
        // Добавляем интерполяционные узлы
        config_.interp_nodes.push_back(InterpolationNode(0.0, 1.0));
        config_.interp_nodes.push_back(InterpolationNode(0.5, 0.5));
        config_.interp_nodes.push_back(InterpolationNode(1.0, 1.0));
        
        // Добавляем аппроксимирующие точки
        config_.approx_points.push_back(WeightedPoint(0.25, 0.75, 1.0));
        config_.approx_points.push_back(WeightedPoint(0.75, 0.75, 1.0));
    }
    
    ApproximationConfig config_;
};

TEST_F(ParameterizationBuilderTest, FullWorkflow) {
    ParameterizationBuilder builder;
    
    // Валидация узлов
    EXPECT_TRUE(builder.validate_nodes(config_));
    
    // Коррекция формулировки
    EXPECT_TRUE(builder.correct_formulation(config_));
    
    // Построение базиса
    EXPECT_TRUE(builder.build_basis());
    
    // Построение весового множителя
    EXPECT_TRUE(builder.build_weight_multiplier());
    
    // Построение корректирующего полинома
    EXPECT_TRUE(builder.build_correction_poly());
    
    // Сборка составного полинома
    EXPECT_TRUE(builder.assemble_composite());
    
    // Верификация
    EXPECT_TRUE(builder.verify_parameterization());
}

TEST_F(ParameterizationBuilderTest, DiagnosticInfo) {
    ParameterizationBuilder builder;
    builder.validate_nodes(config_);
    
    const auto& log = builder.get_build_log();
    EXPECT_FALSE(log.empty());
    
    const auto& warnings = builder.get_warnings();
    EXPECT_TRUE(warnings.empty());  // Ожидаем пустой список предупреждений
    
    const auto& errors = builder.get_errors();
    EXPECT_TRUE(errors.empty());  // Ожидаем пустой список ошибок
}

// Тест для FunctionalEvaluator
class FunctionalEvaluatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.polynomial_degree = 5;
        config_.interval_start = 0.0;
        config_.interval_end = 1.0;
        config_.gamma = 0.001;
        
        config_.interp_nodes.push_back(InterpolationNode(0.0, 1.0));
        config_.interp_nodes.push_back(InterpolationNode(0.5, 0.5));
        config_.interp_nodes.push_back(InterpolationNode(1.0, 1.0));
        
        config_.approx_points.push_back(WeightedPoint(0.25, 0.75, 1.0));
        config_.approx_points.push_back(WeightedPoint(0.75, 0.75, 1.0));
        
        config_.repel_points.push_back(RepulsionPoint(0.4, 0.9, 1.0));
    }
    
    ApproximationConfig config_;
};

TEST_F(FunctionalEvaluatorTest, ObjectiveEvaluation) {
    ParameterizationBuilder builder;
    
    ASSERT_TRUE(builder.validate_nodes(config_));
    ASSERT_TRUE(builder.correct_formulation(config_));
    ASSERT_TRUE(builder.build_basis());
    ASSERT_TRUE(builder.build_weight_multiplier());
    ASSERT_TRUE(builder.build_correction_poly());
    ASSERT_TRUE(builder.assemble_composite());
    
    const auto& param = builder.get_parameterization();
    
    // Нулевые коэффициенты Q(x) = 0
    std::vector<double> q(param.num_free_parameters(), 0.0);
    
    FunctionalEvaluator evaluator(config_);
    double J = evaluator.evaluate_objective(param, q);
    
    EXPECT_GE(J, 0.0);  // Функционал должен быть неотрицательным
}

TEST_F(FunctionalEvaluatorTest, GradientEvaluation) {
    ParameterizationBuilder builder;
    
    ASSERT_TRUE(builder.validate_nodes(config_));
    ASSERT_TRUE(builder.correct_formulation(config_));
    ASSERT_TRUE(builder.build_basis());
    ASSERT_TRUE(builder.build_weight_multiplier());
    ASSERT_TRUE(builder.build_correction_poly());
    ASSERT_TRUE(builder.assemble_composite());
    
    const auto& param = builder.get_parameterization();
    
    std::vector<double> q(param.num_free_parameters(), 0.0);
    
    FunctionalEvaluator evaluator(config_);
    std::vector<double> grad;
    evaluator.evaluate_gradient(param, q, grad);
    
    EXPECT_EQ(grad.size(), q.size());
}

TEST_F(FunctionalEvaluatorTest, CombinedEvaluation) {
    ParameterizationBuilder builder;
    
    ASSERT_TRUE(builder.validate_nodes(config_));
    ASSERT_TRUE(builder.correct_formulation(config_));
    ASSERT_TRUE(builder.build_basis());
    ASSERT_TRUE(builder.build_weight_multiplier());
    ASSERT_TRUE(builder.build_correction_poly());
    ASSERT_TRUE(builder.assemble_composite());
    
    const auto& param = builder.get_parameterization();
    
    std::vector<double> q(param.num_free_parameters(), 0.0);
    
    FunctionalEvaluator evaluator(config_);
    double f;
    std::vector<double> grad;
    evaluator.evaluate_objective_and_gradient(param, q, f, grad);
    
    EXPECT_GE(f, 0.0);
    EXPECT_EQ(grad.size(), q.size());
}

// Тест для ParameterizationWorkspace
TEST(ParameterizationWorkspaceTest, MemoryManagement) {
    ParameterizationWorkspace workspace;
    
    // Выделение памяти
    auto* vec1 = workspace.allocate_values(100);
    auto* vec2 = workspace.allocate_values(200);
    
    EXPECT_EQ(workspace.pool_size(), 2);
    
    // Освобождение
    workspace.release_values(vec1);
    EXPECT_EQ(workspace.pool_size(), 1);
    
    workspace.release_values(vec2);
    EXPECT_EQ(workspace.pool_size(), 0);
}

// Тест для краевых случаев
TEST(EdgeCasesTest, NoInterpolationNodes) {
    ApproximationConfig config;
    config.polynomial_degree = 3;
    config.interval_start = 0.0;
    config.interval_end = 1.0;
    
    // Без интерполяционных узлов - ожидаем что валидация вернёт false
    // (требуется минимум 1 узел)
    ParameterizationBuilder builder;
    
    EXPECT_FALSE(builder.validate_nodes(config));  // Ожидаем false - нет узлов
    
    // При отсутствии узлов validate_nodes возвращает false и добавляет ошибку
    // Это ожидаемое поведение - без узлов параметризация невозможна
    EXPECT_EQ(builder.get_errors().size(), 1);  // Одна ошибка ожидаема
}

}  // namespace test
}  // namespace mixed_approx
