# Включение Google Test
include(GoogleTest)

# Базовый тест полиномов
add_executable(test_polynomial_core
    test_polynomial_core.cpp
)
target_link_libraries(test_polynomial_core
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_polynomial_core)

# Тест шага 2.1.4 - параметризация корректирующего полинома
add_executable(test_2_1_4_correction_polynomial_parameterization
    test_2_1_4_correction_polynomial_parameterization.cpp
)
target_link_libraries(test_2_1_4_correction_polynomial_parameterization
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_4_correction_polynomial_parameterization)

# Базовый тест функционала
add_executable(test_functional_core
    test_functional_core.cpp
)
target_link_libraries(test_functional_core
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_functional_core)

# Тест шага 1.1 - валидация входных данных
add_executable(test_1_1_input_data_validation
    test_1_1_input_data_validation.cpp
)
target_link_libraries(test_1_1_input_data_validation
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_1_1_input_data_validation)

# Тест шага 1.2 - исправление математической формулировки
add_executable(test_1_2_mathematical_formulation_correction
    test_1_2_mathematical_formulation_correction.cpp
)
target_link_libraries(test_1_2_mathematical_formulation_correction
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_1_2_mathematical_formulation_correction)

# Тест шага 2.1.2 - построение интерполяционного полинома
add_executable(test_2_1_2_interpolation_polynomial_construction
    test_2_1_2_interpolation_polynomial_construction.cpp
)
target_link_libraries(test_2_1_2_interpolation_polynomial_construction
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_2_interpolation_polynomial_construction)

# Полная интеграция системы
add_executable(test_full_integration
    test_full_integration.cpp
)
target_link_libraries(test_full_integration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_full_integration)

# Тест шага 2.1 - интеграция параметризации
add_executable(test_2_1_integration
    test_2_1_integration.cpp
)
target_link_libraries(test_2_1_integration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_integration)

# Тест шага 2.1.3 - построение мультипликатора веса
add_executable(test_2_1_3_weight_multiplier_construction
    test_2_1_3_weight_multiplier_construction.cpp
)
target_link_libraries(test_2_1_3_weight_multiplier_construction
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_3_weight_multiplier_construction)

# Тест шага 2.1.1 - математическое обоснование базисной коррекции
add_executable(test_2_1_1_basis_correction_math_foundation
    test_2_1_1_basis_correction_math_foundation.cpp
)
target_link_libraries(test_2_1_1_basis_correction_math_foundation
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_1_basis_correction_math_foundation)

# Объединённый тестовый запуск (опционально)
add_executable(test_all
    test_polynomial_core.cpp
    test_1_1_input_data_validation.cpp
    test_1_2_mathematical_formulation_correction.cpp
    test_2_1_2_interpolation_polynomial_construction.cpp
    test_full_integration.cpp
    test_2_1_integration.cpp
    test_2_1_3_weight_multiplier_construction.cpp
    test_2_1_1_basis_correction_math_foundation.cpp
    test_2_1_4_correction_polynomial_parameterization.cpp
    test_4_1_optimizer_selection_configuration.cpp
    test_4_2_optimizer_initialization_strategy.cpp
)
target_link_libraries(test_all
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_all)

# Тест шага 2.1 - обзор параметризации полиномов
add_executable(test_2_1_polynomial_parameterization_overview
    test_2_1_polynomial_parameterization_overview.cpp
)
target_link_libraries(test_2_1_polynomial_parameterization_overview
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_polynomial_parameterization_overview)

# Тест шага 2.1.5 - сборка итогового полинома F(x)
add_executable(test_2_1_5_final_polynomial_assembly
    test_2_1_5_final_polynomial_assembly.cpp
)
target_link_libraries(test_2_1_5_final_polynomial_assembly
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_5_final_polynomial_assembly)

# Тест шага 2.1.6 - верификация корректности параметризации
add_executable(test_2_1_6_parameterization_verification
    test_2_1_6_parameterization_verification.cpp
)
target_link_libraries(test_2_1_6_parameterization_verification
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_6_parameterization_verification)

# Тест шага 2.1.7 - структуры данных для хранения параметризации
add_executable(test_2_1_7_parameterization_data_structures
    test_2_1_7_parameterization_data_structures.cpp
)
target_link_libraries(test_2_1_7_parameterization_data_structures
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_7_parameterization_data_structures)

# Тест шага 2.1.8 - обработка крайних случаев параметризации
add_executable(test_2_1_8_edge_case_handling
    test_2_1_8_edge_case_handling.cpp
)
target_link_libraries(test_2_1_8_edge_case_handling
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_8_edge_case_handling)

# Тест шага 2.1.9 - интеграция параметризации с оптимизацией
add_executable(test_2_1_9_optimization_integration
    test_2_1_9_optimization_integration.cpp
)
target_link_libraries(test_2_1_9_optimization_integration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_2_1_9_optimization_integration)

# Тест шага 3.1 - реализация класса полинома
add_executable(test_3_1_polynomial_class_implementation
    test_3_1_polynomial_class_implementation.cpp
)
target_link_libraries(test_3_1_polynomial_class_implementation
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_3_1_polynomial_class_implementation)

# Тест шага 3.2 - вычисление компонент функционала
add_executable(test_3_2_functional_components_computation
    test_3_2_functional_components_computation.cpp
)
target_link_libraries(test_3_2_functional_components_computation
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_3_2_functional_components_computation)

# Тест шага 3.3 - вычисление градиента функционала
add_executable(test_3_3_functional_gradient_computation
    test_3_3_functional_gradient_computation.cpp
)
target_link_libraries(test_3_3_functional_gradient_computation
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_3_3_functional_gradient_computation)

# Тест шага 4.1 - выбор и конфигурация оптимизатора
add_executable(test_4_1_optimizer_selection_configuration
    test_4_1_optimizer_selection_configuration.cpp
)
target_link_libraries(test_4_1_optimizer_selection_configuration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_4_1_optimizer_selection_configuration)

# Тест шага 4.2 - стратегия инициализации параметров
add_executable(test_4_2_optimizer_initialization_strategy
    test_4_2_optimizer_initialization_strategy.cpp
)
target_link_libraries(test_4_2_optimizer_initialization_strategy
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_4_2_optimizer_initialization_strategy)

# Тест шага 4.3 - критерии останова оптимизации
add_executable(test_4_3_optimization_stopping_criteria
    test_4_3_optimization_stopping_criteria.cpp
)
target_link_libraries(test_4_3_optimization_stopping_criteria
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_4_3_optimization_stopping_criteria)

# Тест шага 5.1 - защита от особенностей отталкивающего члена
add_executable(test_5_1_barrier_singularity_protection
    test_5_1_barrier_singularity_protection.cpp
)
target_link_libraries(test_5_1_barrier_singularity_protection
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_5_1_barrier_singularity_protection)

# Тест шага 5.2 - масштабирование переменных
add_executable(test_5_2_variable_normalization_scaling
    test_5_2_variable_normalization_scaling.cpp
)
target_link_libraries(test_5_2_variable_normalization_scaling
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_5_2_variable_normalization_scaling)
