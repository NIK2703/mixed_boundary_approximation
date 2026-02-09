# Включение Google Test
include(GoogleTest)

# Полиномы
add_executable(test_polynomial
    polynomial/test_polynomial.cpp
)
target_link_libraries(test_polynomial
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_polynomial)

# Correction Polynomial (шаг 2.1.4)
add_executable(test_correction_polynomial
    correction_polynomial/test_correction_polynomial.cpp
)
target_link_libraries(test_correction_polynomial
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_correction_polynomial)

# Функционал
add_executable(test_functional
    functional/test_functional.cpp
)
target_link_libraries(test_functional
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_functional)

# Валидатор - базовые тесты
add_executable(test_validator_basic
    validator/test_validator_basic.cpp
)
target_link_libraries(test_validator_basic
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_validator_basic)

# Валидатор - расширенные тесты
add_executable(test_validator_advanced
    validator/test_validator_advanced.cpp
)
target_link_libraries(test_validator_advanced
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_validator_advanced)

# Декомпозиция
add_executable(test_decomposition
    decomposition/test_decomposition.cpp
)
target_link_libraries(test_decomposition
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_decomposition)

# Интеграционные тесты - основной файл
add_executable(test_integration
    integration/test_integration.cpp
)
target_link_libraries(test_integration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_integration)

# Интеграционные тесты - шаг 2.1
add_executable(test_step_2_1_integration
    integration/test_step_2_1_integration.cpp
)
target_link_libraries(test_step_2_1_integration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_integration)

# Weight Multiplier (шаг 2.1.3)
add_executable(test_weight_multiplier
    weight_multiplier/test_weight_multiplier.cpp
)
target_link_libraries(test_weight_multiplier
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_weight_multiplier)

# Interpolation Basis (шаг 2.1.2)
add_executable(test_interpolation_basis
    interpolation_basis/test_interpolation_basis.cpp
)
target_link_libraries(test_interpolation_basis
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_interpolation_basis)

# Объединённый тестовый запуск (опционально)
add_executable(test_all
    polynomial/test_polynomial.cpp
    correction_polynomial/test_correction_polynomial.cpp
    functional/test_functional.cpp
    validator/test_validator_basic.cpp
    validator/test_validator_advanced.cpp
    decomposition/test_decomposition.cpp
    integration/test_integration.cpp
    integration/test_step_2_1_integration.cpp
    weight_multiplier/test_weight_multiplier.cpp
    interpolation_basis/test_interpolation_basis.cpp
    test_step_2_1_1.cpp
)
target_link_libraries(test_all
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_all)

# Тест шага 2.1.1 - полная параметризация "базис + коррекция"
add_executable(test_step_2_1_1
    test_step_2_1_1.cpp
)
target_link_libraries(test_step_2_1_1
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_1)

# Тест шага 2.1.5 - сборка итогового полинома F(x)
add_executable(test_step_2_1_5
    test_step_2_1_5.cpp
)
target_link_libraries(test_step_2_1_5
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_5)

# Тест шага 2.1.6 - верификация корректности параметризации
add_executable(test_step_2_1_6
    test_step_2_1_6.cpp
)
target_link_libraries(test_step_2_1_6
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_6)

# Тест шага 2.1.7 - структуры данных для хранения параметризации
add_executable(test_step_2_1_7
    test_parameterization_data.cpp
)
target_link_libraries(test_step_2_1_7
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_7)

# Тест шага 2.1.8 - обработка крайних случаев параметризации
add_executable(test_step_2_1_8
    test_edge_case_handler.cpp
)
target_link_libraries(test_step_2_1_8
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_8)

# Тест шага 2.1.9 - интеграция параметризации с оптимизацией
add_executable(test_step_2_1_9
    test_step_2_1_9.cpp
)
target_link_libraries(test_step_2_1_9
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_2_1_9)

# Тест шага 3.1.1 - ConstrainedPolynomial
add_executable(test_step_3_1_1
    test_step_3_1_1.cpp
)
target_link_libraries(test_step_3_1_1
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_3_1_1)

# Тест шага 3.2 - расширенный функционал с барьерной защитой и нормализацией
add_executable(test_step_3_2
    test_step_3_2.cpp
)
target_link_libraries(test_step_3_2
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_3_2)

# Тест шага 3.3 - градиент с улучшенной барьерной защитой, нормализацией и верификацией
add_executable(test_step_3_3
    test_step_3_3.cpp
)
target_link_libraries(test_step_3_3
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_step_3_3)
