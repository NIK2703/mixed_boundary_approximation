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

# Интеграционные тесты
add_executable(test_integration
    integration/test_integration.cpp
)
target_link_libraries(test_integration
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_integration)

# Объединённый тестовый запуск (опционально)
add_executable(test_all
    polynomial/test_polynomial.cpp
    functional/test_functional.cpp
    validator/test_validator_basic.cpp
    validator/test_validator_advanced.cpp
    decomposition/test_decomposition.cpp
    integration/test_integration.cpp
)
target_link_libraries(test_all
    mixed_approximation
    GTest::gtest_main
)
gtest_discover_tests(test_all)
