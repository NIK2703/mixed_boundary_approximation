cmake_minimum_required(VERSION 3.10)
project(MixedApproximation LANGUAGES CXX)

# Настройка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опции сборки
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Поиск yaml-cpp для поддержки YAML конфигураций
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    # Если yaml-cpp не найден, используем встроенный простой парсер
    message(STATUS "yaml-cpp not found, will use simple config parser")
    add_definitions(-DUSE_SIMPLE_PARSER)
endif()

# Добавляем библиотеку
add_library(mixed_approximation
    src/polynomial.cpp
    src/validator.cpp
    src/functional.cpp
    src/optimizer.cpp
    src/mixed_approximation.cpp
    src/config_reader.cpp
)

target_include_directories(mixed_approximation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Линковка yaml-cpp если найден
if(yaml-cpp_FOUND)
    target_link_libraries(mixed_approximation PRIVATE yaml-cpp)
    target_compile_definitions(mixed_approximation PRIVATE HAS_YAML_CPP)
endif()

# Настройки компилятора для разных платформ
if(MSVC)
    target_compile_options(mixed_approximation PRIVATE /W4)
else()
    target_compile_options(mixed_approximation PRIVATE -Wall -Wextra -pedantic)
endif()

# Установка библиотеки
install(TARGETS mixed_approximation
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

# Сборка примеров
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Сборка тестов
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Создание пакета
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MixedApproximationConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)
