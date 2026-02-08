add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example mixed_approximation)

add_executable(reading_from_files reading_from_files.cpp)
target_link_libraries(reading_from_files mixed_approximation)

# Копирование данных файлов в директорию сборки для примера reading_from_files
add_custom_command(TARGET reading_from_files POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    $<TARGET_FILE_DIR:reading_from_files>/data
    COMMENT "Copying data files to output directory"
)
