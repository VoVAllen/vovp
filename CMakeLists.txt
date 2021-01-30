cmake_minimum_required(VERSION 3.4...3.18)
project(vovp)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
execute_process(COMMAND python3 -c "import sys; print(sys.executable)"
                      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      OUTPUT_VARIABLE PYTHON_EXECUTABLE
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
add_subdirectory(third_party/pybind11)

# set(CMAKE_BUILD_TYPE Release)
# include(DefineArrowOptions)
# add_subdirectory(third_party/arrow/cpp)

find_package(Arrow REQUIRED)

pybind11_add_module(vovp MODULE src/main.cpp)
target_link_libraries(vovp PRIVATE arrow_shared)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
set_target_properties(vovp PROPERTIES OUTPUT_NAME "_vovp")
target_compile_definitions(vovp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
