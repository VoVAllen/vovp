cmake_minimum_required(VERSION 3.4...3.18)
project(vovp)
execute_process(COMMAND python3 -c "import sys; print(sys.executable)"
                      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      OUTPUT_VARIABLE PYTHON_EXECUTABLE
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
add_subdirectory(third_party/pybind11)
pybind11_add_module(vovp src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(vovp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
