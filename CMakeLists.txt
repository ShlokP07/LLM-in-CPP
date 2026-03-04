cmake_minimum_required(VERSION 3.16)
project(LLMFromScratch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional OpenBLAS / BLAS for efficient GEMM.
# If you configure CMake with the vcpkg toolchain and have openblas in vcpkg.json,
# CMake will see OpenBLAS via find_package(OpenBLAS CONFIG).
find_package(OpenBLAS CONFIG QUIET)
if(NOT OpenBLAS_FOUND)
  find_package(BLAS QUIET)
endif()

if(OpenBLAS_FOUND)
  message(STATUS "OpenBLAS found (via vcpkg or system)")
  add_compile_definitions(LLM_USE_BLAS=1)
elseif(BLAS_FOUND)
  message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
  add_compile_definitions(LLM_USE_BLAS=1)
else()
  message(STATUS "No BLAS/OpenBLAS found - using naive matmul")
endif()

# Source structure
set(LLM_SOURCES
  src/tensor.cpp
  src/autograd.cpp
  src/ops.cpp
  src/nn/module.cpp
  src/nn/linear.cpp
  src/nn/embedding.cpp
  src/nn/layernorm.cpp
  src/nn/dropout.cpp
  src/nn/activation.cpp
  src/nn/softmax.cpp
  src/nn/attention.cpp
  src/nn/loss.cpp
  src/optim/sgd.cpp
  src/optim/adamw.cpp
  src/data/dataset.cpp
  src/data/dataloader.cpp
  src/utils/init.cpp
  src/utils/checkpoint.cpp
)

# Create library (Phase 1: tensor + autograd + ops + basic nn module)
add_library(llm_core STATIC
  src/core_stub.cpp
  src/tensor.cpp
  src/autograd.cpp
  src/ops.cpp
  src/nn/module.cpp
  src/nn/linear.cpp
  src/nn/layernorm.cpp
  src/nn/embedding.cpp
  src/nn/dropout.cpp
  src/nn/activation.cpp
  src/nn/softmax.cpp
  src/nn/loss.cpp
  src/optim/sgd.cpp
  src/utils/init.cpp
)

target_include_directories(llm_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenBLAS_FOUND)
  target_link_libraries(llm_core PUBLIC OpenBLAS::OpenBLAS)
elseif(BLAS_FOUND)
  target_link_libraries(llm_core PUBLIC ${BLAS_LIBRARIES})
endif()

# Main executable (placeholder for testing)
add_executable(llm_main src/main.cpp)
target_link_libraries(llm_main PRIVATE llm_core)

# Tests (optional, for grad checks etc.)
option(LLM_BUILD_TESTS "Build tests" ON)
if(LLM_BUILD_TESTS)
  enable_testing()
  add_executable(llm_tests tests/test_main.cpp)
  target_link_libraries(llm_tests PRIVATE llm_core)
  add_test(NAME llm_tests COMMAND llm_tests)
endif()
