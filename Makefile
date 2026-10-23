.RECIPEPREFIX := >

CXX := /usr/bin/g++-12
NVCC := nvcc

GPU ?= 0
CUDA_ARCH ?= sm_75

BOOST_ROOT ?= external/boost_1_88_0
INCLUDES := -Iinclude -I$(BOOST_ROOT)

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra \
            -fno-fast-math -ffp-contract=off

NVCCFLAGS := -std=c++17 -O2 \
             -arch=$(CUDA_ARCH) -ccbin $(CXX) \
             --fmad=false \
             -Xcompiler=-fno-fast-math,-ffp-contract=off

HEADERS := $(wildcard include/*.hpp include/*.cuh)

IO := src/core/matrix_io.cpp
ANALYSIS := src/core/matrix_analysis.cpp
SYMBOLIC := src/core/spgemm_symbolic.cpp
GPU_CORE := src/core/spgemm_gpu.cu
EXACT := src/core/exact_binary.cpp

APPS := build/inspect_inputs \
        build/inspect_workload \
        build/inspect_structure \
        build/inspect_numeric \
        build/compare_cpu_gpu

TESTS := build/test_matrix_analysis \
         build/test_spgemm_symbolic \
         build/test_spgemm_numeric \
         build/test_spgemm_gpu \
         build/test_exact_binary

EXAMPLES := build/gpu_check \
            build/cpu_spgemm \
            build/gpu_spgemm \
            build/precision_order

.PHONY: all apps tests examples test test-cpu test-gpu compare_cpu_gpu

all: apps tests examples

apps: $(APPS)

tests: $(TESTS)

examples: $(EXAMPLES)

compare_cpu_gpu: build/compare_cpu_gpu

build:
>mkdir -p build

build/inspect_inputs: src/apps/inspect_inputs.cpp $(IO) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/inspect_workload: src/apps/inspect_workload.cpp $(IO) $(ANALYSIS) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/inspect_structure: src/apps/inspect_structure.cpp $(IO) $(ANALYSIS) $(SYMBOLIC) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/inspect_numeric: src/apps/inspect_numeric.cpp $(IO) $(SYMBOLIC) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/compare_cpu_gpu: src/apps/compare_cpu_gpu.cu $(IO) $(SYMBOLIC) $(GPU_CORE) $(HEADERS) | build
>$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cpp %.cu,$^) -o $@

build/test_matrix_analysis: src/tests/test_matrix_analysis.cpp $(IO) $(ANALYSIS) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/test_spgemm_symbolic: src/tests/test_spgemm_symbolic.cpp $(IO) $(SYMBOLIC) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/test_spgemm_numeric: src/tests/test_spgemm_numeric.cpp $(IO) $(SYMBOLIC) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/test_spgemm_gpu: src/tests/test_spgemm_gpu.cu $(IO) $(SYMBOLIC) $(GPU_CORE) $(HEADERS) | build
>$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cpp %.cu,$^) -o $@

build/test_exact_binary: src/tests/test_exact_binary.cpp $(EXACT) $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/gpu_check: src/examples/gpu_check.cu $(HEADERS) | build
>$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cu,$^) -o $@

build/cpu_spgemm: src/examples/cpu_spgemm.cpp $(HEADERS) | build
>$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/gpu_spgemm: src/examples/gpu_spgemm.cu $(HEADERS) | build
>$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cu,$^) -o $@

build/precision_order: src/examples/precision_order.cu $(HEADERS) | build
>$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cu,$^) -o $@

test-cpu: build/test_matrix_analysis build/test_spgemm_symbolic build/test_spgemm_numeric build/test_exact_binary
>./build/test_matrix_analysis
>./build/test_spgemm_symbolic
>./build/test_spgemm_numeric
>./build/test_exact_binary

test-gpu: build/test_spgemm_gpu
>CUDA_VISIBLE_DEVICES=$(GPU) ./build/test_spgemm_gpu

test: test-cpu test-gpu
