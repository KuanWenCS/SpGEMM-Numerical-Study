
ifeq ($(origin CXX), default)
CXX := $(if $(filter Darwin,$(shell uname -s)),clang++,/usr/bin/g++-12)
endif
NVCC ?= nvcc
HAS_CUDA := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)

GPU ?= 0
CUDA_ARCH ?= sm_75

BOOST_ROOT ?= external/boost_1_88_0
INCLUDES := -Iinclude -I$(BOOST_ROOT)

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra \
            -fno-fast-math -ffp-contract=off -pthread

NVCCFLAGS := -std=c++17 -O2 \
             -arch=$(CUDA_ARCH) -ccbin $(CXX) \
             --fmad=false --ftz=false \
             -Xcompiler=-fno-fast-math,-ffp-contract=off,-pthread

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

all: cpu $(if $(filter 1,$(HAS_CUDA)),cuda)

cpu: build/inspect_inputs build/inspect_workload build/inspect_structure build/inspect_numeric build/study build/cpu_spgemm

cuda: build/study_cuda build/compare_cpu_gpu build/gpu_check build/gpu_spgemm build/precision_order

apps: $(APPS)

tests: $(TESTS)

examples: $(EXAMPLES)

compare_cpu_gpu: build/compare_cpu_gpu

build:
	mkdir -p build

build/inspect_inputs: src/apps/inspect_inputs.cpp $(IO) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/inspect_workload: src/apps/inspect_workload.cpp $(IO) $(ANALYSIS) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/inspect_structure: src/apps/inspect_structure.cpp $(IO) $(ANALYSIS) $(SYMBOLIC) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/inspect_numeric: src/apps/inspect_numeric.cpp $(IO) $(SYMBOLIC) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/compare_cpu_gpu: src/apps/compare_cpu_gpu.cu $(IO) $(SYMBOLIC) $(GPU_CORE) $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cpp %.cu,$^) -o $@

build/test_matrix_analysis: src/tests/test_matrix_analysis.cpp $(IO) $(ANALYSIS) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/test_spgemm_symbolic: src/tests/test_spgemm_symbolic.cpp $(IO) $(SYMBOLIC) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/test_spgemm_numeric: src/tests/test_spgemm_numeric.cpp $(IO) $(SYMBOLIC) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/test_spgemm_gpu: src/tests/test_spgemm_gpu.cu $(IO) $(SYMBOLIC) $(GPU_CORE) $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cpp %.cu,$^) -o $@

build/test_exact_binary: src/tests/test_exact_binary.cpp $(EXACT) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/gpu_check: src/examples/gpu_check.cu $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cu,$^) -o $@

build/cpu_spgemm: src/examples/cpu_spgemm.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp,$^) -o $@

build/gpu_spgemm: src/examples/gpu_spgemm.cu $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cu,$^) -o $@

build/precision_order: src/examples/precision_order.cu $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(filter %.cu,$^) -o $@

test-cpu: build/test_matrix_analysis build/test_spgemm_symbolic build/test_spgemm_numeric build/test_exact_binary build/test_experiment build/study
	./build/test_matrix_analysis
	./build/test_spgemm_symbolic
	./build/test_spgemm_numeric
	./build/test_exact_binary
	./build/test_experiment
	python3 scripts/test_integration.py

test-gpu: build/test_spgemm_gpu build/study_cuda
	CUDA_VISIBLE_DEVICES=$(GPU) ./build/test_spgemm_gpu
	CUDA_VISIBLE_DEVICES=$(GPU) python3 scripts/test_integration.py --cuda

test: test-cpu $(if $(filter 1,$(HAS_CUDA)),test-gpu)

.PHONY: cpu cuda test-integration

STUDY_OBJECTS := build/obj/matrix_io.o build/obj/matrix_analysis.o build/obj/spgemm_symbolic.o build/obj/exact_binary.o build/obj/experiment_cpu.o build/obj/error_analysis.o

build/obj:
	mkdir -p build/obj

build/obj/%.o: src/core/%.cpp $(HEADERS) | build/obj
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

build/run_study_cpu.o: src/apps/run_study.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DSTUDY_BUILD_FLAGS='"$(CXXFLAGS)"' -c $< -o $@

build/run_study_cuda.o: src/apps/run_study.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DSTUDY_CUDA -DSTUDY_CUDA_FLAGS='"$(NVCCFLAGS)"' -DSTUDY_BUILD_FLAGS='"$(CXXFLAGS)"' -c $< -o $@

build/spgemm_gpu.o: src/core/spgemm_gpu.cu $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

build/study: build/run_study_cpu.o $(STUDY_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/study_cuda: build/run_study_cuda.o $(STUDY_OBJECTS) build/spgemm_gpu.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

build/test_experiment: src/tests/test_experiment.cpp $(STUDY_OBJECTS) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(filter %.cpp %.o,$^) -o $@

test-integration: build/study
	python3 scripts/test_integration.py
