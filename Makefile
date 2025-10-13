CC := $(shell command -v hipcc 2>/dev/null || echo g++)
CFLAGS = --std=c++17 -lm

# GPU architecture selection: override via `make GPU_ARCH=gfx1151` or env var.
# When set to `auto` (default), try to detect via hipconfig; fallback to gfx90a.
GPU_ARCH ?= auto
ifeq ($(GPU_ARCH),auto)
  DETECTED_ARCH := $(shell hipconfig 2>/dev/null | grep -Eo 'gfx[0-9a-z]+' | head -n1)
  ifeq ($(strip $(DETECTED_ARCH)),)
    DETECTED_ARCH := $(shell rocminfo 2>/dev/null | grep -Eo 'gfx[0-9a-z]+' | head -n1)
  endif
  ifeq ($(strip $(DETECTED_ARCH)),)
    DETECTED_ARCH := gfx90a
  endif
  AMDGPU_FLAGS := --offload-arch=$(DETECTED_ARCH)
else
  # Allow passing features in GPU_ARCH, e.g. GPU_ARCH="gfx1151:wavefrontsize64"
  AMDGPU_FLAGS := $(addprefix --offload-arch=,$(GPU_ARCH))
endif

ifneq ($(CC),g++)
CFLAGS += $(AMDGPU_FLAGS)
endif

# WMMA path (gfx11+): include rocWMMA headers and enable WMMA code paths.
# Also disable getp until all kernels are ported.
ifeq ($(GPU_ARCH),auto)
  ifneq (,$(findstring gfx11,$(DETECTED_ARCH)))
    CFLAGS += -DOSS_USE_WMMA=1 -IrocWMMA/library/include
  endif
else
  ifneq (,$(findstring gfx11,$(GPU_ARCH)))
    CFLAGS += -DOSS_USE_WMMA=1 -IrocWMMA/library/include
  endif
endif

CPP_FILES = src/run.cpp src/tokenizer.cpp

.PHONY: run
run: $(CPP_FILES) tokenizer-bin
	$(CC) -g -O0 -o build/run $(CPP_FILES)

rundebug: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -g -o build/run $(CPP_FILES)

.PHONY: runfast
runfast: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -o build/run $(CPP_FILES)

.PHONY: runomp
runomp: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -fopenmp -march=native $(CPP_FILES) -o build/run

.PHONY: decode
decode: src/decode.cpp src/tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -O3 src/decode.cpp src/tokenizer.cpp -o build/decode

.PHONY: wmma-test
wmma-test: tests/wmma_gemm_test.cpp src/hip/BLAS.hip src/hip/gemms/gemm_logits.hip
	$(CC) $(CFLAGS) -DTESTING -DOSS_USE_WMMA=1 -IrocWMMA/library/include -O3 \
	  tests/wmma_gemm_test.cpp src/hip/gemms/gemm_logits.hip -o build/wmma_test

.PHONY: wmma-gemm-bias-test
wmma-gemm-bias-test: tests/wmma_gemm_bias_test.cpp \
    src/hip/gemms/gemm_qkv.hip src/hip/gemms/gemm_o.hip src/hip/gemms/gemm_router.hip
	$(CC) $(CFLAGS) -DTESTING -DOSS_USE_WMMA=1 -IrocWMMA/library/include -O3 \
	  tests/wmma_gemm_bias_test.cpp \
	  src/hip/gemms/gemm_qkv.hip src/hip/gemms/gemm_o.hip src/hip/gemms/gemm_router.hip \
	  -o build/wmma_gemm_bias_test

.PHONY: tokenizer-bin
tokenizer-bin: tools/export_tokenizer_bin.py
	python3 tools/export_tokenizer_bin.py -o build/tokenizer.bin

.PHONY: tokenizer-test
tokenizer-test: tests/test_tokenizer.cpp src/tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -DTESTING -O3 tests/test_tokenizer.cpp src/tokenizer.cpp -o build/test_tokenizer

.PHONY: clean
clean:
	rm -f build/run build/decode build/tokenizer.bin build/test_tokenizer
LDLIB ?= /opt/rocm/lib
RUN_ENV = LD_LIBRARY_PATH=$(LDLIB):$${LD_LIBRARY_PATH}
TIMEOUT ?= 60

.PHONY: quick-tests
quick-tests: wmma-test wmma-gemm-bias-test
	@echo "Running wmma_test (timeout $(TIMEOUT)s)"
	- timeout $(TIMEOUT) env $(RUN_ENV) ./build/wmma_test || true
	@echo "Running wmma_gemm_bias_test (timeout $(TIMEOUT)s)"
	- timeout $(TIMEOUT) env $(RUN_ENV) ./build/wmma_gemm_bias_test || true
