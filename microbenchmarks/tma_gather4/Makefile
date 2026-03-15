NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda

NVCCFLAGS = -std=c++20 -gencode arch=compute_100a,code=sm_100a -O3 --expt-relaxed-constexpr
INCLUDES  = -I$(CUDA_PATH)/include
LDFLAGS   = -lcuda

TARGET = tma_gather4_bench

HEADERS = common.cuh tensor_map_utils.cuh index_patterns.cuh gather4_kernels.cuh

.PHONY: all clean sass

all: $(TARGET)

$(TARGET): main.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ main.cu $(LDFLAGS)

# Dump SASS for instruction inspection
sass: $(TARGET)
	cuobjdump --dump-sass $(TARGET) > $(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).sass
