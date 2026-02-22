NVCC = nvcc
CXXFLAGS = -O3 -Xcompiler -fopenmp

all:
	$(NVCC) $(CXXFLAGS) \
	main.cpp cpu_mc.cpp \
	gpu_mc_naive.cu \
	gpu_mc_optimized.cu \
	-o mc -lcurand

clean:
	rm -f mc