NVCC = nvcc
CXXFLAGS = -O3 -Xcompiler -fopenmp
NVFLAGS  = -O3 --use_fast_math

all:
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) \
	main.cpp cpu_mc.cpp gpu_mc_cub.cu \
	-o mc -lcurand

clean:
	rm -f mc