COMPUTECAP=52
LDFLGS=$(CUDA)/lib64

rcog: rcog.cu rcog_read_gupp.cu rcog_spectro.cu rcog_stat.cu
	nvcc -arch=compute_$(COMPUTECAP) -g -o rcog  rcog.cu rcog_read_gupp.cu rcog_spectro.cu rcog_stat.cu -L$(LDFLGS) -lcufft

clean:
	rm -f rcog

info:
	@echo "CUDA=" $(CUDA)
	@echo "LD_LIBRARY_PATH=" $(LD_LIBRARY_PATH)

# works for srbs-hpc1 and flag2
