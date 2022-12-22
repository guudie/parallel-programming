.PHONY: all clean

NVCC = nvcc
ARC = -arch=sm_61

all: sc-device.o sc-host.o main.o seam-carving

sc-device.o: device.cuh sc-device.cu
	$(NVCC) $(ARC) -c sc-device.cu -o sc-device.o

sc-host.o: host.cuh sc-host.cu
	$(NVCC) $(ARC) -O2 -c sc-host.cu -o sc-host.o

main.o: sc-device.o sc-host.o
	$(NVCC) $(ARC) -Xptxas -O3,-v -Xcompiler -O2 -c sc-device.o sc-host.o -o main.o

seam-carving: main.o
	$(NVCC) $(ARC) main.o -o seam-carving

clean:
	rm *.o *.exe