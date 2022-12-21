ARC = -arch=sm_61

NVCC = nvcc
GPP = g++

seam-carving-device.o: seam-carving-device.h seam-carving-device.cu
	$(NVCC) $(ARC) -c seam-carving-device.cu -o seam-carving-device.o

seam-carving-host.o: seam-carving-host.h seam-carving-host.cpp
	$(GPP) -O2 -c seam-carving-host.cpp -o seam-carving-host.o