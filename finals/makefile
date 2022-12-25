.PHONY: all run clean

VER = 0
CC = 61
NVCC = nvcc
ARC = -arch=sm_$(CC)
CUDAARGS = -Xptxas -O3,-v -Xcompiler -O2
RUNARGS :=

GARBAGE := *.o *.obj *.exe *.exp *.lib
GARBAGE_SUBDIR := $(foreach PAT,$(GARBAGE),$(addprefix ./*/,$(PAT)))

all: utils.o ./ver_$(VER)/sc-device_v$(VER).o sc-host.o main_v$(VER).o seam-carving_v$(VER)

utils.o: utils.cuh utils.cu
	$(NVCC) $(ARC) -c utils.cu -o utils.o

./ver_$(VER)/sc-device_v$(VER).o: utils.cuh ./ver_$(VER)/device.cuh ./ver_$(VER)/sc-device.cu
	$(NVCC) $(ARC) $(CUDAARGS) -c -I. ./ver_$(VER)/sc-device.cu -o ./ver_$(VER)/sc-device_v$(VER).o

sc-host.o: utils.cuh host.cuh sc-host.cu
	$(NVCC) $(ARC) -O2 -c sc-host.cu -o sc-host.o

main_v$(VER).o: utils.cuh host.cuh ./ver_$(VER)/device.cuh main.cu
	$(NVCC) $(ARC) -c -I. -I./ver_$(VER) main.cu -o main_v$(VER).o

seam-carving_v$(VER): main_v$(VER).o utils.o sc-host.o ./ver_$(VER)/sc-device_v$(VER).o
	$(NVCC) $(ARC) main_v$(VER).o utils.o sc-host.o ./ver_$(VER)/sc-device_v$(VER).o -o seam-carving_v$(VER)

run:
	make VER=$(VER) CC=$(CC) && ./seam-carving_v$(VER) $(RUNARGS)

clean:
	-rm $(GARBAGE)
	-rm $(GARBAGE_SUBDIR)