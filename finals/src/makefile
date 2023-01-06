.PHONY: all direct run clean

VER = 1
CC = 61
NVCC = nvcc
ARC = -arch=sm_$(CC)
CUDA_ARGS = -Xptxas -O3,-v -Xcompiler -O2
RUN_ARGS :=

GARBAGE := *.o *.exe *.exp *.lib
GARBAGE_SUBDIR := $(foreach PAT,$(GARBAGE),$(addprefix ./*/,$(PAT)))

DEFAULT_RUN_ON = HOST DEVICE
RUN_ON =
DEFINE_ARGS =
ifneq ($(RUN_ON),)
VALID_RUN_ON = $(filter $(DEFAULT_RUN_ON),$(RUN_ON))
ifneq ($(VALID_RUN_ON),)
DONT_RUN_ON = $(filter-out $(VALID_RUN_ON),$(DEFAULT_RUN_ON))
DEFINE_ARGS = $(foreach PAT,$(DONT_RUN_ON),$(addprefix -DDONT_RUN_ON_,$(PAT)))
endif
endif

all: utils.o ./ver_$(VER)/sc-device_v$(VER).o sc-host.o main_v$(VER).o seam-carving_v$(VER)

direct:
	$(NVCC) $(ARC) $(CUDA_ARGS) $(DEFINE_ARGS) -O2 -I. -I./ver_$(VER) main.cu utils.cu sc-host.cu ./ver_$(VER)/sc-device.cu -o seam-carving_v$(VER)

utils.o: utils.cuh utils.cu
	$(NVCC) $(ARC) -c utils.cu -o utils.o

./ver_$(VER)/sc-device_v$(VER).o: utils.cuh ./ver_$(VER)/device.cuh ./ver_$(VER)/sc-device.cu
	$(NVCC) $(ARC) $(CUDA_ARGS) -c -I. ./ver_$(VER)/sc-device.cu -o ./ver_$(VER)/sc-device_v$(VER).o

sc-host.o: utils.cuh host.cuh sc-host.cu
	$(NVCC) $(ARC) -O2 -c sc-host.cu -o sc-host.o

main_v$(VER).o: utils.cuh host.cuh ./ver_$(VER)/device.cuh main.cu
	$(NVCC) $(ARC) $(DEFINE_ARGS) -c -I. -I./ver_$(VER) main.cu -o main_v$(VER).o

seam-carving_v$(VER): main_v$(VER).o utils.o sc-host.o ./ver_$(VER)/sc-device_v$(VER).o
	$(NVCC) $(ARC) main_v$(VER).o utils.o sc-host.o ./ver_$(VER)/sc-device_v$(VER).o -o seam-carving_v$(VER)

run:
	make VER=$(VER) CC=$(CC) && ./seam-carving_v$(VER) $(RUN_ARGS)

clean:
	-rm $(GARBAGE)
	-rm $(GARBAGE_SUBDIR)