CPP=nvcc

FLAGS := --std=c++11 -O3 -arch=sm_70
CFLAGS := $(OPT) -Wextra,-Wall
MODULE := conv1 conv2 class1 class2

.PHONY: all clean

all: $(MODULE)

conv1: convolution.cu
	$(CPP) $^ $(FLAGS) -Xcompiler $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=64 -DNn=64 -DTii=32 -DTi=16 -DTnn=32 -DTn=13 -DTx=8 -DTy=9 -DBatch=16

conv2: convolution.cu
	$(CPP) $^ $(FLAGS) -Xcompiler $(CFLAGS) -o $@ -DNx=14 -DNy=14 -DKx=3 -DKy=3 -DNi=512 -DNn=512 -DTii=32 -DTi=16 -DTnn=32 -DTn=1 -DTx=3 -DTy=4 -DBatch=16

class1: classifier.cu
	$(CPP) $^ $(FLAGS) -Xcompiler $(CFLAGS) -o $@ -DNi=25088 -DNn=4096 -DTii=512 -DTi=64 -DTnn=32 -DTn=16 -DBatch=16

class2: classifier.cu
	$(CPP) $^ $(FLAGS) -Xcompiler $(CFLAGS) -o $@ -DNi=4096 -DNn=1024 -DTii=32 -DTi=32 -DTnn=32 -DTn=16 -DBatch=16

clean:
	@rm -f $(MODULE) 