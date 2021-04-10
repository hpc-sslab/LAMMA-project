#commnet for makefile
MM := mm

SOURCES	:= $(%.cu)
OBJECTS	:= $(%.o)						   
#OBJECTS	:= $(patsubst src%,obj%, $(patsubst %.cu,%.cu.o, $(patsubst %.cpp,%.o, $(patsubst %.c,%.o, $(SOURCES)))))						   
#INCLUDE := -I/usr/local/cuda/include
#LIBPATH := -L/usr/local/cuda/lib
#LIBS 	:=	-lcudart -lGL -lglut 

NVCC := /usr/local/cuda-7.0/bin/nvcc

#Flags
CCFLAGS := -O3 -G -g
GENCODE_FLAGS := -gencode arch=compute_35,code=sm_35
NVCCFLAGS := $(CCFLAGS) $(GENCODE_FLAGS) --compile --relocatable-device-code=false
LINKFLAGS := $(GENCODE_FLAGS) --relocatable-device-code=false --cudart static -link

#Targets

$(MM) : mm_s_d.o
	@printf "#Building $(MM)...\n"
	$(NVCC) $(LINKFLAGS) $< -o $@

mm_s_d.o : mm_s_d.cu 
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean
	
clean:
	rm -rf *.o
	rm -f $(MM)