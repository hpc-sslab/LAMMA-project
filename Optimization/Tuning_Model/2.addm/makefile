#commnet for makefile
AM := am

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

$(AM) : addm_s_d.o
	@printf "#Building $(AM)...\n"
	$(NVCC) $(LINKFLAGS) $< -o $@

addm_s_d.o : addm_s_d.cu 
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean
	
clean:
	rm -rf *.o
	rm -f $(AM)
