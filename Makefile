# Compiler
# --------
CC=g++
#CC= g++
#CC=icpc
#CC=mpic++

SOURCES=main.cpp

# Compiler flags
# -------------------------------------------------------------------------
CFLAGS = -O0 #-fno-stack-protector
#CFLAGS=-g -O3 
#CFLAGS=-fopenmp -g -O3

# Linker flags
# ------------
LDFLAGS=
INCLUDES= -I ${SLEPC_DIR}/include -I ${PETSC_DIR}/include -I ${PETSC_DIR}/${PETSC_ARCH}/include

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=ED

#all: $(SOURCES) $(EXECUTABLE)
default: ED
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common


#$(EXECUTABLE): $(OBJECTS)
#	${CC} -o $(OBJECTS) $(SOURCES) $(INCLUDES) #${PETSC_LIB} ${SLEPC_LIB} 
#	$(CC) $(OBJECTS) $(LDFLAGS) -o $@  

ED: main.o chkopts
	-${CLINKER} -o ED main.o ${SLEPC_EPS_LIB}
	${RM} main.o

#clean:
#	rm $(OBJECTS) 

#.cpp.o:
#	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ 



#default: ex1
#include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
#ex1: ex1.o chkopts
#        -${CLINKER} -o ex1 ex1.o ${SLEPC_EPS_LIB}
#        ${RM} ex1.o
#ex1f: ex1f.o chkopts
#        -${FLINKER} -o ex1f ex1f.o ${SLEPC_EPS_LIB}
#        ${RM} ex1f.o


