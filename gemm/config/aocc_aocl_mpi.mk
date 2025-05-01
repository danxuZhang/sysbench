CC = mpicc
CFLAGS = -O3 -march=native
LDFLAGS = 

LA_PATH = /scratch/libs/aocl/5.0.0/aocc
LA_INC = $(LA_PATH)/include
LA_LIB = $(LA_PATH)/lib
LA_LDFLAGS = -lblis 

USE_MPI = 1
USE_SINGLE = 0

