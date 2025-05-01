CC = mpicc
CFLAGS = -O3 -march=native
LDFLAGS = 

LA_PATH = ${MKLROOT}
LA_INC = $(LA_PATH)/include
LA_LIB = $(LA_PATH)/lib
LA_LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

USE_MPI = 1
USE_SINGLE = 0

