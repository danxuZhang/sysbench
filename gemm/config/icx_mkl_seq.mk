CC = icx
CFLAGS = -O2 -g
LDFLAGS = 

LA_PATH = $(MKLROOT)
LA_INC = $(LA_PATH)/include
LA_LIB = $(LA_PATH)/lib
LA_LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

USE_MPI = 0
USE_SINGLE = 0

