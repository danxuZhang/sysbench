CONFIG ?= default

include config/$(CONFIG).mk

SRCS = ./gemm.c

all: gemm

gemm: 
	$(CC) $(SRCS) \
		-DUSE_MPI=$(USE_MPI) -DUSE_SINGLE=$(USE_SINGLE) \
		$(CFLAGS) $(LDFLAGS) \
		-I$(LA_INC) -L$(LA_LIB) $(LA_LDFLAGS) \
		-o ./bin/gemm_$(CONFIG)

clean:
	rm -f $(OUT) 

.PHONY: all gemm clean

