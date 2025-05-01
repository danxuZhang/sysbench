CC = icx
CFLAGS = -qopenmp -O2 -fPIC

STREAM_FLAGS = -DSTREAM_ARRAY_SIZE=10000000 -DNTIMES=10 

SRC=stream.c

stream:
	$(CC) $(CFLAGS) $(STREAM_FLAGS) $(SRC) -o stream

