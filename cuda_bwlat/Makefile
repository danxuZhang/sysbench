NVCC=nvcc
NVCCFLAGS=-I. -O2

all: bandwidthTest p2pBandwidthLatencyTest

p2pBandwidthLatencyTest:
	$(NVCC) $(NVCCFLAGS) p2pBandwidthLatencyTest.cu -o ./bin/p2pBandwidthLatencyTest

bandwidthTest:
	$(NVCC) $(NVCCFLAGS) bandwidthTest.cu -o ./bin/bandwidthTest

clean:
	rm bin/*
