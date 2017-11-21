CFLAGS = -g -Wall -O3
sample: sample.o bmp_util.o

clean:
	rm *.o sample
