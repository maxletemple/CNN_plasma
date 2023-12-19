#set gcc path ./gcc_mips/gnu_toolchain-1.5.4-Linux-i686/bin

CC = gcc
CFLAGS = -std=c99 
LDFLAGS =

SOURCES = $(wildcard src/*)

all: ./main

./main: src/main.c src/cnn.c src/cnn.h src/fixed.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^
	./main

test_fixed: ./fixed
	./fixed

./fixed: src/test_fixed.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

convert_weights: ./conv_weights
	./conv_weights

./conv_weights: src/convert_weights.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f ./main ./fixed ./conv_weights

