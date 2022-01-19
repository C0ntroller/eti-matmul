CC = gcc -Wall

all: matmul matmulO1 matmulO2 matmulO3 matmulOMP

matmul: matmul.c
	$(CC) matmul.c -o bin/matmul

matmulO1: matmul.c
	$(CC) matmul.c -o bin/matmulO1 -O1

matmulO2: matmul.c
	$(CC) matmul.c -o bin/matmulO2 -O2

matmulO3: matmul.c
	$(CC) matmul.c -o bin/matmulO3 -O3

matmulOMP: matmul.c
	$(CC) matmul.c -o bin/matmulOMP -fopenmp

.PHONY: clean
clean:
	rm -f *.o bin/matmul bin/matmulO1 bin/matmulO2 bin/matmulO3 bin/matmulOMP