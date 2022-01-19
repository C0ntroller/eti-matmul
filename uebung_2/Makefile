CC = gcc -Wall

all: matmul matmulO1 matmulO2 matmulO3 matmulOMP

matmul: matmul.c
	$(CC) matmul.c -o matmul

matmulO1: matmul.c
	$(CC) matmul.c -o matmulO1 -O1

matmulO2: matmul.c
	$(CC) matmul.c -o matmulO2 -O2

matmulO3: matmul.c
	$(CC) matmul.c -o matmulO3 -O3

matmulOMP: matmul.c
	$(CC) matmul.c -o matmulOMP -fopenmp

.PHONY: clean
clean:
	rm -f *.o matmul matmulO1 matmulO2 matmulO3 matmulOMP