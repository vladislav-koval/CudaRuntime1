#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <algorithm>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 1024 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM_VALS THREADS*BLOCKS

using namespace std;

void print_elapsed(clock_t start, clock_t stop) {
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);
}

void array_print(int* arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		printf("%1.3f ", arr[i]);
	}
	printf("\n");
}

void array_fill(int* arr, int length) {
	srand(time(NULL));
	int i;
	for (i = 0; i < length; ++i) {
		arr[i] = rand();
	}
}

bool comparison_arrays(int* arr1, int* arr2, int length) {
	for (int i = 0; i < length; i++) {
		if (arr1[i] != arr2[i]) {
			return false;
		}
	}
	return true;
}

int* get_copy_array(int* sourse, int length) {
	int* dest = new int[length];

	for (int i = 0; i < length; i++) {
		dest[i] = sourse[i];
	}
	return dest;
}

int power_ceil(int x) {
	if (x <= 1) return 1;
	int power = 2;
	x--;
	while (x >>= 1) power <<= 1;
	return power;
}

__global__ void bitonic_sort_step(int* dev_values, int j, int k) {
	unsigned int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;

	if ((ixj) > i) {
		if ((i & k) == 0) {
			/* Sort ascending */
			if (dev_values[i] > dev_values[ixj]) {
				/* exchange(i,ixj); */
				int temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
		if ((i & k) != 0) {
			/* Sort descending */
			if (dev_values[i] < dev_values[ixj]) {
				/* exchange(i,ixj); */
				int temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
	}
}

void bitonic_sort(int* values) {
	int* dev_values;
	size_t size = NUM_VALS * sizeof(int);

	cudaMalloc((void**)&dev_values, size);
	cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

	dim3 blocks(BLOCKS, 1);
	dim3 threads(THREADS, 1);

	int j, k;
	for (k = 2; k <= NUM_VALS; k <<= 1) {
		for (j = k >> 1; j > 0; j = j >> 1) {
			bitonic_sort_step <<<blocks, threads>>> (dev_values, j, k);
		}
	}
	cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
}

bool is_bitonic(int*v, int length) {
	bool was_decreasing = v[length - 1] > v[0];
	int num_inflections = 0;
	for (int i = 0; i < length && num_inflections <= 2; i++) {
		bool is_decreasing = v[i] > v[(i + 1) % length];
		// Check if this element and next one are an inflection.
		if (was_decreasing != is_decreasing) {
			num_inflections++;
			was_decreasing = is_decreasing;
		}
	}

	return 2 == num_inflections;
}

int main(void)
{
	clock_t start, stop;

	int length = 0;
	cout << "Enter length of the array: ";
	cin >> length;

	int* values = (int*)malloc(NUM_VALS * sizeof(int));
	array_fill(values, NUM_VALS);
	int* temp = get_copy_array(values, NUM_VALS);


	sort(temp, temp + NUM_VALS);


	start = clock();
	bitonic_sort(values);
	stop = clock();

	cout << "is_bitonic: " << is_bitonic(values, NUM_VALS) << endl;;
	cout << "is equals: " << comparison_arrays(values, temp, NUM_VALS) << endl;
	print_elapsed(start, stop);
}
