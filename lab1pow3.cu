#include <stdio.h>
#include <iostream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <chrono> > 
#include <iomanip>
#include <limits>

using namespace std::chrono;

#define  N          100000000
#define  BLOCK_SIZE 1024

float      hArray[N];
float* dArray;
int        blocks;


void prologue(void) {
	memset(hArray, 0, sizeof(hArray));
	for (int i = 0; i < N; i++) {
		hArray[i] = i + 1;
	}
	cudaMalloc((void**)&dArray, sizeof(hArray));
	cudaMemcpy(dArray, hArray, sizeof(hArray), cudaMemcpyHostToDevice);
}

void epilogue(void) {
	cudaMemcpy(hArray, dArray, sizeof(hArray), cudaMemcpyDeviceToHost);
	cudaFree(dArray);
}


// Kernel
__global__ void pow3(float* A) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	//A[0] = 1 * 1 * 1 + 1 * 1 + 1 = 3
	//A[1] = 3 * 3 * 3 + 3 * 3 + 3 = 27 + 9 + 3 = 39
	if (x < N)
	{
		A[x] = A[x] * A[x] * A[x] + A[x] * A[x] + A[x];
	}
}


/**
 * Host main routine
 */
int main(void)
{
	std::cout << "JG::Starting program which uses GPu and CPU to calculate polynomial for <" << N << "> values" << std::endl;
	std::cout << "Warning1: prologue() is not included in the GPU calculation time " << std::endl;
	std::cout << "Warning2: prologue() is not included in the CPU calculation time " << std::endl;

	int indexToBePrinted = N - 30;
	int  devCnt;
	cudaGetDeviceCount(&devCnt);
	if (devCnt == 0)
	{
		perror("No CUDA devices available -- exiting.");
		return 1;
	}

	prologue();

	std::cout << "~~~~~~~~~~GPU START" << " no. of blocks : <" << BLOCK_SIZE << ">" << std::endl;
	auto gpuCalculationTimeStart = high_resolution_clock::now();
	blocks = N / BLOCK_SIZE;
	if (N % BLOCK_SIZE)
	{
		blocks++;
	}

	//Obliczanie wartosci pewnego wielomianiu dla n elementow wektora
	pow3 << <blocks, BLOCK_SIZE >> > (dArray);
	cudaThreadSynchronize();
	epilogue();
	auto gpuCalculationTimeStop = high_resolution_clock::now();
	auto gpuDuration = duration_cast<milliseconds>(gpuCalculationTimeStop - gpuCalculationTimeStart);
	std::cout << "~~~~~~~~~~GPU STOP" << std::endl;
	std::cout << "~~~~~~~~~~Duration GPU: <" << gpuDuration.count() << "> [ms]" << std::endl;
	std::cout << std::setprecision(20) << std::showpoint;
	std::cout << indexToBePrinted << " nth Value calculated from GPU: <" << hArray[indexToBePrinted] << ">" << std::endl;
	/////////////////// GPU GPU GPU /////////////////// GPU GPU GPU /////////////////// GPU GPU GPU

	/////////////////// CPU CPU CPU/////////////////// CPU CPU CPU/////////////////// CPU CPU CPU
	prologue();

	std::cout << "~~~~~~~~~~CPU START" << std::endl;
	auto cpuCalculationTimeStart = high_resolution_clock::now();
	//Obliczanie wartosci pewnego wielomianiu dla n elementow wektora
	for (int i = 0; i < N - 1; i++)
	{
		hArray[i] = hArray[i] * hArray[i] * hArray[i] + hArray[i] * hArray[i] + hArray[i];
	}

	auto cpuCalculationTimeStop = high_resolution_clock::now();
	auto cpuDuration = duration_cast<milliseconds>(cpuCalculationTimeStop - cpuCalculationTimeStart);
	std::cout << "~~~~~~~~~~CPU END" << std::endl;
	std::cout << "~~~~~~~~~~Duration CPU: <" << cpuDuration.count() << "> [ms]" << std::endl;
	std::cout << std::setprecision(20) << std::showpoint;
	std::cout << indexToBePrinted << " nth Value calculated from CPU: <" << hArray[indexToBePrinted] << ">" << std::endl;
	epilogue();
	return 0;
}

