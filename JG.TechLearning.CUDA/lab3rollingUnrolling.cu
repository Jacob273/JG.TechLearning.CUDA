#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
//For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <chrono> 
#include <iomanip>
#include <math.h>
#include "lab3rollingUnrolling.h"
#include <chrono> 
#include "jakubgmur-shared.h"



#define  BLOCK_SIZE 32

using namespace std::chrono;


int* Convert2Dto1D(int**& tab, int width, int height)
{
    int sizeOfArray = width * height;

    int* output = (int*)malloc(sizeOfArray * sizeof(int));
    memset(output, 0, sizeof(output));


    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            output[(j * width) + i] = tab[i][j];
        }
    }

    return output;
}

void FillArrayWithZeroes(int rows, int cols, int** tab)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            tab[i][j] = 0;
        }
    }
}

void PrintStars()
{
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 40; j++)
        {
            printf("**");
        }
        printf("\n");
    }
}
void FillArrayWithData(int rows, int cols, int**& tab, int fixedIncrementalValue)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            tab[i][j] = rows + i + j + fixedIncrementalValue;
        }
    }
}


void MultiplyMatrixes_CPU(int widthC, int heightC, int** tabA, int** tabB, int** tabC)
{
    int sum = 0;
    for (int i = 0; i < widthC; i++)
    {
        for (int j = 0; j < heightC; j++)
        {
            for (int k = 0; k < widthC; k++)
            {
                int tmp = tabA[i][k] * tabB[k][j];
                sum += tmp;
            }
            tabC[i][j] = sum;
            sum = 0;
        }
    }
}
//blockDim.x,y,z gives the number of threads in a block, in the particular direction
//gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
//blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction)


__global__ void MatrixMultiplication__CudaKernel(int* in_tabA, int* in_tabB, int* out_tabC, int N)
{
    //np. blok 0, watek 1, size 32
    //np. blok 1, watek 1, size 32 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int tmp_sum = 0; 
        for (int i = 0; i < N; i++) 
        {
            //todo1: find a way to get proper index
            //tmp_sum += in_tabA[row * N + i] * in_tabB[i * N + col];
            //tmp_sum += in_tabA[0] * in_tabB[0];
        }
        out_tabC[0] = tmp_sum;
    }

}

//array[rows][col]

void ReleaseMatrix(int width, int** tab)
{
    for (int i = 0; i < width; i++)
    {
        free(tab[i]);
    }
}

int main(void)
{
	std::cout << "LAB3" << std::endl;

    /** Declaration of arrays **/
    int** tabA, int** tabB, int** tabC;

    int widthA = 300, heightA = 150;
    int widthB = 150, heightB = 300;
    int widthC = 0, heightC = 0;

    /** Printing matrices sizes **/
    printf("Matrix A: %d x %d \Matrix B: %d x %d \n", widthA, heightA, widthB, heightB);

    /** Size check, 2x5 * 5x2 = 5x2 **/
    if (heightA == widthB)
    {
        heightC = heightA;
        widthC = widthB;

        /**Allocation for tabA, tabB, tabC **/
        MallocMatrix(tabA, widthA, heightA);
        MallocMatrix(tabB, widthB, heightB);
        MallocMatrix(tabC, widthC, heightC);

        /** Filling tabA with values  **/
        const int fixedIncrementalValueA = 1;
        FillArrayWithData(widthA, heightA, tabA, fixedIncrementalValueA);

        /** Filling tabB with values **/
        const int fixedIncrementlValueB = 3;
        FillArrayWithData(widthB, heightB, tabB, fixedIncrementlValueB);
        
        /** Matrices multiplication - by CPU **/
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix Multiplication by CPU " << std::endl;
        auto cpuStart = high_resolution_clock::now();
        MultiplyMatrixes_CPU(widthC, heightC, tabA, tabB, tabC);
        printf("\n");
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix Multiplication by CPU " << std::endl;
        auto cpuEnd = high_resolution_clock::now();

        /** Printing some rows from result matrix: tabC **/
        const int rowsToPrint = 10;
        const int colsToPrint = 3;
        PrintStars();
        PrintMatrix(tabC, colsToPrint, rowsToPrint);
        PrintStars();

        //Allocation for CUDA
        int* d_tabA_cuda;
        int* d_tabB_cuda;
        int* d_tabC_cuda;
         
        MallocFlattenMatrix_Cuda(d_tabA_cuda, widthA, heightA);
        MallocFlattenMatrix_Cuda(d_tabB_cuda, widthB, heightB);
        MallocFlattenMatrix_Cuda(d_tabC_cuda, widthC, heightC);

        //Conversion
        int* convertedTabA = Convert2Dto1D(tabA, widthA, heightA);
        int* convertedTabB = Convert2Dto1D(tabB, widthB, heightB);
        int* convertedTabC = Convert2Dto1D(tabC, widthC, heightC);

        PrintStars();
        PrintFlatten(convertedTabA, 5, 5, widthA, heightA);
        PrintStars();

        PrintStars();
        PrintMatrix(tabA, 5, 5);
        PrintStars();



        //Copying to device
        ErrorCheckCUDA(cudaMemcpy(d_tabA_cuda, convertedTabA, heightA * widthA * sizeof(int), cudaMemcpyHostToDevice));
        ErrorCheckCUDA(cudaMemcpy(d_tabB_cuda, convertedTabB, heightB * widthB * sizeof(int), cudaMemcpyHostToDevice));
        cudaMemset(d_tabC_cuda, 0, heightC * widthC * sizeof(int));

        //todo2:calculate number of blocks and threads properly
        // Initialize number of blocks and threads
        dim3 blocksPerGrid(512, 1, 1);
        dim3 threadsPerBlock(512, 1, 1);

        /** Matrices multiplication - by GPU **/
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix multiplication by GPU " << std::endl;
        auto gpuStart = high_resolution_clock::now();
        MatrixMultiplication__CudaKernel << <blocksPerGrid, threadsPerBlock >> > (d_tabA_cuda, d_tabB_cuda, d_tabC_cuda, widthC * heightC);
        ErrorCheckCUDA(cudaThreadSynchronize());
        auto gpuEnd = high_resolution_clock::now();
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix Multiplication by GPU" << std::endl;

        //Preparing memory for device result
        int* h_tabC;
        Malloc2DFlattenArray(h_tabC, widthC, heightC);
        memset(h_tabC, 0, widthC * heightC * sizeof(int));
        PrintFlatten(h_tabC, 10, 10, widthC, heightC);

        //Copying from device
        ErrorCheckCUDA(cudaMemcpy(h_tabC, d_tabC_cuda, heightC * widthC * sizeof(int), cudaMemcpyDeviceToHost));
       
        PrintStars();
        PrintFlatten(h_tabC, 10, 10, widthC, heightC);
        PrintStars();

        /** Freeing tabA, tabB, tabC - part 1**/
        ReleaseMatrix(widthA, tabA);
        ReleaseMatrix(widthB, tabB);
        ReleaseMatrix(widthC, tabC);

        /** Freeing - part 2 **/
        free(tabA);
        free(tabB);
        free(tabC);


        auto cpuDuration = duration_cast<nanoseconds>(cpuEnd - cpuStart);
        auto gpuDuration = duration_cast<nanoseconds>(gpuEnd - gpuStart);

        std::cout << "CPU duration " << cpuDuration.count() << "  nanoseconds " << std::endl;
        std::cout << "GPU duration " << gpuDuration.count() << "  nanoseconds " << std::endl;

        printf("\n");
    }
    else
    {
        printf("Matrices sizes are incorrect.\n\n");
    }
    system("pause");
    return 0;
}

void PrintMatrix(int**&tab, int toWhichColumn, int toWhichRow)
{
    for (int i = 0; i < toWhichColumn; i++)
    {
        for (int j = 0; j < toWhichRow; j++)
        {
            printf("%d ", tab[i][j]);
        }
        printf("\n");
    }
}

void PrintFlatten(int*& tab, int toWhichColumn, int toWhichRow, int width, int height)
{
    for (int i = 0; i < toWhichRow; i++)
    {
        for (int j = 0; j < toWhichColumn; j++)
        {
            printf("%d ", tab[j * width + i]);
        }
        printf("\n");
    }
}


void MallocMatrix(int**& tab, int width, int height)
{
    tab = (int**)malloc(width * sizeof(int*));
    for (int i = 0; i < width; i++)
    {
        tab[i] = (int*)malloc(height * sizeof(int));
    }
}

void Malloc2DFlattenArray(int*& tab, int width, int height)
{
    tab = new int[width * height];
}

void MallocFlattenMatrix_Cuda(int*& tab, int width, int height)
{
    //single dimension array on cuda (?)(?)(?)
    ErrorCheckCUDA(cudaMalloc((void**)&tab, sizeof(int) * width * height));
}