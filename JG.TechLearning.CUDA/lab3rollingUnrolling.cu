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

#define  BLOCK_SIZE_THREAD_NO 32

using namespace std::chrono;


/**
Notes:

ROWS = HEIGHT
COLUMNS = WIDTH
[RowIndex,ColumnIndex]

blockDim.x,y,z gives the number of threads in a block, in the particular direction
gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction)

**/


void PrintMatrixFromEnd(int**& tab, int toWhichColumn, int toWhichRow, int height, int width);
int* Convert2Dto1D(int**& tab, int width, int height);
void FillArrayWithZeroes(int rows, int cols, int** tab);
void PrintStars();
void FillArrayWithData(int rows, int cols, int**& tab, int fixedIncrementalValue);
void MultiplyMatrixes_CPU(int widthC, int heightC, int** tabA, int** tabB, int** tabC);
__global__ void MatrixMultiplication__CudaKernel(int* in_tabA, int* in_tabB, int* out_tabC, int outTabWidth);
void ReleaseMatrix(int height, int** tab);
void MultiplyMatrixes_CPU(int widthC, int heightC, int** tabA, int** tabB, int** tabC);
void MallocMatrix(int**& tab, int width, int height);
void Malloc2DFlattenArray(int*& tab, int width, int height);
void PrintMatrix(int**& tab, int toWhichColumn, int toWhichRow);
void PrintFlatten(int*& tab, int toWhichColumn, int toWhichRow, int width, int height);
void ReleaseMatrix(int height, int** tab);
void MallocFlattenMatrix_Cuda(int*& tab, int width, int height);
void FillArrayWithData(int rows, int cols, int**& tab, int fixedIncrementalValue);

int main(void)
{
	std::cout << ">>Lab3 CPU VS GPU & Pragma Rolling/Unrolling << BlockSize: <" << BLOCK_SIZE_THREAD_NO << ">" << std::endl;
    /** Declaration of arrays **/
    int** tabA, int** tabB, int** tabC;

    /** Declaration of size of arrays **/
    int widthA = 300, heightA = 300;
    int widthB = 300, heightB = 300;
    int widthC = 0, heightC = 0;

    /** First N Rows & M columns that are going to be printed **/
    const int rowsToPrint = 3;
    const int colsToPrint = 3;

    /** Printing matrices sizes **/
    printf("Matrix A: %d x %d \Matrix B: %d x %d \n", widthA, heightA, widthB, heightB);

    /** Size check: (2x5 * 5x2 = 5x2) **/
    if (heightA == widthB)
    {
        heightC = heightA;
        widthC = widthB;

        printf("Matrix C: %d x %d \n", widthC, heightC);
        
        /**Allocation for tabA, tabB, tabC **/
        MallocMatrix(tabA, widthA, heightA);
        MallocMatrix(tabB, widthB, heightB);
        MallocMatrix(tabC, widthC, heightC);

        /** Filling tabA and tabB with values  **/
        const int fixedIncrementalValueA = 1;
        FillArrayWithData(heightA, widthA, tabA, fixedIncrementalValueA);
        const int fixedIncrementlValueB = 3;
        FillArrayWithData(heightB, widthB, tabB, fixedIncrementlValueB);
        


        /** Matrices multiplication - by CPU **/
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix Multiplication by CPU - STARTED " << std::endl;
        auto cpuStart = high_resolution_clock::now();
        MultiplyMatrixes_CPU(widthC, heightC, tabA, tabB, tabC);
        printf("\n");
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix Multiplication by CPU - ENDED " << std::endl;
        auto cpuEnd = high_resolution_clock::now();

        /** Printing result matrix: tabC **/

        std::cout << "Printing matrix calculated by CPU:::Few first indexes" << std::endl;
        PrintStars();
        PrintMatrix(tabC, colsToPrint, rowsToPrint);
        PrintStars();

        std::cout << "Printing matrix calculated by CPU::: Few last indexes" << std::endl;
        PrintStars();
        PrintMatrixFromEnd(tabC, colsToPrint, rowsToPrint, heightC, widthC);
        PrintStars();

        //Allocation for CUDA
        int* d_tabA_cuda;
        int* d_tabB_cuda;
        int* d_tabC_cuda;
         
        MallocFlattenMatrix_Cuda(d_tabA_cuda, widthA, heightA);
        MallocFlattenMatrix_Cuda(d_tabB_cuda, widthB, heightB);
        MallocFlattenMatrix_Cuda(d_tabC_cuda, widthC, heightC);

        //Conversion from 2D arrays to 1D arrays
        int* convertedTabA = Convert2Dto1D(tabA, widthA, heightA);
        int* convertedTabB = Convert2Dto1D(tabB, widthB, heightB);

        //Copying to device 1d array
        ErrorCheckCUDA(cudaMemcpy(d_tabA_cuda, convertedTabA, heightA * widthA * sizeof(int), cudaMemcpyHostToDevice));
        ErrorCheckCUDA(cudaMemcpy(d_tabB_cuda, convertedTabB, heightB * widthB * sizeof(int), cudaMemcpyHostToDevice));
        cudaMemset(d_tabC_cuda, 0, heightC * widthC * sizeof(int));

        // Initializing number of blocks and threads
        dim3 threadsPerBlock(BLOCK_SIZE_THREAD_NO, BLOCK_SIZE_THREAD_NO, 1);
        dim3 gridOfBlocks(ceil((widthA - 1) / BLOCK_SIZE_THREAD_NO) + 1, ceil((widthB - 1) / BLOCK_SIZE_THREAD_NO) + 1);

        /** Matrices multiplication - by GPU **/
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix multiplication by GPU - STARTED " << std::endl;
        auto gpuStart = high_resolution_clock::now();
        MatrixMultiplication__CudaKernel << <gridOfBlocks, threadsPerBlock >> > (d_tabA_cuda, d_tabB_cuda, d_tabC_cuda, widthC);
        ErrorCheckCUDA(cudaThreadSynchronize());
        auto gpuEnd = high_resolution_clock::now();
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Matrix Multiplication by GPU - ENDED" << std::endl;

        /** Preparing memory for device result **/
        int* h_tabC;
        std::cout << "Printing matrix calculated by GPU::: Few first indexes " << std::endl;
        Malloc2DFlattenArray(h_tabC, widthC, heightC);
        memset(h_tabC, 0, widthC * heightC * sizeof(int));

        //Copying from device
        ErrorCheckCUDA(cudaMemcpy(h_tabC, d_tabC_cuda, heightC * widthC * sizeof(int), cudaMemcpyDeviceToHost));
       
        /** Printing result from CUDA **/
        PrintStars();
        PrintFlatten(h_tabC, colsToPrint, rowsToPrint, widthC, heightC);
        PrintStars();

        //Some randoms to verify if CPU and GPU has calculated the same values
        std::cout << "Control values for verification:: " << std::endl;
        std::cout << "value control check on GPU: [0] =" << h_tabC[0] << std::endl;
        std::cout << "value control check on CPU: [0][0] =" << tabC[0][0] << std::endl;
        std::cout << std::endl << std::endl;
        std::cout << "value control check on GPU: [" << widthC <<"]  =" << h_tabC[widthC] << std::endl;
        std::cout << "value control check on CPU: [1][0] =" << tabC[1][0] << std::endl;
        std::cout << std::endl << std::endl;
        std::cout << "value control check on GPU: [" << 100 * widthC << "]  =" << h_tabC[100 * widthC] << std::endl;
        std::cout << "value control check on CPU: [100][0] =" << tabC[100][0] << std::endl;
        std::cout << std::endl << std::endl;
        std::cout << "value control check on GPU: [" << 200 * widthC << "]  =" << h_tabC[200 * widthC] << std::endl;
        std::cout << "value control check on CPU: [200][0] =" << tabC[200][0] << std::endl;
        std::cout << std::endl << std::endl;
        std::cout << "value control check on GPU: [" << 250 * widthC << "]  =" << h_tabC[250 * widthC] << std::endl;
        std::cout << "value control check on CPU: [250][0] =" << tabC[250][0] << std::endl;
        std::cout << std::endl << std::endl;

        std::cout << std::endl << std::endl;
        ReleaseMatrix(heightA, tabA);
        ReleaseMatrix(heightB, tabB);
        ReleaseMatrix(heightC, tabC);

        free(tabA);
        free(tabB);
        free(tabC);

        free(convertedTabA);
        free(convertedTabB);

        ErrorCheckCUDA(cudaFree(d_tabA_cuda));
        ErrorCheckCUDA(cudaFree(d_tabB_cuda));
        ErrorCheckCUDA(cudaFree(d_tabC_cuda));

        auto cpuDuration = duration_cast<milliseconds>(cpuEnd - cpuStart);
        auto gpuDuration = duration_cast<milliseconds>(gpuEnd - gpuStart);

        std::cout << "CPU duration " << cpuDuration.count() << "  milliseconds " << std::endl;
        std::cout << "GPU duration " << gpuDuration.count() << "  milliseconds " << std::endl;

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
    //height
    for (int i = 0; i < toWhichRow; i++)
    {
        //width
        for (int j = 0; j < toWhichColumn; j++)
        {
            printf( "[%d , %d ] = %d  ",i , j, tab[i][j]);
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
            printf("[offset:%d] =  %d", i * width + j,  tab[i * width + j]);
        }
        printf("\n");
    }
}

void MallocMatrix(int**& tab, int width, int height)
{
    tab = (int**)malloc(height * sizeof(int*));
    for (int i = 0; i < height; i++)
    {
        tab[i] = (int*)malloc(width * sizeof(int));
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


void PrintMatrixFromEnd(int**& tab, int toWhichColumn, int toWhichRow, int height, int width)
{
    //height
    for (int i = height - 1; i > height - toWhichRow - 1; i--)
    {
        //width
        for (int j = width - 1; j > width - toWhichColumn - 1; j--)
        {
            printf("[%d , %d ] = %d  ", i, j, tab[i][j]);
        }
        printf("\n");
    }
}

int* Convert2Dto1D(int**& tab, int width, int height)
{
    int sizeOfArray = width * height;

    int* output = (int*)malloc(sizeOfArray * sizeof(int));
    memset(output, 0, sizeof(output));

    //rows
    for (int i = 0; i < height; i++)
    {
        //cols
        for (int j = 0; j < width; j++)
        {
            output[(i * width) + j] = tab[i][j];    //[1,0] = [300]
        }
    }

    return output;
}

void FillArrayWithZeroes(int rows, int cols, int** tab)
{
    //height
    for (int i = 0; i < rows; i++)
    {
        //width
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
    //height
    for (int i = 0; i < rows; i++)
    {
        //width
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

__global__ void MatrixMultiplication__CudaKernel(int* in_tabA, int* in_tabB, int* out_tabC, int outTabWidth)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    //making sure that extra threads will do not any work
    if (row < outTabWidth && col < outTabWidth)
    {
        int tmp_sum = 0;

        //#pragma unroll
        for (int i = 0; i < outTabWidth; i++)
        {
            tmp_sum += in_tabA[row * outTabWidth + i] * in_tabB[i * outTabWidth + col];
        }
        out_tabC[row * outTabWidth + col] = tmp_sum;
    }
}

void ReleaseMatrix(int height, int** tab)
{
    for (int i = 0; i < height; i++)
    {
        free(tab[i]);
    }
}
