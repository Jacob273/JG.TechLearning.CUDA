#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
//For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <chrono> 
#include <iomanip>
#include <limits>
#include <EasyBMP.h>
#include <windows.h>
#include <string.h>
#include <EasyBMP_VariousBMPutilities.h>

using namespace std::chrono;

#define  BLOCK_SIZE 1024

void GrayScaleCPU(BMP& originalBmp)
{
    for (int i = 0; i < originalBmp.TellWidth(); i++)
    {
        for (int j = 0; j < originalBmp.TellHeight(); j++)
        {

            double temporaryValue =
                ( (originalBmp(i, j)->Red) +
                  (originalBmp(i, j)->Green) +
                  (originalBmp(i, j)->Blue)
                ) / 3;

            //Setting all three to avg
            originalBmp(i, j)->Red = originalBmp(i, j)->Green = originalBmp(i, j)->Blue = (BYTE)temporaryValue;
        }
    }
}

int GetImgTotalSize(BMP& img)
{
    int rgbChannels = 3;
    return img.TellWidth() * img.TellHeight() * rgbChannels;
}

unsigned char* BitmapToCharArray(BMP& img)
{
    int sizeOfArray = GetImgTotalSize(img);

    unsigned char* output = new unsigned char[sizeOfArray];
    memset(output, 0, sizeof(output));

    int width = img.TellWidth();
    int height = img.TellWidth();

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            output[(j * width) + i] = img(i, j)->Red;
            output[(j * width) + i] = img(i, j)->Green;
            output[(j * width) + i] = img(i, j)->Blue;
        }
    }
    return output;
}

BMP CharArrayToBitmap(unsigned char* img, int width, int height, int depth)
{
    int rgbChannels = 3;
    int sizeOfArray = width * height * rgbChannels;

    BMP newBmp = BMP();
    newBmp.SetSize(width, height);
    newBmp.SetBitDepth(depth);

    for (int i = 0; i < newBmp.TellWidth(); i++)
    {
        for (int j = 0; j < newBmp.TellHeight(); j++)
        {
            newBmp(i, j)->Red = img[(j * width) + i];
            newBmp(i, j)->Green = img[(j * width) + i];
            newBmp(i, j)->Blue = img[(j * width) + i];
        }
    }

    return newBmp;
}

__global__ void GrayScaleImage__CudaKernel(unsigned char* in_Image, unsigned char* out_GrayImage, const int width, const int height)
{
    int tx = (blockIdx.y * blockDim.y) + threadIdx.y;
    int ty = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ((ty < height && tx < width))
    {
        unsigned int pixelIndex = ty * width + tx;
        float r = static_cast<float>(in_Image[pixelIndex * 3 + 0]);
        float g = static_cast<float>(in_Image[pixelIndex * 3 + 1]);
        float b = static_cast<float>(in_Image[pixelIndex * 3 + 2]);

        float grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

        out_GrayImage[pixelIndex] = static_cast<unsigned char>(grayPix);
    }
}

#define ErrorCheckCUDA(ans) { CheckErrorCUDA((ans), __FILE__, __LINE__); }
inline void CheckErrorCUDA(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, ":::::::::::::CheckErrorCUDA::::::::::::: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(void)
{
    //Path preparation
    char currentPathBuffer[256];
    GetCurrentDirectoryA(256, currentPathBuffer);
    std::string currentPathString = currentPathBuffer;
    std::string fullFilePath = currentPathString.append("\\img\\example.bmp");

    const int bitDepth = 16;
    //Loading image for CPU
	BMP bmpImageGrayScaledByCPU = BMP();
	bmpImageGrayScaledByCPU.ReadFromFile(fullFilePath.c_str());

    std::cout << "Width of am image: " << bmpImageGrayScaledByCPU.TellWidth() << std::endl;
    std::cout << "Height of am image: " << bmpImageGrayScaledByCPU.TellHeight() << std::endl;
    std::cout << "Resolution (total number of pixels): " << bmpImageGrayScaledByCPU.TellWidth() * bmpImageGrayScaledByCPU.TellHeight() << std::endl;
    std::cout << "Number of colors: " << bmpImageGrayScaledByCPU.TellNumberOfColors() << std::endl;

    bmpImageGrayScaledByCPU.SetBitDepth(bitDepth);
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Started Gray Scaling Img on CPU " << std::endl;
    GrayScaleCPU(bmpImageGrayScaledByCPU);
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Finished Gray Scaling Img on CPU" << std::endl;
	bmpImageGrayScaledByCPU.WriteToFile(fullFilePath.insert(fullFilePath.length() - 4, "JakubGmur-Output-CPUCPU").c_str());

    //Conversion tests
    //char* imageArr = BitmapToCharArray(bmpImageGrayScaledByCPU);
    //BMP converted = CharArrayToBitmap(imageArr, bmpImageGrayScaledByCPU.TellWidth(), bmpImageGrayScaledByCPU.TellHeight(), bitDepth);
    //converted.WriteToFile(fullFilePath.insert(fullFilePath.length() - 4, "JakubGmur-Output-Converted2222").c_str());
    
    //Loading image for GPU
    BMP sourceImageForGPU = BMP();
    sourceImageForGPU.ReadFromFile(fullFilePath.c_str());
    sourceImageForGPU.SetBitDepth(bitDepth);
    
    //BMP conversion
    unsigned char* hostImageArr = BitmapToCharArray(sourceImageForGPU);
    const int totalSizeOfImg = GetImgTotalSize(sourceImageForGPU);
    
    //CUDA check
    int  devCnt;
    cudaGetDeviceCount(&devCnt);
    if (devCnt == 0)
    {
        perror("No CUDA devices available -- exiting.");
        return 1;
    }

    //Allocation for CUDA
    unsigned char* deviceImageArr;
    ErrorCheckCUDA(cudaMalloc((void**)&deviceImageArr, totalSizeOfImg * sizeof(unsigned char)));
    //Copying hostImageArr from Host to Device
    ErrorCheckCUDA(cudaMemcpy(deviceImageArr, hostImageArr, totalSizeOfImg * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Block size calculation
    int imageSize = sourceImageForGPU.TellWidth() * sourceImageForGPU.TellHeight();
    int rgbChannels = 3;

    int totalNumberOfBlocksToUse = 1024;
    int blocks = (sourceImageForGPU.TellWidth() * sourceImageForGPU.TellHeight()) / totalNumberOfBlocksToUse;
    if (imageSize % totalNumberOfBlocksToUse)
    {
        blocks++;
    }

    //Obliczanie wartosci sredniej piksela
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Started Gray Scaling Img on GPU " << std::endl;
    GrayScaleImage__CudaKernel << <blocks, totalNumberOfBlocksToUse>> > (hostImageArr, deviceImageArr, sourceImageForGPU.TellWidth(), sourceImageForGPU.TellHeight());
    ErrorCheckCUDA(cudaThreadSynchronize());
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Finished Gray Scaling Img on GPU" << std::endl;


    //Copying deviceImageArr from Device to Host
    const int graySize = sourceImageForGPU.TellWidth() * sourceImageForGPU.TellHeight();
    ErrorCheckCUDA(cudaMemcpy(hostImageArr, deviceImageArr, graySize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    ErrorCheckCUDA(cudaFree(deviceImageArr));
    BMP gpuResult = CharArrayToBitmap(hostImageArr, sourceImageForGPU.TellWidth(), sourceImageForGPU.TellHeight(), bitDepth);
    gpuResult.WriteToFile(fullFilePath.insert(fullFilePath.length() - 4, "JakubGmur-Output-GPUGPU").c_str());
	return 0;
}