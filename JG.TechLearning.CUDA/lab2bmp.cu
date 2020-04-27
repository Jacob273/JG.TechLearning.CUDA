#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
//For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <chrono> 
#include <iomanip>
#include <limits>
#include <EasyBMP.h>
#include <windows.h>    //to use GetCurrentDir
#include <string.h>

using namespace std::chrono;

#define RGB_CHANNELS 3

void GrayScaleCPU(BMP& originalBmp)
{
	for (int i = 0; i < originalBmp.TellWidth(); i++)
	{
		for (int j = 0; j < originalBmp.TellHeight(); j++)
		{

			double temporaryValue =
				((originalBmp(i, j)->Red) +
				(originalBmp(i, j)->Green) +
					(originalBmp(i, j)->Blue)
					) / 3.0;

			//Setting all three to avg
			originalBmp(i, j)->Red = originalBmp(i, j)->Green = originalBmp(i, j)->Blue = (ebmpBYTE)temporaryValue;
		}
	}
}

BMP GrayScaleCPU(unsigned char* rgbBmpArray, BMP basedOn)
{
	BMP newBmp = BMP(basedOn);

	int width = newBmp.TellWidth();
	int height = newBmp.TellHeight();

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{

		double temporaryValue = (rgbBmpArray[(j * width) + i + 0]  +
								 rgbBmpArray[(j * width) + i + 1]  +
								 rgbBmpArray[(j * width) + i + 2]) 
									/ 3.0;

			//Setting all three to avg
			newBmp(i, j)->Red = newBmp(i, j)->Green = newBmp(i, j)->Blue = (ebmpBYTE)temporaryValue;
		}
	}
	return newBmp;
}

int GetImgResolution(BMP& img)
{
	return img.TellWidth() * img.TellHeight();
}

int GetTotalNumberOfPixels(BMP& img)
{
	return GetImgResolution(img) * RGB_CHANNELS;
}

unsigned char* RgbBitmapTo1dRgbCharArray(BMP& img)
{
	int sizeOfArray = GetTotalNumberOfPixels(img);

	unsigned char* output = new unsigned char[sizeOfArray];
	memset(output, 0, sizeof(output));

	int width = img.TellWidth();
	int height = img.TellHeight();

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			output[(j * width) + i + 0] = img(i, j)->Red;
			output[(j * width) + i + 1] = img(i, j)->Green;
			output[(j * width) + i + 2] = img(i, j)->Blue;
		}
	}

	return output;
}

BMP OneDimensionCharArrayToRgbBitmap(unsigned char* img, BMP basedOn)
{
	int width = basedOn.TellWidth();
	int height = basedOn.TellHeight();

	BMP newBmp = BMP(basedOn);

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			newBmp(i, j)->Red = img[(j * width) + i];
			newBmp(i, j)->Green = img[(j * width) + i];
			newBmp(i, j)->Blue = img[(j * width) + i];
		}
	}
	return newBmp;
}

__global__ void GrayScaleImage__CudaKernel(unsigned char* in_ImageRGB, unsigned char* out_GrayImage, const int width, const int height)
{
	int cols = width;
	int rows = height;

	//grid X
	//block	    0,0
	//thread	0,0	thread 1,0	thread 2,0 etc...

	//block		1,0
	//thread	0,0	thread	1,0	thread	2,0	etc...
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (col < width && row < height)
	{
		int grey_offset = row * cols + col;

		double r = in_ImageRGB[grey_offset + 0];
		double g = in_ImageRGB[grey_offset + 1];
		double b = in_ImageRGB[grey_offset + 2];

		out_GrayImage[grey_offset] = double((r + g + b) / 3.0);
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
	std::string fileName = "example2.bmp";
	std::string originalImageFullFilePath = currentPathString.append("\\img\\"+ fileName);
	const int exampleNameOffset = fileName.length();
	std::string cpuOutputPath = std::string(originalImageFullFilePath).insert(originalImageFullFilePath.length() - exampleNameOffset, "CPU-JakubGmur-Output-");
	std::string gpuOutputPath = std::string(originalImageFullFilePath).insert(originalImageFullFilePath.length() - exampleNameOffset, "GPU-JakubGmur-Output-");

	//Loading image for CPU
	BMP sourceImageForCPU = BMP();
	if (!sourceImageForCPU.ReadFromFile(originalImageFullFilePath.c_str()))
	{
		std::cout << "Could not load image... on a given path: " << originalImageFullFilePath << std::endl;
		return -1;
	}

	const int bitDepth = 16;

	std::cout << "Width of an image: " << sourceImageForCPU.TellWidth() << std::endl;
	std::cout << "Height of an image: " << sourceImageForCPU.TellHeight() << std::endl;
	std::cout << "Resolution (total number of pixels): " << sourceImageForCPU.TellWidth() * sourceImageForCPU.TellHeight() << std::endl;
	std::cout << "Number of colors: " << sourceImageForCPU.TellNumberOfColors() << std::endl;

	sourceImageForCPU.SetBitDepth(bitDepth);
	//Gray scaling on CPU
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Started Gray Scaling Img on CPU " << std::endl;
	auto cpuStart = high_resolution_clock::now();
	GrayScaleCPU(sourceImageForCPU);
	auto cpuEnd = high_resolution_clock::now();
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Finished Gray Scaling Img on CPU" << std::endl;
	if (!sourceImageForCPU.WriteToFile(cpuOutputPath.c_str()))
	{
		std::cout << "Could save image... on a given path: " << cpuOutputPath << std::endl;
	}

	//Loading image for GPU
	BMP sourceImageForGPU = BMP();
	if (!sourceImageForGPU.ReadFromFile(originalImageFullFilePath.c_str()))
	{
		std::cout << "Could not load image... on a given path: " << originalImageFullFilePath << std::endl;
	}

	sourceImageForGPU.SetBitDepth(bitDepth);

	//BMP conversion
	unsigned char* h_rgbImageArr = RgbBitmapTo1dRgbCharArray(sourceImageForGPU); 
	
	//reversed conversion test
	//BMP bmpresult1 = GrayScaleCPU(h_rgbImageArr, sourceImageForGPU);
	//bmpresult1.WriteToFile((gpuOutputPath).c_str());
	//return 0;

	//CUDA device check
	int  devCnt;
	cudaGetDeviceCount(&devCnt);
	if (devCnt == 0)
	{
		std::cout << "No CUDA devices available -- exiting." << std::endl;
		return -1;
	}

	//Allocation for CUDA
	const int imageResolution = GetImgResolution(sourceImageForGPU);
	unsigned char* d_rgbInputImageArr_cuda;      //array for storing RGB data on cuda device
	unsigned char* d_AvgOfRgbImgArr_cuda;        //array for storing (avg of RGB) data on cuda device

	ErrorCheckCUDA(cudaMalloc((void**)&d_rgbInputImageArr_cuda, imageResolution * RGB_CHANNELS * sizeof(unsigned char)));
	ErrorCheckCUDA(cudaMalloc((void**)&d_AvgOfRgbImgArr_cuda, imageResolution * sizeof(unsigned char)));
	cudaMemset(d_AvgOfRgbImgArr_cuda, 255, imageResolution * sizeof(unsigned char));//setting all pixels to be gray ~222
	ErrorCheckCUDA(cudaMemcpy(d_rgbInputImageArr_cuda, h_rgbImageArr, imageResolution * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Block size
	dim3 dimBlock(128, 128);
	dim3 dimGrid(ceil((sourceImageForGPU.TellWidth() - 1) / dimBlock.x), ceil((sourceImageForGPU.TellHeight() + dimBlock.y - 1) / dimBlock.y));
	
	//Gray scaling on GPU
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Started Gray Scaling Img on GPU " << std::endl;
	auto gpuStart = high_resolution_clock::now();
	GrayScaleImage__CudaKernel << <dimGrid, dimBlock >> > (d_rgbInputImageArr_cuda, d_AvgOfRgbImgArr_cuda, sourceImageForGPU.TellWidth(), sourceImageForGPU.TellHeight());
	ErrorCheckCUDA(cudaThreadSynchronize());
	auto gpuEnd = high_resolution_clock::now();
	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~Finished Gray Scaling Img on GPU" << std::endl;

	//Copying from Device to Host
	unsigned char* h_AvgOfRgbImgArr = new unsigned char[imageResolution];  //array for storing (avg of RGB) data on local device
	ErrorCheckCUDA(cudaMemcpy(h_AvgOfRgbImgArr, d_AvgOfRgbImgArr_cuda, imageResolution * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	//test code to verify what are the values which were calculated 
	//for (int i = 0; i < 1000; i++)
	//{
	//	std::cout << (int)h_AvgOfRgbImgArr[i] << std::endl;
	//}

	//Writing 
	BMP gpuResult = OneDimensionCharArrayToRgbBitmap(h_AvgOfRgbImgArr, sourceImageForGPU);
	if (!gpuResult.WriteToFile(gpuOutputPath.c_str()))
	{
		std::cout << "Could save image... on a given path: " << cpuOutputPath << std::endl;
	}

	ErrorCheckCUDA(cudaFree(d_AvgOfRgbImgArr_cuda));
	ErrorCheckCUDA(cudaFree(d_rgbInputImageArr_cuda));
	free(h_rgbImageArr);
	free(h_AvgOfRgbImgArr);

	auto gpuDuration = duration_cast<nanoseconds>(gpuEnd - gpuStart);
	auto cpuDuration = duration_cast<nanoseconds>(cpuEnd - cpuStart);

	std::cout << "CPU duration " << gpuDuration.count() << "  nanoseconds " << std::endl;
	std::cout << "GPU duration " << cpuDuration.count() << "  nanoseconds" <<std::endl;
	return 0;
}