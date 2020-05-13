CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia.

https://en.wikipedia.org/wiki/CUDA

* Code written on NVIDIA Cuda Runtime 10.2

* Tested on Quadro M2000M

Implemented & measured:

1. Polynomial calculation on CPU & GPU (part of the code was provided by an academic teacher)
2. Image gray scaling on CPU & GPU
3. Matrix multiplication on CPU & GPU (how the loop rolling/unrolling performed by the NVCC compiler affects the time it takes to perform  calculations)


_Tasks that have been completed here and theoretical introduction to laboratories was developed by IT engineer and academic teacher - Slawomir Wernikowski at the West Pomeranian University of Technology_

## Polynomial calculation results

__Test 1__

JG::Starting program which uses GPu and CPU to calculate polynomial for __<1000000>__ values
* GPU no. of blocks : __<16>__
* Duration GPU: <3> [ms]
* Duration CPU: <2> [ms]

__Test 2__
JG::Starting program which uses GPu and CPU to calculate polynomial for __<10000000>__ values
* GPU no. of blocks : __<16>__
* Duration GPU: <28> [ms]
* Duration CPU: <31> [ms]

__Test 2.1__
JG::Starting program which uses GPu and CPU to calculate polynomial for __<10000000>__ values
* GPU no. of blocks : __<1024>__
* Duration GPU: <17> [ms]
* Duration CPU: <32> [ms]

__Test 3__
JG::Starting program which uses GPu and CPU to calculate polynomial for __<100000000>__ values
* GPU no. of blocks : __<16>__
* Duration GPU: <290> [ms]
* Duration CPU: <348> [ms]

__Test 4__
* JG::Starting program which uses GPu and CPU to calculate polynomial for __<100000000>__ values
* GPU no. of blocks : __<1024>__
* Duration GPU: <189> [ms]
* Duration CPU: <362> [ms]

Conclusions:
* When size of vector storing values is bigger than 10^6 the GPU calculations are much faster than CPU (even twice faster)
* When number of blocks (and threads inside blocks) are fully used we may get results pretty fast, decreasing number of threads used in blocks will extend the time 

## Image gray scaling on CPU & GPU output files & measurements

![Input 1](./outputs/example.bmp)
![Output 1](./outputs/Output-example.bmp)
![Input 2](./outputs/example2.bmp)
![Output 2](./outputs/Output-example2.bmp)

---------------------------------------------
Results
---------------------------------------------
CPU: Intel64 Family 6 Model 94 Stepping 3
---------------------------------------------
GPU:NVIDIA Quadro M2000M
---------------------------------------------
__TEST1__

Width of an image: 640

Height of an image: 480

Resolution (total number of pixels): 307200

Number of colors: 256

__GPU duration 2370833  nanoseconds__

__CPU duration 69864629  nanoseconds__

---------------------------------------------
__TEST2__

Width of an image: 640

Height of an image: 480

Resolution (total number of pixels): 307200

Number of colors: 256

__GPU duration 2420297  nanoseconds__

__CPU duration 68725466  nanoseconds__

---------------------------------------------
__TEST3__

Width of an image: 1419

Height of an image: 1001

Resolution (total number of pixels): 1420419

Number of colors: 16777216

__GPU duration 11482626  nanoseconds__

__CPU duration 346865093  nanoseconds__

---------------------------------------------

__TEST4__

Width of an image: 1419

Height of an image: 1001

Resolution (total number of pixels): 1420419

Number of colors: 16777216

__GPU duration 11523404  nanoseconds__

__CPU duration 343563144  nanoseconds__


## Matrix multiplication on CPU & GPU Results


a) Matrix A=300x300, Matrix B=300x300
   * CPU (avg) = 133ms
   * GPU version "without pragma unroll" (avg) = 123ms 
   * GPU version with pragma unroll (avg) = 111ms ( 10% time faster)

b) Matrix A=900x900, Matrix B=900x900

  * CPU (avg) = 4839 ms,
  * GPU version "without pragma unroll" (avg) = 1646ms
  * GPU version with pragma unroll (widthC calculated at runtime, no boost) = 1633ms (not faster at all)
  * GPU version with pragma roll but used CONST_WIDTH_C 400ms (almost 24-30% time faster)
   

 Loop below has been tested with #pragma unroll

        for (int i = 0; i < CONST_WIDTH_C; i++)
        {
            tmp_sum += in_tabA[row * CONST_WIDTH_C+ i] * in_tabB[i * CONST_WIDTH_C + col];
        }
        

Conclusions:

* The effectiveness of the #pragma unroll directive in the context of performance - strongly depends on what is calculated in a loop
* When performing tests, it turned out that the CONST_WIDTH_C variable must be either constant or defined using #define - if it was calculated in runtime based on the size of dynamic matrix (i.e. not known at the compilation stage) - then #pragma unroll did not bring any profit
* #pragma unroll at the compilation stage reduce the number of operations that would have to be carried out in the case of dynamic loop development is reduced


