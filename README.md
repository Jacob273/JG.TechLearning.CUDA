CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia.

https://en.wikipedia.org/wiki/CUDA

* Code written on NVIDIA Cuda Runtime 10.2

* Tested on Quadro M2000M

Implemented & measured:

1. Polynomial calculation on CPU & GPU (part of the code was provided by an academic teacher)
2. Image gray scaling on CPU & GPU
3. Matrix multiplication on CPU & GPU (how the loop rolling/unrolling performed by the NVCC compiler affects the time it takes to perform  calculations)


_Tasks that have been completed here and theoretical introduction to laboratories was developed by IT engineer and academic teacher - Slawomir Wernikowski at the West Pomeranian University of Technology_

## Image gray scaling on CPU & GPU output files & measurements

![Input 1](./outputs/example.bmp)
![Output 1](./outputs/Output-example.bmp)
![Input 2](./outputs/example2.bmp)
![Output 2](./outputs/Output-example2.bmp)

---------------------------------------------
LAB2
---------------------------------------------

Tests performed on following CPU:

Caption           : Intel64 Family 6 Model 94 Stepping 3

DeviceID          : CPU0

Manufacturer      : GenuineIntel

MaxClockSpeed     : 2712

Name              : Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz

SocketDesignation : U3E1

---------------------------------------------
Tested performed on following GPU:
NVIDIA Quadro M2000M

---------------------------------------------
__TEST1__
Width of an image: 640

Height of an image: 480

Resolution (total number of pixels): 307200

Number of colors: 256

GPU duration 2370833  nanoseconds

CPU duration 69864629  nanoseconds

---------------------------------------------
__TEST2__

Width of an image: 640

Height of an image: 480

Resolution (total number of pixels): 307200

Number of colors: 256

GPU duration 2420297  nanoseconds

CPU duration 68725466  nanoseconds

---------------------------------------------
__TEST3__

Width of an image: 1419

Height of an image: 1001

Resolution (total number of pixels): 1420419

Number of colors: 16777216

GPU duration 11482626  nanoseconds

CPU duration 346865093  nanoseconds

---------------------------------------------

__TEST4__

Width of an image: 1419

Height of an image: 1001

Resolution (total number of pixels): 1420419

Number of colors: 16777216

GPU duration 11523404  nanoseconds

CPU duration 343563144  nanoseconds


## Matrix multiplication on CPU & GPU (how the loop rolling/unrolling performed by the NVCC compiler affects the time it takes to perform  calculations)

