#pragma once

void MultiplyMatrixes_CPU(int szerC, int dlC, int** tabA, int** tabB, int** tabC);

void MallocMatrix(int**& tabA, int widthA, int heightA);

void Malloc2DFlattenArray(int*& tab, int width, int height);

void PrintMatrix(int**& tab, int toWhichColumn, int toWhichRow);

void PrintFlatten(int*& tab, int toWhichColumn, int toWhichRow, int width, int height);

void ReleaseMatrix(int widthA, int** tabA);

void MallocFlattenMatrix_Cuda(int*& tab, int width, int height);

void FillArrayWithData(int rows, int cols, int**& tab, int fixedIncrementalValue);