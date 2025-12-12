#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <string>

#ifndef PI
#define PI 3.14159265358979323846f
#endif // PI

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
    {                                                                          \
        auto status = static_cast<cudaError_t>(call);                          \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr,                                                    \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed " \
                    "with "                                                    \
                    "%s (%d).\n",                                              \
                    #call, __LINE__, __FILE__, cudaGetErrorString(status),     \
                    status);                                                   \
            exit(status);                                                      \
        }                                                                      \
    }
#endif // CUDA_RT_CALL

__global__ void scalingKernel(float* data, int elementCount, int scale);

__global__ void makeSheppTableKernel(float* sheppTable, int width,
                                     float pixelSize);
void makeSheppTable(float** d_sheppTable, int width, float pixelSize);

__global__ void convolutionKernel(float* sinogramIn, float* sinogramOut,
                                  float* sheppTable, int width, int batchSize);

void convolution(float* sinogram, float* d_sheppTable, int width,
                 int batchSize);

__global__ void makeCosSinTableKernel(int projectionCount, float* cosSinTable);

void makeCosSinTable(int projectionCount, float** d_cosSinTable);

// currentBatchSize of slices processed in one kernel launch
__global__ void backProjectionKernel(int currentBatchSize, int width,
                                     float pixelSize, int projectionCount,
                                     float roaShift, float* cosSinTable,
                                     float* sinograms, float* voxels);

__global__ void normalizeVoxelsKernel(float* voxels, int* counter,
                                      long long voxelCount);

void backProjection(int currentBatchSize, const float* sinogram, int width,
                    float pixelSize, int projectionCount, float roaShift,
                    float* voxels, float* d_cosSinTable);

// 重畳積分法(CBP)
void cbp(const int detectorWidth, const int detectorHeight,
         const int projectionCount, const float detectorPixelSize,
         const float roaShift, const std::string& inputFolder,
         const std::string& inputFilename, const std::string& outputFolder,
         const std::string& outputPrefix = "voxels_cbp_");

void cbp2(const int detectorWidth, const int detectorHeight,
          const int projectionCount, const float detectorPixelSize,
          const float roaShift, const std::string& inputFolder,
          const std::string& inputFilenamePrefix,
          const std::string& outputFolder,
          const std::string& outputPrefix = "voxels_cbp_");
