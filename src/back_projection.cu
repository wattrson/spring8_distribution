#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#include "back_projection.h"
#include "file_io.h"

__global__ void scalingKernel(float* data, int elementCount, int scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elementCount) {
        data[i] /= static_cast<float>(scale);
    }
}

__global__ void makeSheppTableKernel(float* sheppTable, int width,
                                     float pixelSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        sheppTable[i] = 2.f / (PI * PI * pixelSize * (1.f - 4.f * i * i));
    }
}

void makeSheppTable(float** d_sheppTable, int width, float pixelSize) {
    CUDA_RT_CALL(cudaMalloc(d_sheppTable, sizeof(float) * width));
    makeSheppTableKernel<<<(width + 255) / 256, 256>>>(*d_sheppTable, width,
                                                       pixelSize);
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

__global__ void convolutionKernel(float* sinogramIn, float* sinogramOut,
                                  float* sheppTable, int width, int batchSize) {
    int    i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < batchSize) {
        float value = 0.f;
        for (int k = 0; k < width; k++) {
            value += sinogramIn[j * width + k] * sheppTable[abs(k - i)];
        }
        sinogramOut[j * width + i] = value;
    }
}

void convolution(float* sinogram, float* d_sheppTable, int width,
                 int batchSize) {
    float *d_sinogramIn, *d_sinogramOut;
    CUDA_RT_CALL(cudaMalloc(&d_sinogramIn, sizeof(float) * width * batchSize));
    CUDA_RT_CALL(cudaMalloc(&d_sinogramOut, sizeof(float) * width * batchSize));
    CUDA_RT_CALL(cudaMemcpy(d_sinogramIn, sinogram,
                            sizeof(float) * width * batchSize,
                            cudaMemcpyHostToDevice));
    convolutionKernel<<<dim3((width + 31) / 32, (batchSize + 15) / 16),
                        dim3(32, 16)>>>(d_sinogramIn, d_sinogramOut,
                                        d_sheppTable, width, batchSize);
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaMemcpy(sinogram, d_sinogramOut,
                            sizeof(float) * width * batchSize,
                            cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaFree(d_sinogramIn));
    CUDA_RT_CALL(cudaFree(d_sinogramOut));
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

__global__ void makeCosSinTableKernel(int projectionCount, float* cosSinTable) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < projectionCount) {
        float angle = static_cast<float>(i) * PI /
                      static_cast<float>(projectionCount - 1);
        cosSinTable[2 * i]     = cos(angle);
        cosSinTable[2 * i + 1] = sin(angle);
    }
}

void makeCosSinTable(int projectionCount, float** d_cosSinTable) {
    CUDA_RT_CALL(
        cudaMalloc(d_cosSinTable, sizeof(float) * projectionCount * 2));
    makeCosSinTableKernel<<<(projectionCount + 255) / 256, 256>>>(
        projectionCount, *d_cosSinTable);
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

// currentBatchSize of slices processed in one kernel launch
__global__ void backProjectionKernel(int currentBatchSize, int width,
                                     float pixelSize, int projectionCount,
                                     float roaShift, float* cosSinTable,
                                     float* sinograms, float* voxels) {
    size_t ix = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t iy = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t iz = (size_t)blockIdx.z;
    if (ix >= width || iy >= width || iz >= currentBatchSize)
        return;
    float y     = (iy + 0.5f * (1.f - width)) * pixelSize;
    float x     = (ix + 0.5f * (1.f - width)) * pixelSize;
    int   count = 0;
    float sum   = 0.f;
    for (int i = 0; i < projectionCount - 1; i++) {
        float u  = x * cosSinTable[2 * i] - y * cosSinTable[2 * i + 1];
        int   iu = static_cast<int>(u / pixelSize + width / 2.f + roaShift);
        if (iu >= 0 && iu < width) {
            sum += sinograms[iz * projectionCount * width + i * width + iu];
            count++;
        }
    }
    if (count != 0) {
        voxels[iz * width * width + iy * width + ix] =
            sum / static_cast<float>(count);
    } else {
        voxels[iz * width * width + iy * width + ix] = 0.f;
    }
}

__global__ void normalizeVoxelsKernel(float* voxels, int* counter,
                                      long long voxelCount) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < voxelCount) {
        if (counter[i] > 0) {
            voxels[i] /= static_cast<float>(counter[i]);
        }
    }
}

void backProjection(int currentBatchSize, const float* sinogram, int width,
                    float pixelSize, int projectionCount, float roaShift,
                    float* voxels, float* d_cosSinTable) {
    float *d_sinogram, *d_voxels;
    CUDA_RT_CALL(
        cudaMalloc(&d_sinogram,
                   sizeof(float) * width * projectionCount * currentBatchSize));
    CUDA_RT_CALL(
        cudaMemcpy(d_sinogram, sinogram,
                   sizeof(float) * width * projectionCount * currentBatchSize,
                   cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMalloc(&d_voxels,
                            sizeof(float) * width * width * currentBatchSize));
    backProjectionKernel<<<dim3((width + 31) / 32, (width + 31) / 32,
                                currentBatchSize),
                           dim3(32, 32)>>>(currentBatchSize, width, pixelSize,
                                           projectionCount, roaShift,
                                           d_cosSinTable, d_sinogram, d_voxels);
    CUDA_RT_CALL(cudaMemcpy(voxels, d_voxels,
                            sizeof(float) * width * width * currentBatchSize,
                            cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaFree(d_sinogram));
    CUDA_RT_CALL(cudaFree(d_voxels));
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

// 重畳積分法(CBP): inputFolder has a raw file containing all projection angles
void cbp(const int detectorWidth, const int detectorHeight,
         const int projectionCount, const float detectorPixelSize,
         const float roaShift, const std::string& inputFolder,
         const std::string& inputFilename, const std::string& outputFolder,
         const std::string& outputPrefix) {
    // make cos and sin table on GPU
    float* d_cosSinTable;
    makeCosSinTable(projectionCount, &d_cosSinTable);

    float* d_sheppTable;
    makeSheppTable(&d_sheppTable, detectorWidth, detectorPixelSize);

    size_t freeMem, totalMem;
    CUDA_RT_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "GPU Memory: Free " << freeMem / (1024 * 1024)
              << " MB / Total " << totalMem / (1024 * 1024) << " MB"
              << std::endl;

    size_t sliceMem =
        std::max(sizeof(float) * detectorWidth * projectionCount * 2,
                 sizeof(float) * (detectorWidth * projectionCount +
                                  detectorWidth * detectorWidth));

    int sliceBatchSize = 1;
    if (freeMem < sliceMem) {
        std::cerr << "Not enough GPU memory for reconstructing one slice."
                  << std::endl;
        CUDA_RT_CALL(cudaFree(d_cosSinTable));
        CUDA_RT_CALL(cudaFree(d_sheppTable));
        return;
    } else {
        const float safetyFactor = 0.9f;
        sliceBatchSize =
            (int)(((size_t)(freeMem * safetyFactor) + sliceMem - 1) / sliceMem);
        std::cout << "Reconstructing " << sliceBatchSize
                  << " slices in one batch." << std::endl;
    }

    // file open
    std::string   inPath = getPath(inputFolder, inputFilename);
    std::ifstream inFile(inPath, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file: " << inPath << std::endl;
        return;
    }
    std::string   outFilename = outputPrefix + inputFilename;
    std::string   outPath     = getPath(outputFolder, outFilename);
    std::ofstream outFile(outPath, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error creating file: " << outPath << std::endl;
        return;
    }

    for (int h = 0; h < detectorHeight; h += sliceBatchSize) {
        int    currentBatchSize = std::min(sliceBatchSize, detectorHeight - h);
        float* sinogram = new float[(size_t)detectorWidth * projectionCount *
                                    currentBatchSize];
        float* voxels =
            new float[(size_t)detectorWidth * detectorWidth * currentBatchSize];

        // read sinograms for current batch
        for (int bh = 0; bh < currentBatchSize; bh++) {
            for (int p = 0; p < projectionCount; p++) {
                inFile.seekg(
                    (static_cast<size_t>(p) * detectorWidth * detectorHeight +
                     detectorWidth * (h + bh)) *
                        sizeof(float),
                    std::ios::beg);
                inFile.read(reinterpret_cast<char*>(sinogram +
                                                    bh * detectorWidth *
                                                        projectionCount +
                                                    p * detectorWidth),
                            detectorWidth * sizeof(float));
            }
        }

        convolution(sinogram, d_sheppTable, detectorWidth,
                    projectionCount * currentBatchSize);

        backProjection(currentBatchSize, sinogram, detectorWidth,
                       detectorPixelSize, projectionCount, roaShift, voxels,
                       d_cosSinTable);

        // write voxels for current batch
        outFile.write(reinterpret_cast<char*>(voxels),
                      sizeof(float) * detectorWidth * detectorWidth *
                          currentBatchSize);
        delete[] sinogram;
        delete[] voxels;
        std::cout << "\rProcessed slice "
                  << std::min(h + currentBatchSize, detectorHeight) << " / "
                  << detectorHeight << std::flush;
    }

    // // read  each sinogram from projection file
    // for (int h = 0; h < detectorHeight; h++) {
    //     float* sinogram = new float[(size_t)detectorWidth * projectionCount];
    //     float* voxels   = new float[(size_t)detectorWidth * detectorWidth];

    //     for (int p = 0; p < projectionCount; p++) {
    //         inFile.seekg(
    //             (static_cast<size_t>(p) * detectorWidth * detectorHeight +
    //              detectorWidth * h) *
    //                 sizeof(float),
    //             std::ios::beg);
    //         inFile.read(reinterpret_cast<char*>(sinogram + p *
    //         detectorWidth),
    //                     detectorWidth * sizeof(float));
    //     }

    //     convolution(sinogram, d_sheppTable, detectorWidth, projectionCount);

    //     backProjection(sinogram, detectorWidth, detectorPixelSize,
    //                    projectionCount, roaShift, voxels, d_cosSinTable);

    //     outFile.write(reinterpret_cast<char*>(voxels),
    //                   sizeof(float) * detectorWidth * detectorWidth);

    //     delete[] sinogram;
    //     delete[] voxels;
    //     std::cout << "\rProcessed slice " << h + 1 << " / " << detectorHeight
    //               << std::flush;
    // }

    std::cout << std::endl;
    CUDA_RT_CALL(cudaFree(d_cosSinTable));
    CUDA_RT_CALL(cudaFree(d_sheppTable));
    inFile.close();
    outFile.close();
}

// 重畳積分法(CBP): inputFolder has raw files for each projection angle
void cbp2(const int detectorWidth, const int detectorHeight,
          const int projectionCount, const float detectorPixelSize,
          const float roaShift, const std::string& inputFolder,
          const std::string& inputFilenamePrefix,
          const std::string& outputFolder, const std::string& outputPrefix) {
    // make cos and sin table on GPU
    float* d_cosSinTable;
    makeCosSinTable(projectionCount, &d_cosSinTable);

    float* d_sheppTable;
    makeSheppTable(&d_sheppTable, detectorWidth, detectorPixelSize);

    size_t freeMem, totalMem;
    CUDA_RT_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "GPU Memory: Free " << freeMem / (1024 * 1024)
              << " MB / Total " << totalMem / (1024 * 1024) << " MB"
              << std::endl;

    size_t sliceMem =
        std::max(sizeof(float) * detectorWidth * projectionCount * 2,
                 sizeof(float) * (detectorWidth * projectionCount +
                                  detectorWidth * detectorWidth));

    int sliceBatchSize = 1;
    if (freeMem < sliceMem) {
        std::cerr << "Not enough GPU memory for reconstructing one slice."
                  << std::endl;
        CUDA_RT_CALL(cudaFree(d_cosSinTable));
        CUDA_RT_CALL(cudaFree(d_sheppTable));
        return;
    } else {
        const float safetyFactor = 0.9f;
        sliceBatchSize =
            (int)(((size_t)(freeMem * safetyFactor) + sliceMem - 1) / sliceMem);
        sliceBatchSize = std::min(sliceBatchSize, 128);  // limit max batch size
        std::cout << "Reconstructing " << sliceBatchSize
                  << " slices in one batch." << std::endl;
    }

    // // file open
    // std::string   inPath = getPath(inputFolder, inputFilename);
    // std::ifstream inFile(inPath, std::ios::binary);
    // if (!inFile) {
    //     std::cerr << "Error opening file: " << inPath << std::endl;
    //     return;
    // }

    // open all projection files
    std::vector<std::string> inPaths;
    // std::vector<std::ifstream> inFiles(projectionCount);
    for (int p = 0; p < projectionCount; p++) {
        std::ostringstream ss;
        ss << inputFilenamePrefix << std::setw(4) << std::setfill('0') << p
           << "_" << detectorWidth << "x" << detectorHeight;
        inPaths.push_back(getPath(inputFolder, ss.str()));
        // std::string inPath = getPath(inputFolder, ss.str());
        // inFiles[p].open(inPath, std::ios::binary);
        // if (!inFiles[p]) {
        //     std::cerr << "Error opening file: " << inPath << std::endl;
        //     return;
        // }
    }
    // for (const auto& path : inPaths) {
    //     std::ifstream testFile(path,  std::ios::binary);
    //     if (!testFile) {
    //         std::cerr << "Error opening file: " << path << std::endl;
    //         return;
    //     }
    // }

    std::ostringstream ss;
    ss << outputPrefix << inputFilenamePrefix << detectorWidth << "x" << detectorWidth << "x"
       << detectorHeight;
    std::string   outFilename = ss.str();
    std::string   outPath     = getPath(outputFolder, outFilename);
    std::ofstream outFile(outPath, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error creating file: " << outPath << std::endl;
        return;
    }

    for (int h = 0; h < detectorHeight; h += sliceBatchSize) {
        int    currentBatchSize = std::min(sliceBatchSize, detectorHeight - h);
        float* sinogram = new float[(size_t)detectorWidth * projectionCount *
                                    currentBatchSize];
        float* voxels =
            new float[(size_t)detectorWidth * detectorWidth * currentBatchSize];

        // read sinograms for current batch
        // for (int bh = 0; bh < currentBatchSize; bh++) {
        for (int p = 0; p < projectionCount; p++) {
            // read from each projection file
            std::ifstream inFile(inPaths[p], std::ios::binary);
            if (!inFile) {
                std::cerr << "Error opening file: " << inPaths[p]
                            << std::endl;
                return;
            }
            for (int bh = 0; bh < currentBatchSize; bh++) {
                inFile.seekg(static_cast<size_t>(detectorWidth) * (h + bh) *
                                 sizeof(float),
                             std::ios::beg);
                inFile.read(reinterpret_cast<char*>(sinogram +
                                                    bh * detectorWidth *
                                                        projectionCount +
                                                    p * detectorWidth),
                            detectorWidth * sizeof(float));
            }
        }

        convolution(sinogram, d_sheppTable, detectorWidth,
                    projectionCount * currentBatchSize);

        backProjection(currentBatchSize, sinogram, detectorWidth,
                       detectorPixelSize, projectionCount, roaShift, voxels,
                       d_cosSinTable);

        // write voxels for current batch
        outFile.write(reinterpret_cast<char*>(voxels),
                      sizeof(float) * detectorWidth * detectorWidth *
                          currentBatchSize);
        delete[] sinogram;
        delete[] voxels;
        std::cout << "\rProcessed slice "
                  << std::min(h + currentBatchSize, detectorHeight) << " / "
                  << detectorHeight << std::flush;
    }

    std::cout << std::endl;
    CUDA_RT_CALL(cudaFree(d_cosSinTable));
    CUDA_RT_CALL(cudaFree(d_sheppTable));
    // inFile.close();
    outFile.close();
}