#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>

#include <iostream>
#include <vector>

#include "cuda_util.h"
#include "pre_process.h"

__global__ void darkBrightCorrectionKernel(const int    batch_size,
                                           const int    img_size,
                                           const float* dark_img,
                                           const float* bright_img,
                                           float*       imgs) {
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    int img_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < img_size && img_idx < batch_size) {
        size_t batch_idx = (size_t)img_idx * img_size + idx;
        float  pixel     = imgs[batch_idx];
        float  divisor   = bright_img[idx] - dark_img[idx];
        if (divisor < 1e-6f) { // X線がほぼ当たってない領域
            // divisor = 1e-6f; // Prevent division by zero
            imgs[batch_idx] = -logf(1e-6f);
        } else if (pixel > bright_img[idx]) {
            imgs[batch_idx] = -logf(1.f); // 飽和している場合は1に設定
        } else {
            imgs[batch_idx] = -logf((pixel - dark_img[idx]) / divisor);
        }
    }
}

void darkBrightCorrection(const size_t img_num, const size_t img_size,
                          const std::vector<float>& dark_img,
                          const std::vector<float>& bright_img,
                          std::vector<float>&       imgs) {
    size_t freeMem, totalMem;
    CUDA_RT_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "GPU Memory: Free " << freeMem / (1024 * 1024)
              << " MB / Total " << totalMem / (1024 * 1024) << " MB"
              << std::endl;

    size_t usableMem = freeMem * 8 / 10; // 80% of free memory can be used

    float *d_dark_img, *d_bright_img, *d_imgs;
    size_t img_bytes = img_size * sizeof(float);
    CUDA_RT_CALL(cudaMalloc(&d_dark_img, img_bytes));
    CUDA_RT_CALL(cudaMalloc(&d_bright_img, img_bytes));
    CUDA_RT_CALL(cudaMemcpy(d_dark_img, dark_img.data(), img_bytes,
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_bright_img, bright_img.data(), img_bytes,
                            cudaMemcpyHostToDevice));
    usableMem -= 2 * img_bytes;
    size_t max_batch_size = usableMem / img_bytes;
    size_t batch_size     = std::min(
        max_batch_size, img_num); // Process at most img_num images at once
    // batch_size = std::min(batch_size,(size_t)128);
    size_t max_batch_bytes = batch_size * img_size * sizeof(float);
    CUDA_RT_CALL(cudaMalloc(&d_imgs, max_batch_bytes));
    std::cout << "Processing " << batch_size << " images per batch."
              << std::endl;
    for (size_t i = 0; i < img_num; i += batch_size) {
        size_t current_batch_size = std::min(batch_size, img_num - i);
        size_t current_batch_bytes =
            current_batch_size * img_size * sizeof(float);
        CUDA_RT_CALL(cudaMemcpy(d_imgs, imgs.data() + i * img_size,
                                current_batch_bytes, cudaMemcpyHostToDevice));
        dim3 block(32, 16);
        dim3 grid((img_size + block.x - 1) / block.x,
                  (current_batch_size + block.y - 1) / block.y);
        darkBrightCorrectionKernel<<<grid, block>>>(
            current_batch_size, img_size, d_dark_img, d_bright_img, d_imgs);
        CUDA_RT_CALL(cudaMemcpy(imgs.data() + i * img_size, d_imgs,
                                current_batch_bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_RT_CALL(cudaFree(d_dark_img));
    CUDA_RT_CALL(cudaFree(d_bright_img));
    CUDA_RT_CALL(cudaFree(d_imgs));
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

__global__ void setROIKernel(const int batch_size, const int img_width,
                             const int img_height, const int roi_x,
                             const int roi_y, const int roi_width,
                             const int roi_height, const float* input_imgs,
                             float* output_imgs) {
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    int idy     = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx < roi_width && idy < roi_height && img_idx < batch_size) {
        size_t out_idx =
            (size_t)img_idx * (roi_width * roi_height) + idy * roi_width + idx;
        size_t in_idx = (size_t)img_idx * (img_width * img_height) +
                        (idy + roi_y) * img_width + (idx + roi_x);
        output_imgs[out_idx] = input_imgs[in_idx];
    }
}

__global__ void binningKernel(const int binning, const int batch_size,
                              const int output_width, const int output_height,
                              const float* input_imgs, float* output_imgs) {
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    int idy      = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx  = blockIdx.z * blockDim.z + threadIdx.z;
    int out_size = output_width * output_height;
    if (idx < output_width && idy < output_height && img_idx < batch_size) {
        size_t out_idx = (size_t)img_idx * out_size + idy * output_width + idx;
        float  sum     = 0.f;
        for (int by = 0; by < binning; by++) {
            for (int bx = 0; bx < binning; bx++) {
                int    in_x   = idx * binning + bx;
                int    in_y   = idy * binning + by;
                size_t in_idx = img_idx * (output_width * binning) *
                                    (output_height * binning) +
                                in_y * (output_width * binning) + in_x;
                sum += input_imgs[in_idx];
            }
        }
        output_imgs[out_idx] = sum / (binning * binning);
    }
}

void preProcessing(const size_t img_num, const size_t img_width,
                   const size_t img_height, const std::vector<float>& dark_img,
                   const std::vector<float>& bright_img, const int roi_x,
                   const int roi_y, const int roi_width, const int roi_height,
                   const int binning, std::vector<float>& imgs) {
    size_t freeMem, totalMem;
    CUDA_RT_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    // std::cout << "GPU Memory: Free " << freeMem / (1024 * 1024)
    //           << " MB / Total " << totalMem / (1024 * 1024) << " MB"
    //           << std::endl;

    float *d_dark_img, *d_bright_img, *d_imgs, *d_roi_imgs, *d_binned_imgs;
    size_t img_size         = img_width * img_height;
    size_t img_bytes        = img_size * sizeof(float);
    size_t output_img_size  = (roi_width / binning) * (roi_height / binning);
    size_t output_img_bytes = output_img_size * sizeof(float);
    CUDA_RT_CALL(cudaMalloc(&d_dark_img, img_bytes));
    CUDA_RT_CALL(cudaMalloc(&d_bright_img, img_bytes));
    CUDA_RT_CALL(cudaMemcpy(d_dark_img, dark_img.data(), img_bytes,
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_bright_img, bright_img.data(), img_bytes,
                            cudaMemcpyHostToDevice));
    freeMem -= 2 * img_bytes;
    size_t max_batch_size = freeMem / (img_size * 2 * sizeof(float)) * 9 /
                            10; // 90% of free memory will be used at most
    size_t batch_size = std::min(
        max_batch_size, img_num); // Process at most img_num images at once
    size_t max_batch_bytes = batch_size * img_size * sizeof(float);
    CUDA_RT_CALL(cudaMalloc(&d_imgs, max_batch_bytes));
    CUDA_RT_CALL(cudaMalloc(&d_roi_imgs, batch_size * roi_width * roi_height *
                                             sizeof(float)));
    // std::cout << "Processing " << batch_size << " images per batch."
    //           << std::endl;
    for (size_t i = 0; i < img_num; i += batch_size) {
        size_t current_batch_size = std::min(batch_size, img_num - i);
        size_t current_batch_bytes =
            current_batch_size * img_size * sizeof(float);
        size_t current_batch_output_bytes =
            current_batch_size * output_img_size * sizeof(float);
        CUDA_RT_CALL(cudaMemcpy(d_imgs, imgs.data() + i * img_size,
                                current_batch_bytes, cudaMemcpyHostToDevice));
        dim3 block(32, 16);
        dim3 grid((img_size + block.x - 1) / block.x,
                  (current_batch_size + block.y - 1) / block.y);
        darkBrightCorrectionKernel<<<grid, block>>>(
            current_batch_size, img_size, d_dark_img, d_bright_img, d_imgs);
        grid = dim3((roi_width + block.x - 1) / block.x,
                    (roi_height + block.y - 1) / block.y, current_batch_size);
        setROIKernel<<<grid, block>>>(current_batch_size, img_width, img_height,
                                      roi_x, roi_y, roi_width, roi_height,
                                      d_imgs, d_roi_imgs);
        binningKernel<<<grid, block>>>(
            binning, current_batch_size, roi_width / binning,
            roi_height / binning, d_roi_imgs, d_imgs);
        CUDA_RT_CALL(cudaMemcpy(imgs.data() + i * output_img_size, d_imgs,
                                current_batch_output_bytes,
                                cudaMemcpyDeviceToHost));
    }
    CUDA_RT_CALL(cudaFree(d_dark_img));
    CUDA_RT_CALL(cudaFree(d_bright_img));
    CUDA_RT_CALL(cudaFree(d_imgs));
    CUDA_RT_CALL(cudaFree(d_roi_imgs));
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

void averageImages(const size_t img_num, const size_t img_size,
                   const std::vector<uint16_t>& imgs, std::vector<float>& img) {
    // std::fill(img.begin(), img.end(), 0.f);
    for (size_t i = 0; i < img_num; i++) {
        for (size_t j = 0; j < img_size; j++) {
            img[j] += static_cast<float>(imgs[i * img_size + j]);
        }
    }
#pragma omp parallel for
    for (int j = 0; j < img_size; j++) {
        img[j] /= static_cast<float>(img_num);
    }
    std::cout << "Averaged " << img_num << " images." << std::endl;
}
