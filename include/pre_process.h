#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>

__global__ void darkBrightCorrectionKernel(const int    batch_size,
                                           const int    img_size,
                                           const float* dark_img,
                                           const float* bright_img,
                                           float*       imgs);

void darkBrightCorrection(const size_t img_num, const size_t img_size,
                          const std::vector<float>& dark_img,
                          const std::vector<float>& bright_img,
                          std::vector<float>&       imgs);

__global__ void setROIKernel(const int batch_size, const int img_width,
                             const int img_height, const int roi_x,
                             const int roi_y, const int roi_width,
                             const int roi_height, const float* input_imgs,
                             float* output_imgs);

__global__ void binningKernel(const int binning, const int batch_size,
                              const int output_width, const int output_height,
                              const float* input_imgs, float* output_imgs);

void preProcessing(const size_t img_num, const size_t img_width,
                   const size_t img_height, const std::vector<float>& dark_img,
                   const std::vector<float>& bright_img, const int roi_x,
                   const int roi_y, const int roi_width, const int roi_height,
                   const int binning, std::vector<float>& imgs);

void averageImages(const size_t img_num, const size_t img_size,
                   const std::vector<uint16_t>& imgs, std::vector<float>& img);