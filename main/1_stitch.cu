#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <time.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <H5Cpp.h>

#include "cuda_util.h"
#include "file_io.h"
#include "h5_util.h"
#include "pre_process.h"
#include "stitch.h"

//! ===========================================================================
//* ---ファイル入出力設定---
std::string              data_dir      = "D:/EuroChart_20251205/";
bool                     I0_retraction = true; // I0を固定位置で撮ったかどうか
std::string              I0_filename   = "I0_20251205_044122"; // I0のファイル名
std::vector<std::string> exclude_keywords = {/*"x000", "x019"*/};
//* ---ファイル入出力設定ここまで---

//* ---パラメータ設定---
// 画素サイズの初期値(mm/pixel)
float pixel_size_mm =
    // 6.5f / 4.f * 0.001f;
// 6.5um[pixel]、4倍対物レンズ、ビニングなし、18keVは12/2以前に撮ったもの
    6.5f / 2.f * 0.001f;
// 6.5um[pixel]、2倍対物レンズ、ビニングなし、18keVは12/3以降に撮ったもの
    // 6.5f / 2.f * 2.f * 0.001f;
// 6.5um[pixel]、2倍対物レンズ, ビニング2 with mirror, iPhone AppleWatch
//! metaに"pixel_size_um"があればそちらを優先

// ROI設定 //! 0_check_roi.cuで確認してから設定すること
const int ROI_X      = 30;
const int ROI_Y      = 67;
const int ROI_WIDTH  = 904;
const int ROI_HEIGHT = 376;

// さらにビニングする(データが大きすぎる)場合
const int BINNING = 1; //! ROI_WIDTH,HEIGHTはBINNINGで割り切れる値にすること
//* ---パラメータ設定ここまで---

//* ---以降適宜変更---
std::string save_stitched_dir = data_dir + "stitched/";
std::string save_voxel_dir    = data_dir;
size_t      max_num_i0        = 100; // 最大で使用するI0画像の枚数
//! ===========================================================================

std::string i0_dir = data_dir; // I0画像のディレクトリ

using namespace H5;
namespace fs = std::filesystem;

int main() {
    clock_t start, end;

    //! stitching
    start = clock();
    std::cout << "defalut pixel size: " << pixel_size_mm << " mm" << std::endl;
    try {
        // create stitched directory if not exists
        if (fs::create_directories(save_stitched_dir)) {
            std::cout << "Created directory: " << save_stitched_dir
                      << std::endl;
        } else {
            // std::cout << "Directory already exists." << std::endl;
        }
    } catch (std::exception &e) {
        // Error handling for cases such as lack of permissions
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
    std::vector<H5std_string> h5_files =
        getH5FileList(data_dir, exclude_keywords);
    if (h5_files.empty()) {
        std::cerr << "No HDF5 files found in directory: " << data_dir
                  << std::endl;
        return -1;
    }
    if (I0_retraction) {
        i0_dir = data_dir + "I0_Data/";
    }

    H5File  I0_file(getPath(i0_dir, I0_filename, ".h5"), H5F_ACC_RDONLY);
    DataSet I0_dataset;
    if (I0_retraction) {
        I0_dataset = I0_file.openDataSet("i0_images");
    } else {
        I0_dataset = I0_file.openDataSet("images");
    }
    // DataSet   I0_dataset   = I0_file.openDataSet("images");
    DataSpace I0_dataspace = I0_dataset.getSpace();
    hsize_t   dims[3];
    I0_dataspace.getSimpleExtentDims(dims);
    std::cout << "I0 dataset dimensions: " << dims[0] << " x " << dims[1]
              << " x " << dims[2] << std::endl;
    size_t use_num_i0 = std::min(static_cast<size_t>(dims[0]), max_num_i0);
    std::vector<uint16_t> I0_imgs(use_num_i0 * dims[1] * dims[2]);
    //! offset_i0[0]はI0imagesの先頭から何枚目か
    hsize_t offset_i0[3] = {0, 0, 0};
    hsize_t count_i0[3]  = {use_num_i0, dims[1], dims[2]};
    I0_dataspace.selectHyperslab(H5S_SELECT_SET, count_i0, offset_i0);
    DataSpace memspace_i0(3, count_i0);
    I0_dataset.read(I0_imgs.data(), PredType::NATIVE_UINT16, memspace_i0,
                    I0_dataspace);
    std::vector<float> bright_img(dims[1] * dims[2], 0.f);
    averageImages(use_num_i0, dims[1] * dims[2], I0_imgs, bright_img);
    I0_imgs.clear();

    // std::ofstream bright_file(getPath(data_dir, "bright_img"),
    //                           std::ios::binary);
    // if (!bright_file) {
    //     std::cerr << "Error opening bright image output file." << std::endl;
    //     return -1;
    // }
    // bright_file.write(reinterpret_cast<const char *>(bright_img.data()),
    //                   bright_img.size() * sizeof(float));
    // bright_file.close();
    // std::cout << "Saved bright image to file: "
    //           << getPath(data_dir, "bright_img") << std::endl;

    std::vector<float> dark_img(dims[1] * dims[2], 0.f);

    // std::vector<H5File>  h5_files_opened(h5_files.size());
    // std::vector<DataSet> datasets(h5_files.size());
    std::vector<float> x_positions(h5_files.size());
    std::vector<float> y_positions(h5_files.size());
    std::vector<bool>  is_counter_clockwise(h5_files.size());

    hsize_t out_dims[3];
    for (size_t i = 0; i < h5_files.size(); i++) {
        H5File h5_file_opened = H5File(h5_files[i], H5F_ACC_RDONLY);
        // std::cout << "Opened file: " << h5_files[i] << std::endl;
        Group   meta    = h5_file_opened.openGroup("meta");
        DataSet dataset = h5_file_opened.openDataSet("images");
        if (i == 0) {
            DataSpace dataspace = dataset.getSpace();
            dataspace.getSimpleExtentDims(dims);
            std::cout << "Dataset dimensions: " << dims[0] << " x " << dims[1]
                      << " x " << dims[2] << std::endl;
            if (meta.attrExists("pixel_size_um")) {
                Attribute pixel_size_attr = meta.openAttribute("pixel_size_um");
                double    pixel_size_um;
                pixel_size_attr.read(pixel_size_attr.getDataType(),
                                     &pixel_size_um);
                pixel_size_mm = static_cast<float>(pixel_size_um * 0.001);
                std::cout << "Pixel size from metadata: " << pixel_size_mm
                          << " mm" << std::endl;
            }
        }
        Attribute           pos_attr = meta.openAttribute("grid_pos_xy_mm");
        std::vector<double> grid_pos(2);
        pos_attr.read(pos_attr.getDataType(), grid_pos.data());
        // std::cout << "Grid Position (mm): X=" << grid_pos[0]
        //           << ", Y=" << grid_pos[1] << std::endl;
        x_positions[i]     = static_cast<float>(grid_pos[0]);
        y_positions[i]     = static_cast<float>(-grid_pos[1]);
        Attribute ccw_attr = meta.openAttribute("counter_clockwise");
        uint8_t   ccw;
        ccw_attr.read(ccw_attr.getDataType(), &ccw);
        // std::cout << "Counter Clockwise: " << static_cast<int>(ccw)
        //           << std::endl; // 1 for true, 0 for false
        is_counter_clockwise[i] = (ccw == 1);
    }

    out_dims[0]   = dims[0];
    out_dims[2]   = ROI_WIDTH / BINNING;
    out_dims[1]   = ROI_HEIGHT / BINNING;
    pixel_size_mm = pixel_size_mm * static_cast<float>(BINNING);
    std::cout << "Using pixel size: " << pixel_size_mm << " mm" << std::endl;

    // images of same rotation angle of each position
    std::vector<float>    float_imgs(dims[1] * dims[2] * h5_files.size());
    std::vector<uint16_t> imgs(dims[1] * dims[2] * h5_files.size());

    // prepare for stitching
    size_t out_width, out_height;

    int  grid_x_size, grid_y_size;
    auto grid_index_map =
        createGridIndexMap(x_positions, y_positions, out_dims[2], out_dims[1],
                           pixel_size_mm, grid_x_size, grid_y_size, 0.1f);
    out_width = (size_t)std::ceilf(
                    *std::max_element(x_positions.begin(), x_positions.end())) +
                out_dims[2];
    out_height = (size_t)std::ceilf(*std::max_element(y_positions.begin(),
                                                      y_positions.end())) +
                 out_dims[1];
    // std::cout << "Stitch will use memory size: "
    //           << ((sizeof(float) * 2 + sizeof(uint16_t)) * h5_files.size() *
    //                   dims[1] * dims[2] +
    //               sizeof(float) * ROI_WIDTH * ROI_HEIGHT +
    //               (sizeof(float) * 2) * out_width * out_height) /
    //                  (1024.0f * 1024.0f * 1024.0f)
    //           << " GB" << std::endl;
    std::vector<std::vector<float>> weight_maps(
        h5_files.size(), std::vector<float>(out_dims[2] * out_dims[1],
                                            1.f)); // initialize to 1.0f
    computeWeightMaps(weight_maps, x_positions, y_positions, out_dims[2],
                      out_dims[1], grid_index_map);

    std::vector<float> stitched_image(out_width * out_height);
    std::vector<float> weight_sum(out_width * out_height);

    for (size_t n_deg = 0; n_deg < dims[0]; n_deg += 1 /*dims[0] - 1*/) {
        std::cout << "\rStitching image angle: " << std::setw(4)
                  << std::setfill(' ') << n_deg + 1 << " / " << dims[0]
                  << std::flush;
        hsize_t offset[3] = {0, 0, 0};
        hsize_t count[3]  = {1, dims[1], dims[2]};
        for (size_t i = 0; i < h5_files.size(); i++) {
            if (is_counter_clockwise[i]) {
                offset[0] = n_deg;
            } else {
                offset[0] = dims[0] - 1 - n_deg;
            }
            H5File    h5_file_opened = H5File(h5_files[i], H5F_ACC_RDONLY);
            DataSet   dataset        = h5_file_opened.openDataSet("images");
            DataSpace dataspace      = dataset.getSpace();
            DataSpace memspace(3, count);
            dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
            dataset.read(imgs.data() + i * dims[1] * dims[2],
                         PredType::NATIVE_UINT16, memspace, dataspace);
        }
        // std::cout << std::endl;

        for (size_t i = 0; i < imgs.size(); i++) {
            float_imgs[i] = static_cast<float>(imgs[i]);
        }
        preProcessing(h5_files.size(), dims[2], dims[1], dark_img, bright_img,
                      ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, BINNING, float_imgs);

        std::fill(stitched_image.begin(), stitched_image.end(), 0.f);
        std::fill(weight_sum.begin(), weight_sum.end(), 0.f);
        for (auto &[grid_pos, index] : grid_index_map) {
            const float *img = &float_imgs[index * out_dims[2] * out_dims[1]];
            auto        &weight_map = weight_maps[index];
            float        x_offset   = x_positions[index];
            float        y_offset   = y_positions[index];
            int          x_start    = (int)std::ceilf(x_offset);
            int          y_start    = (int)std::ceilf(y_offset);
            int x_end = (int)std::floorf(x_offset + (float)out_dims[2]);
            int y_end = (int)std::floorf(y_offset + (float)out_dims[1]);

#pragma omp parallel for collapse(2)
            for (int x = x_start; x < x_end; x++) {
                for (int y = y_start; y < y_end; y++) {
                    float x_local = (float)x - x_offset;
                    float y_local = (float)y - y_offset;
                    // bilinear interpolation
                    int   x0          = (int)std::floorf(x_local);
                    int   x1          = (int)std::ceilf(x_local);
                    int   y0          = (int)std::floorf(y_local);
                    int   y1          = (int)std::ceilf(y_local);
                    float dx          = x_local - (float)x0;
                    float dy          = y_local - (float)y0;
                    float w00         = (1.f - dx) * (1.f - dy);
                    float w01         = (1.f - dx) * dy;
                    float w10         = dx * (1.f - dy);
                    float w11         = dx * dy;
                    float pixel_value = img[y0 * out_dims[2] + x0] * w00 +
                                        img[y1 * out_dims[2] + x0] * w01 +
                                        img[y0 * out_dims[2] + x1] * w10 +
                                        img[y1 * out_dims[2] + x1] * w11;
                    float weight_value =
                        weight_map[y0 * out_dims[2] + x0] * w00 +
                        weight_map[y1 * out_dims[2] + x0] * w01 +
                        weight_map[y0 * out_dims[2] + x1] * w10 +
                        weight_map[y1 * out_dims[2] + x1] * w11;
                    stitched_image[y * out_width + x] +=
                        pixel_value * weight_value;
                    weight_sum[y * out_width + x] += weight_value;
                }
            }
        }

        // Normalize by weight sum
        for (size_t i = 0; i < out_width * out_height; i++) {
            if (weight_sum[i] > 0.f) {
                stitched_image[i] /= weight_sum[i];
            }
        }

        // 出力
        std::ostringstream ss;
        ss << "stitched_image_" << std::setw(4) << std::setfill('0') << n_deg
           << "_" << out_width << "x" << out_height;

        std::ofstream stitch_file(getPath(save_stitched_dir, ss.str()),
                                  std::ios::binary);
        if (!stitch_file) {
            std::cerr << "Error opening stitched image output file: "
                      << getPath(save_stitched_dir, ss.str()) << std::endl;
            return -1;
        }
        stitch_file.write(reinterpret_cast<const char *>(stitched_image.data()),
                          stitched_image.size() * sizeof(float));
        stitch_file.close();
    }
    std::cout << std::endl;

    float_imgs.clear();
    imgs.clear();
    bright_img.clear();
    dark_img.clear();
    weight_maps.clear();

    end                 = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Stitching time: " << elapsed_time << " seconds\n";
    std::cout << "Stitched images saved to: " << save_stitched_dir << std::endl;
    return 0;
}