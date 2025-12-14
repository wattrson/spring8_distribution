#include <omp.h>
#include <time.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <H5Cpp.h>

#include "file_io.h"
#include "pre_process.h"

//! ===========================================================================
//* ---ファイル入出力設定---
std::string data_dir      = "D:/EuroChart_20251205/";
bool        I0_retraction = true; // I0を固定位置で撮ったかどうか
std::string I0_filename   = "I0_20251205_044122"; // I0のファイル名
std::string image_filename =
    "EuroChart_20251205_044146_x000_y001_CCW"; // ROIを設定するための代表ファイル
//* ---ファイル入出力設定ここまで---

//* ---以降適宜変更--
size_t max_num_i0 = 100; // 最大で使用するI0画像の枚数
//! ===========================================================================

std::string i0_dir = data_dir; // I0画像のディレクトリ

using namespace H5;
namespace fs = std::filesystem;

int main() {
    clock_t start, end;

    start = clock();
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
    DataSpace I0_dataspace = I0_dataset.getSpace();
    hsize_t   dims[3];
    I0_dataspace.getSimpleExtentDims(dims);
    std::cout << "I0 dataset dimensions: " << dims[0] << " x " << dims[1]
              << " x " << dims[2] << std::endl;
    size_t use_num_i0 = std::min(static_cast<size_t>(dims[0]), max_num_i0);
    std::vector<uint16_t> I0_imgs(use_num_i0 * dims[1] * dims[2]);
    //! offset_i0[0]はI0imagesの先頭から何枚目から読み込むか
    hsize_t offset_i0[3] = {0, 0, 0};
    hsize_t count_i0[3]  = {use_num_i0, dims[1], dims[2]};
    I0_dataspace.selectHyperslab(H5S_SELECT_SET, count_i0, offset_i0);
    DataSpace memspace_i0(3, count_i0);
    I0_dataset.read(I0_imgs.data(), PredType::NATIVE_UINT16, memspace_i0,
                    I0_dataspace);
    std::vector<float> bright_img(dims[1] * dims[2], 0.f);
    averageImages(use_num_i0, dims[1] * dims[2], I0_imgs, bright_img);
    I0_imgs.clear();
    std::ostringstream ss;
    ss << "bright_img_" << dims[2] << "x" << dims[1];
    std::ofstream bright_file(getPath(data_dir, ss.str()),
                              std::ios::binary);
    if (!bright_file) {
        std::cerr << "Error opening bright image output file." << std::endl;
        return -1;
    }
    bright_file.write(reinterpret_cast<const char *>(bright_img.data()),
                      bright_img.size() * sizeof(float));
    bright_file.close();
    std::cout << "Saved bright image to file: "
              << getPath(data_dir, ss.str()) << std::endl;

    std::vector<float> dark_img(dims[1] * dims[2],
                                0.f); // now dark image is zero

    H5File image_file(getPath(data_dir, image_filename, ".h5"), H5F_ACC_RDONLY);
    DataSet   image_dataset   = image_file.openDataSet("images");
    DataSpace image_dataspace = image_dataset.getSpace();
    hsize_t   image_dims[3];
    image_dataspace.getSimpleExtentDims(image_dims);
    std::cout << "Image dataset dimensions: " << image_dims[0] << " x "
              << image_dims[1] << " x " << image_dims[2] << std::endl;
    std::vector<uint16_t> image_img(image_dims[0] * image_dims[1] *
                                    image_dims[2]);
    image_dataset.read(image_img.data(), PredType::NATIVE_UINT16);
    std::vector<float> image_float(image_dims[0] * image_dims[1] *
                                   image_dims[2]);
    // #pragma omp parallel for
    for (size_t i = 0; i < image_float.size(); ++i) {
        image_float[i] = static_cast<float>(image_img[i]);
    }
    image_img.clear();
    darkBrightCorrection(image_dims[0], image_dims[1] * image_dims[2], dark_img,
                         bright_img, image_float);

    ss.str("");
    ss.clear();
    ss << "corrected_image_" << image_dims[2] << "x" << image_dims[1] << "x" << image_dims[0];
    std::ofstream corrected_file(getPath(data_dir, ss.str()),
                                 std::ios::binary);
    if (!corrected_file) {
        std::cerr << "Error opening corrected image output file." << std::endl;
        return -1;
    }
    corrected_file.write(reinterpret_cast<const char *>(image_float.data()),
                         image_float.size() * sizeof(float));
    corrected_file.close();
    std::cout << "Saved corrected image to file: "
              << getPath(data_dir, ss.str()) << std::endl;

    end = clock();
    std::cout << "Processing time: " << double(end - start) / CLOCKS_PER_SEC
              << " sec" << std::endl;

    return 0;
}
