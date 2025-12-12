#include <time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "calculate_cor.h"
#include "file_io.h"

//! --- パラメータ設定ここから ---
std::string data_dir        = "D:/JIMACTchart_20251205/stitched/";
std::string filename_0deg   = "stitched_image_0000_992x1314";
std::string filename_180deg = "stitched_image_4500_992x1314";

int width  = 992;
int height = 1314;
float max_shift = 0.1f * width; // 最大シフト量を画像幅の20%に設定
float step_size = 0.1f;
//! --- パラメータ設定ここまで ---

int main() {
    clock_t start, end;
    start = clock();

    int size   = width * height;

    std::vector<float> projection_0deg(size);
    std::vector<float> projection_180deg(size);
    std::ifstream file_0deg(getPath(data_dir, filename_0deg), std::ios::binary);
    std::ifstream file_180deg(getPath(data_dir, filename_180deg),
                              std::ios::binary);
    if (!file_0deg || !file_180deg) {
        std::cerr << "Error opening files.\n";
        return -1;
    }
    file_0deg.read(reinterpret_cast<char *>(projection_0deg.data()),
                   size * sizeof(float));
    file_180deg.read(reinterpret_cast<char *>(projection_180deg.data()),
                     size * sizeof(float));
    file_0deg.close();
    file_180deg.close();

    float cor = calculateCenterOfRotation(projection_0deg, projection_180deg,
                                          width, height, max_shift, step_size);
    std::cout << "Calculated Center of Rotation: " << cor << " pixels\n";

    end = clock();

    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed time: " << elapsed_time << " seconds\n";

    return 0;
}