#pragma once
#include <vector>
#include <map>


// 位置[mm]をpixel単位に変換する
void normalizePositions(
    std::vector<float>& positions,
    float min_value,
    float pixel_size
);

// 許容誤差内には同一インデックスを割り当てる
std::vector<int> assignIndicesWithTolerance(
    const std::vector<float>& values,
    const float tolerance,
    int& num_indices,
    float& min_value
);

// グリッドの位置とインデックスの対応マップを作成する
std::map<std::pair<int, int>, int> createGridIndexMap(
    std::vector<float>& x_positions,
    std::vector<float>& y_positions,
    const int width,
    const int height,
    const float pixel_size,
    int& grid_x_size,
    int& grid_y_size,
    const float tolerance_percent = 0.1f
);

// 各画像の重みマップを計算する
void computeWeightMaps(
    std::vector<std::vector<float>>& weight_maps,
    const std::vector<float>& x_positions,
    const std::vector<float>& y_positions,
    const int width,
    const int height,
    std::map<std::pair<int, int>, int>& grid_index_map
);

// 画像群を位置に基づいてスティッチする
std::vector<float> stitchImages(
    const std::vector<std::vector<float>>& images,
    const int width,
    const int height,
    std::vector<float>& x_positions,
    std::vector<float>& y_positions,
    const float pixel_size,
    size_t& out_width,
    size_t& out_height
);

// 画像群を位置に基づいてスティッチする（1次元配列版）
std::vector<float> stitchAllImages(
    const std::vector<float>& images,
    const int num_images,
    const int width,
    const int height,
    std::vector<float>& x_positions,
    std::vector<float>& y_positions,
    const float pixel_size,
    size_t& out_width,
    size_t& out_height
);