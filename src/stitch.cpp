#include "stitch.h"

#include <omp.h>
#include <time.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

// 位置[mm]をpixel単位に変換する
void        normalizePositions(std::vector<float>& positions, float min_value,
                               float pixel_size) {
#pragma omp parallel for
    for (int i = 0; i < positions.size(); i++) {
        positions[i] = (positions[i] - min_value) / pixel_size;
    }
}

std::vector<int> assignIndicesWithTolerance(const std::vector<float>& values,
                                            const float               tolerance,
                                            int&   num_indices,
                                            float& min_value) {
    int                                n = values.size();
    std::vector<std::pair<float, int>> value_index_pairs(n);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        value_index_pairs[i] = {values[i], i};
    }
    std::sort(value_index_pairs.begin(), value_index_pairs.end());
    min_value = value_index_pairs.front().first;
    std::vector<int> indices(n, -1);
    int              current_index = 0;
    float            current_value = value_index_pairs[0].first;
    for (const auto& pair : value_index_pairs) {
        if (pair.first - current_value > tolerance) {
            current_index++;
            current_value = pair.first;
        }
        indices[pair.second] = current_index;
    }
    num_indices = current_index + 1;
    return indices;
}

std::map<std::pair<int, int>, int>
createGridIndexMap(std::vector<float>& x_positions,
                   std::vector<float>& y_positions, const int width,
                   const int height, const float pixel_size, int& grid_x_size,
                   int& grid_y_size, const float tolerance_percent) {
    float            x_tolerance = tolerance_percent * pixel_size * width;
    float            y_tolerance = tolerance_percent * pixel_size * height;
    float            x_min, y_min;
    std::vector<int> x_indices = assignIndicesWithTolerance(
        x_positions, x_tolerance, grid_x_size, x_min);
    normalizePositions(x_positions, x_min, pixel_size);
    std::vector<int> y_indices = assignIndicesWithTolerance(
        y_positions, y_tolerance, grid_y_size, y_min);
    normalizePositions(y_positions, y_min, pixel_size);

    std::map<std::pair<int, int>, int> grid_index_map;
    int                                n = x_positions.size();

    for (int i = 0; i < n; i++) {
        grid_index_map[{x_indices[i], y_indices[i]}] = i;
    }

    return grid_index_map;
}

void computeWeightMaps(std::vector<std::vector<float>>& weight_maps,
                       const std::vector<float>&        x_positions,
                       const std::vector<float>& y_positions, const int width,
                       const int                           height,
                       std::map<std::pair<int, int>, int>& grid_index_map) {
    std::vector<std::pair<std::pair<int, int>, int>> grid_entries(
        grid_index_map.begin(), grid_index_map.end());
#pragma omp parallel for
    for (int i = 0; i < grid_entries.size(); i++) {
        const auto& entry = grid_entries[i];
        // for (const auto& entry : grid_index_map){
        int   current_x     = entry.first.first;
        int   current_y     = entry.first.second;
        int   current_index = entry.second;
        auto& weight_map    = weight_maps[current_index];
        std::vector<std::pair<int, int>> neighbor_grids = {
            {current_x - 1, current_y},
            {current_x + 1, current_y},
            {current_x, current_y - 1},
            {current_x, current_y + 1}};
        for (const auto& neighbor : neighbor_grids) {
            auto it = grid_index_map.find(neighbor);
            if (it != grid_index_map.end()) {
                int neighbor_x     = neighbor.first;
                int neighbor_y     = neighbor.second;
                int neighbor_index = it->second;

                // compute overlap in local coordinates of current image
                float overlap_start_x_f =
                    std::max(0.f, x_positions[neighbor_index] -
                                      x_positions[current_index]);
                float overlap_end_x_f =
                    std::min((float)width, x_positions[neighbor_index] -
                                               x_positions[current_index] +
                                               (float)width);
                float overlap_start_y_f =
                    std::max(0.f, y_positions[neighbor_index] -
                                      y_positions[current_index]);
                float overlap_end_y_f =
                    std::min((float)height, y_positions[neighbor_index] -
                                                y_positions[current_index] +
                                                (float)height);
                int overlap_start_x = (int)std::ceilf(overlap_start_x_f);
                int overlap_end_x   = (int)std::floorf(overlap_end_x_f);
                int overlap_start_y = (int)std::ceilf(overlap_start_y_f);
                int overlap_end_y   = (int)std::floorf(overlap_end_y_f);

                int overlap_width  = overlap_end_x - overlap_start_x;
                int overlap_height = overlap_end_y - overlap_start_y;
                if (overlap_width <= 0 || overlap_height <= 0) {
                    continue;
                }
                if (neighbor == neighbor_grids[0]) { // left
                    for (int x = overlap_start_x; x < overlap_end_x; x++) {
                        float weight = (float)(x - overlap_start_x_f) /
                                       (float)(overlap_width);
                        for (int y = overlap_start_y; y < overlap_end_y; y++) {
                            weight_map[y * width + x] *= weight;
                        }
                    }
                }
                if (neighbor == neighbor_grids[1]) { // right
                    for (int x = overlap_start_x; x < overlap_end_x; x++) {
                        float weight = (float)(overlap_end_x_f - x) /
                                       (float)(overlap_width);
                        for (int y = overlap_start_y; y < overlap_end_y; y++) {
                            weight_map[y * width + x] *= weight;
                        }
                    }
                }
                if (neighbor == neighbor_grids[2]) { // top
                    for (int y = overlap_start_y; y < overlap_end_y; y++) {
                        float weight = (float)(y - overlap_start_y_f) /
                                       (float)(overlap_height);
                        for (int x = overlap_start_x; x < overlap_end_x; x++) {
                            weight_map[y * width + x] *= weight;
                        }
                    }
                }
                if (neighbor == neighbor_grids[3]) { // bottom
                    for (int y = overlap_start_y; y < overlap_end_y; y++) {
                        float weight = (float)(overlap_end_y_f - y) /
                                       (float)(overlap_height);
                        for (int x = overlap_start_x; x < overlap_end_x; x++) {
                            weight_map[y * width + x] *= weight;
                        }
                    }
                }
            }
        }
    }
}

std::vector<float> stitchImages(const std::vector<std::vector<float>>& images,
                                const int width, const int height,
                                std::vector<float>& x_positions,
                                std::vector<float>& y_positions,
                                const float pixel_size, size_t& out_width,
                                size_t& out_height) {
    clock_t start, end;
    start = clock();
    int  grid_x_size, grid_y_size;
    auto grid_index_map =
        createGridIndexMap(x_positions, y_positions, width, height, pixel_size,
                           grid_x_size, grid_y_size, 0.1f);
    out_width = (size_t)std::ceilf(
                    *std::max_element(x_positions.begin(), x_positions.end())) +
                width;
    out_height = (size_t)std::ceilf(*std::max_element(y_positions.begin(),
                                                      y_positions.end())) +
                 height;
    std::vector<std::vector<float>> weight_maps(
        images.size(),
        std::vector<float>(width * height, 1.f)); // initialize to 1.0f
    computeWeightMaps(weight_maps, x_positions, y_positions, width, height,
                      grid_index_map);
    std::vector<float> stitched_image(out_width * out_height, 0.0f);
    std::vector<float> weight_sum(out_width * out_height, 0.0f);
    for (auto& [grid_pos, index] : grid_index_map) {
        const auto& img        = images[index];
        auto&       weight_map = weight_maps[index];
        float       x_offset   = x_positions[index];
        float       y_offset   = y_positions[index];
        int         x_start    = (int)std::ceilf(x_offset);
        int         y_start    = (int)std::ceilf(y_offset);
        int         x_end      = (int)std::floorf(x_offset + (float)width);
        int         y_end      = (int)std::floorf(y_offset + (float)height);

        for (int x = x_start; x < x_end; x++) {
            for (int y = y_start; y < y_end; y++) {
                float x_local = (float)x - x_offset;
                float y_local = (float)y - y_offset;
                // bilinear interpolation
                int   x0  = (int)std::floorf(x_local);
                int   x1  = (int)std::ceilf(x_local);
                int   y0  = (int)std::floorf(y_local);
                int   y1  = (int)std::ceilf(y_local);
                float dx  = x_local - (float)x0;
                float dy  = y_local - (float)y0;
                float w00 = (1.f - dx) * (1.f - dy);
                float w01 = (1.f - dx) * dy;
                float w10 = dx * (1.f - dy);
                float w11 = dx * dy;
                float pixel_value =
                    img[y0 * width + x0] * w00 + img[y1 * width + x0] * w01 +
                    img[y0 * width + x1] * w10 + img[y1 * width + x1] * w11;
                float weight_value = weight_map[y0 * width + x0] * w00 +
                                     weight_map[y1 * width + x0] * w01 +
                                     weight_map[y0 * width + x1] * w10 +
                                     weight_map[y1 * width + x1] * w11;
                stitched_image[y * out_width + x] += pixel_value * weight_value;
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
    end            = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Stitching completed in " << elapsed << " seconds.\n";
    return stitched_image;
}

std::vector<float> stitchAllImages(const std::vector<float>& images,
                                   const int num_images, const int width,
                                   const int           height,
                                   std::vector<float>& x_positions,
                                   std::vector<float>& y_positions,
                                   const float pixel_size, size_t& out_width,
                                   size_t& out_height) {
    clock_t start, end;
    start = clock();
    int  grid_x_size, grid_y_size;
    auto grid_index_map =
        createGridIndexMap(x_positions, y_positions, width, height, pixel_size,
                           grid_x_size, grid_y_size, 0.1f);
    out_width = (size_t)std::ceilf(
                    *std::max_element(x_positions.begin(), x_positions.end())) +
                width;
    out_height = (size_t)std::ceilf(*std::max_element(y_positions.begin(),
                                                      y_positions.end())) +
                 height;
    std::vector<std::vector<float>> weight_maps(
        num_images,
        std::vector<float>(width * height, 1.f)); // initialize to 1.0f
    computeWeightMaps(weight_maps, x_positions, y_positions, width, height,
                      grid_index_map);
    std::vector<float> stitched_image(out_width * out_height, 0.0f);
    std::vector<float> weight_sum(out_width * out_height, 0.0f);
    for (auto& [grid_pos, index] : grid_index_map) {
        const float* img        = &images[index * width * height];
        auto&        weight_map = weight_maps[index];
        float        x_offset   = x_positions[index];
        float        y_offset   = y_positions[index];
        int          x_start    = (int)std::ceilf(x_offset);
        int          y_start    = (int)std::ceilf(y_offset);
        int          x_end      = (int)std::floorf(x_offset + (float)width);
        int          y_end      = (int)std::floorf(y_offset + (float)height);

        for (int x = x_start; x < x_end; x++) {
            for (int y = y_start; y < y_end; y++) {
                float x_local = (float)x - x_offset;
                float y_local = (float)y - y_offset;
                // bilinear interpolation
                int   x0  = (int)std::floorf(x_local);
                int   x1  = (int)std::ceilf(x_local);
                int   y0  = (int)std::floorf(y_local);
                int   y1  = (int)std::ceilf(y_local);
                float dx  = x_local - (float)x0;
                float dy  = y_local - (float)y0;
                float w00 = (1.f - dx) * (1.f - dy);
                float w01 = (1.f - dx) * dy;
                float w10 = dx * (1.f - dy);
                float w11 = dx * dy;
                float pixel_value =
                    img[y0 * width + x0] * w00 + img[y1 * width + x0] * w01 +
                    img[y0 * width + x1] * w10 + img[y1 * width + x1] * w11;
                float weight_value = weight_map[y0 * width + x0] * w00 +
                                     weight_map[y1 * width + x0] * w01 +
                                     weight_map[y0 * width + x1] * w10 +
                                     weight_map[y1 * width + x1] * w11;
                stitched_image[y * out_width + x] += pixel_value * weight_value;
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
    end            = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    // std::cout << "Stitching completed in " << elapsed << " seconds.\n";
    return stitched_image;
}
