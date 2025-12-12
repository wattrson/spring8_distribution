#include <omp.h>

#include <iostream>
#include <vector>
#include <fstream>

float getInterpolatedValue(const std::vector<float>& image, int width,
                           int height, float x, float y) {
    if (x < 0.f || x > width - 1.f || y < 0.f || y > height - 1.f) {
        // std::cout << "Coordinates out of bounds: (" << x << ", " << y <<
        // ")\n";
        return 0.f; // Out of bounds
    }
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = static_cast<int>(std::ceilf(x));
    int y1 = static_cast<int>(std::ceilf(y));

    float dx = x - x0;
    float dy = y - y0;

    float value = (1 - dx) * (1 - dy) * image[y0 * width + x0] +
                  dx * (1 - dy) * image[y0 * width + x1] +
                  (1 - dx) * dy * image[y1 * width + x0] +
                  dx * dy * image[y1 * width + x1];
    return value;
}

void flipHorizontally(std::vector<float>& image, int width, int height) {

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width / 2; x++) {
            std::swap(image[y * width + x], image[y * width + (width - 1 - x)]);
        }
    }
}

float calculateCenterOfRotation(const std::vector<float>& projection_0deg,
                                const std::vector<float>& projection_180deg,
                                int width, int height, float max_shift,
                                float step_size) {
    std::vector<float> flipped_180deg = projection_180deg;
    flipHorizontally(flipped_180deg, width, height);

    float  best_shift     = 0.f;
    double min_difference = std::numeric_limits<double>::max();

    std::ofstream log_file("../../cor_search_log.csv");
    log_file << "Shift,Difference\n";

    int ceil_max_shift = static_cast<int>(std::ceilf(max_shift));
    for (float shift = -max_shift; shift <= max_shift; shift += step_size) {
        double difference = 0.0;
#pragma omp parallel for reduction(+ : difference) collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = ceil_max_shift; x < width - ceil_max_shift - 1; x++) {
                float shifted_x  = static_cast<float>(x) + shift;
                float value_0deg = projection_0deg[y * width + x];
                float value_180deg =
                    getInterpolatedValue(flipped_180deg, width, height,
                                         shifted_x, static_cast<float>(y));
                difference += std::abs(value_0deg - value_180deg);
            }
        }
        log_file << shift << "," << difference << "\n";
        if (difference < min_difference) {
            min_difference = difference;
            best_shift     = shift;
        }
    }
    std::cout << "Best shift: " << best_shift
              << " pixels with difference per pixel: "
              << min_difference / ((width - ceil_max_shift * 2) * height)
              << "\n";
    log_file.close();
    return -best_shift / 2.f;
}