#pragma once
#include <vector>

float getInterpolatedValue(const std::vector<float>& image, int width,
                           int height, float x, float y);

void flipHorizontally(std::vector<float>& image, int width, int height);

float calculateCenterOfRotation(const std::vector<float>& projection_0deg,
                                const std::vector<float>& projection_180deg,
                                int width, int height, float max_shift,
                                float step_size);