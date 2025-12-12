#include <string>

#include "back_projection.h"

//! ---パラメータ設定---
const std::string inputFolder         = "D:/AppleWatch_20251205/stitched/";
const std::string inputFilenamePrefix = "stitched_image_";
const std::string outputFolder        = "D:/AppleWatch_20251205/subpixel/";

const int   detectorWidth     = 4440;
const int   detectorHeight    = 1614;
const int   projectionCount   = 4501;
const float detectorPixelSize = 6.5f / 4.f * 0.001f; // 6.5um/pixel
const float roaShift          = 44.74f;
//! ---パラメータ設定ここまで---

int main() {
    cbp2(detectorWidth, detectorHeight, projectionCount, detectorPixelSize,
         roaShift, inputFolder, inputFilenamePrefix, outputFolder);
    return 0;
}