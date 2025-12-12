#include <string>

#include "back_projection.h"

int main() {
    const int detectorWidth   = 4440;
    const int detectorHeight  = 1614;
    const int projectionCount = 4501;
    const float detectorPixelSize = 6.5f / 4.f * 0.001f; // 6.5um/pixel
    // 4倍対物レンズ
    // const float detectorPixelSize =
    //     6.5f / 2.f * 0.001f * 2.f; // 6.5um/pixel 2倍対物レンズ ビニング2x
    const float       roaShift    = 44.74f;
    const std::string inputFolder = "D:/AppleWatch_20251205/stitched/";
    const std::string inputFilenamePrefix = "stitched_image_";
    const std::string outputFolder        = "D:/AppleWatch_20251205/subpixel/";

    cbp2(detectorWidth, detectorHeight, projectionCount, detectorPixelSize,
         roaShift, inputFolder, inputFilenamePrefix, outputFolder);
    return 0;
}