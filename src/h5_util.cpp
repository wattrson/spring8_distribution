#include "h5_util.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <H5Cpp.h>

using namespace H5;
namespace fs = std::filesystem;

std::vector<H5std_string>
getH5FileList(const std::string              &dir_path,
              const std::vector<std::string> &exclude_keywords) {
    std::vector<H5std_string> file_list;
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        std::cerr << "Directory does not exist: " << dir_path << std::endl;
        return file_list;
    }

    for (const auto &entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string filename  = entry.path().filename().string();
            std::string extension = entry.path().extension().string();

            if (extension == ".h5") {
                // if (filename.find(exclude_keyword) == std::string::npos) {
                //     file_list.push_back(entry.path().string());
                // }
                bool exclude = false;
                for (const auto &keyword : exclude_keywords) {
                    if (filename.find(keyword) != std::string::npos) {
                        exclude = true;
                        break;
                    }
                }
                if (!exclude) {
                    file_list.push_back(entry.path().string());
                }
            }
        }
    }
    return file_list;
}
