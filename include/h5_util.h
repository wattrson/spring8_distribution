#pragma once
#include <string>
#include <vector>

std::vector<std::string>
getH5FileList(const std::string              &dir_path,
              const std::vector<std::string> &exclude_keywords);