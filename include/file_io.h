#pragma once
#include <string>

// 出力ファイルパスを取得
std::string getPath(std::string foldername, std::string filename,
                    std::string extension = ".raw");