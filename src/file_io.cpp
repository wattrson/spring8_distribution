#include "file_io.h"

#include <sstream>
#include <string>

std::string getPath(std::string foldername, std::string filename,
                    std::string extension) {
    std::ostringstream ss;
    ss << foldername << filename << extension;
    return ss.str();
}