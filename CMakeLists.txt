cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(spring8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 86 120)

find_package(hdf5 REQUIRED COMPONENTS CXX shared CONFIG)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(ct STATIC
    src/back_projection.cu
    src/file_io.cpp
    src/calculate_cor.cpp
    src/stitch.cpp
    src/pre_process.cu
    src/h5_util.cpp
    )
target_include_directories(ct PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    )
target_link_libraries(ct PUBLIC CUDA::toolkit hdf5::hdf5_cpp-shared OpenMP::OpenMP_CXX)
target_compile_options(ct PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>)

add_executable(_stitch2cbp main/_stitch2cbp.cu)
target_link_libraries(_stitch2cbp PRIVATE ct CUDA::toolkit hdf5::hdf5_cpp-shared OpenMP::OpenMP_CXX)
target_include_directories(_stitch2cbp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    )
target_compile_options(_stitch2cbp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>)

add_executable(0_check_roi main/0_check_roi.cu)
target_link_libraries(0_check_roi PRIVATE ct CUDA::toolkit hdf5::hdf5_cpp-shared OpenMP::OpenMP_CXX)
target_include_directories(0_check_roi PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    )
target_compile_options(0_check_roi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>)

add_executable(1_stitch main/1_stitch.cu)
target_link_libraries(1_stitch PRIVATE ct CUDA::toolkit hdf5::hdf5_cpp-shared OpenMP::OpenMP_CXX)
target_include_directories(1_stitch PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    )
target_compile_options(1_stitch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>)

add_executable(2_cor main/2_cor.cpp)
target_link_libraries(2_cor PRIVATE ct CUDA::toolkit OpenMP::OpenMP_CXX)
target_include_directories(2_cor PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    )

add_executable(3_cbp main/3_cbp.cu)
target_link_libraries(3_cbp PRIVATE ct CUDA::toolkit)
target_include_directories(3_cbp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    )
target_compile_options(3_cbp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>)