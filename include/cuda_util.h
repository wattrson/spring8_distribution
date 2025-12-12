#pragma once

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
    {                                                                          \
        auto status = static_cast<cudaError_t>(call);                          \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr,                                                    \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed " \
                    "with "                                                    \
                    "%s (%d).\n",                                              \
                    #call, __LINE__, __FILE__, cudaGetErrorString(status),     \
                    status);                                                   \
            exit(status);                                                      \
        }                                                                      \
    }
#endif // CUDA_RT_CALL
