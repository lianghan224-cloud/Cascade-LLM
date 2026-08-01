/*
 * Cascade-LLM C ABI for the optional CUTLASS SM86 provider.
 *
 * The shared library intentionally does not include Python or PyTorch headers.
 * Python passes CUDA tensor pointers and the current PyTorch stream through
 * ctypes, which keeps the provider build independent of CPython development
 * packages.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

extern "C" {

enum CascadeDtype : int32_t
{
    CASCADE_BF16 = 0,
    CASCADE_FP16 = 1,
};

int32_t cascade_cutlass_w8a16_run(
    int32_t activation_dtype,
    int32_t scale_dtype,
    void const* activation,
    int8_t const* weight,
    void const* scale,
    void* output,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t group_size,
    cudaStream_t stream);

char const* cascade_cutlass_last_error();

int32_t cascade_cutlass_provider_version();
}
