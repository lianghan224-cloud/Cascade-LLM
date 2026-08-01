/*
 * Copyright (c) 2026 Cascade-LLM contributors.
 *
 * CUTLASS is consumed as a header-only dependency under its BSD-3-Clause
 * license.  This file implements a C ABI and Cascade-specific dispatch.
 */

#include "binding.h"

#include <algorithm>
#include <cstdio>
#include <exception>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

namespace {

thread_local char last_error[1024] = {0};

void set_error(char const* message)
{
    std::snprintf(last_error, sizeof(last_error), "%s", message);
}

void set_cuda_error(char const* operation, cudaError_t error)
{
    std::snprintf(
        last_error,
        sizeof(last_error),
        "%s: %s",
        operation,
        cudaGetErrorString(error));
}

template <typename T>
__device__ float to_float(T value);

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <>
__device__ float to_float<half>(half value)
{
    return __half2float(value);
}

template <typename T>
__device__ T from_float(float value);

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

template <>
__device__ half from_float<half>(float value)
{
    return __float2half_rn(value);
}

template <typename Activation, typename Scale>
__global__ void decode_gemv(
    Activation const* activation,
    int8_t const* weight,
    Scale const* scale,
    Activation* output,
    int n,
    int k,
    int group_size)
{
    int output_column = blockIdx.x;
    float partial = 0.0f;
    int64_t weight_offset = static_cast<int64_t>(output_column) * k;
    for (int index = threadIdx.x; index < k; index += blockDim.x)
    {
        int64_t scale_index = output_column;
        if (group_size > 0)
        {
            int groups = k / group_size;
            scale_index = static_cast<int64_t>(output_column) * groups
                + index / group_size;
        }
        // Match the documented fallback reference: dequantization first
        // rounds into the activation dtype, then GEMM accumulates in FP32.
        Activation dequantized = from_float<Activation>(
            static_cast<float>(weight[weight_offset + index])
            * to_float(scale[scale_index]));
        partial += to_float(activation[index]) * to_float(dequantized);
    }
    __shared__ float reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        output[output_column] = from_float<Activation>(
            reduction[0]);
    }
}

template <typename Activation, typename Scale>
__global__ void apply_per_column_scale(
    Activation* output,
    Scale const* scale,
    int64_t count,
    int n)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < count)
    {
        int column = static_cast<int>(index % n);
        output[index] = from_float<Activation>(
            to_float(output[index]) * to_float(scale[column]));
    }
}

template <typename Activation>
struct CutlassType;

template <>
struct CutlassType<__nv_bfloat16>
{
    using Type = cutlass::bfloat16_t;
};

template <>
struct CutlassType<half>
{
    using Type = cutlass::half_t;
};

template <typename Activation>
cutlass::Status launch_cutlass_gemm(
    Activation const* activation,
    int8_t const* weight,
    Activation* output,
    int m,
    int n,
    int k,
    cudaStream_t stream)
{
    using Element = typename CutlassType<Activation>::Type;
    using OutputOp = cutlass::epilogue::thread::LinearCombination<
        Element,
        128 / cutlass::sizeof_bits<Element>::value,
        float,
        float>;
    using Gemm = cutlass::gemm::device::GemmUniversal<
        Element,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        Element,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<16, 128, 32>,
        cutlass::gemm::GemmShape<16, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        OutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        8,
        8,
        16,
        cutlass::arch::OpMultiplyAddMixedInputUpcast>;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        {1.0f, 0.0f},
        reinterpret_cast<Element const*>(activation),
        weight,
        reinterpret_cast<Element const*>(output),
        reinterpret_cast<Element*>(output),
        int64_t(0),
        int64_t(0),
        int64_t(0),
        int64_t(0),
        k,
        k,
        n,
        n};

    Gemm gemm;
    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
    {
        return status;
    }
    status = gemm.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
    {
        return status;
    }
    return gemm.run(stream);
}

template <typename Activation, typename Scale>
int run_typed(
    void const* activation,
    int8_t const* weight,
    void const* scale,
    void* output,
    int m,
    int n,
    int k,
    int group_size,
    cudaStream_t stream)
{
    auto const* activation_ptr
        = reinterpret_cast<Activation const*>(activation);
    auto const* scale_ptr = reinterpret_cast<Scale const*>(scale);
    auto* output_ptr = reinterpret_cast<Activation*>(output);
    if (m == 1)
    {
        decode_gemv<Activation, Scale><<<n, 256, 0, stream>>>(
            activation_ptr,
            weight,
            scale_ptr,
            output_ptr,
            n,
            k,
            group_size);
    }
    else
    {
        if (group_size > 0)
        {
            set_error(
                "per-group W8A16 is decode-only; M>1 requires an explicit "
                "prefill backend");
            return 8;
        }
        cutlass::Status status = launch_cutlass_gemm(
            activation_ptr, weight, output_ptr, m, n, k, stream);
        if (status != cutlass::Status::kSuccess)
        {
            std::snprintf(
                last_error,
                sizeof(last_error),
                "CUTLASS GEMM failed: %s",
                cutlassGetStatusString(status));
            return 4;
        }
        int64_t count = static_cast<int64_t>(m) * n;
        int blocks = static_cast<int>((count + 255) / 256);
        apply_per_column_scale<Activation, Scale>
            <<<blocks, 256, 0, stream>>>(
                output_ptr, scale_ptr, count, n);
    }
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess)
    {
        set_cuda_error("kernel launch failed", error);
        return 5;
    }
    return 0;
}

template <typename Activation>
int dispatch_scale(
    int scale_dtype,
    void const* activation,
    int8_t const* weight,
    void const* scale,
    void* output,
    int m,
    int n,
    int k,
    int group_size,
    cudaStream_t stream)
{
    if (scale_dtype == CASCADE_BF16)
    {
        return run_typed<Activation, __nv_bfloat16>(
            activation,
            weight,
            scale,
            output,
            m,
            n,
            k,
            group_size,
            stream);
    }
    if (scale_dtype == CASCADE_FP16)
    {
        return run_typed<Activation, half>(
            activation,
            weight,
            scale,
            output,
            m,
            n,
            k,
            group_size,
            stream);
    }
    set_error("unsupported scale dtype");
    return 3;
}

} // namespace

extern "C" int32_t cascade_cutlass_w8a16_run(
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
    cudaStream_t stream)
{
    last_error[0] = '\0';
    if (!activation || !weight || !scale || !output)
    {
        set_error("null tensor pointer");
        return 1;
    }
    if (
        m < 1
        || n < 1
        || k < 1
        || k % 16
        || n % 8
        || (
            group_size != 0
            && group_size != 32
            && group_size != 64
            && group_size != 128))
    {
        set_error(
            "requires M>=1, K aligned to 16, N aligned to 8, and group "
            "size 0/32/64/128");
        return 2;
    }
    if (group_size > 0 && k % group_size)
    {
        set_error("group size must divide K");
        return 2;
    }
    try
    {
        if (activation_dtype == CASCADE_BF16)
        {
            return dispatch_scale<__nv_bfloat16>(
                scale_dtype,
                activation,
                weight,
                scale,
                output,
                m,
                n,
                k,
                group_size,
                stream);
        }
        if (activation_dtype == CASCADE_FP16)
        {
            return dispatch_scale<half>(
                scale_dtype,
                activation,
                weight,
                scale,
                output,
                m,
                n,
                k,
                group_size,
                stream);
        }
        set_error("unsupported activation dtype");
        return 3;
    }
    catch (std::exception const& error)
    {
        set_error(error.what());
        return 6;
    }
    catch (...)
    {
        set_error("unknown provider exception");
        return 7;
    }
}

extern "C" char const* cascade_cutlass_last_error()
{
    return last_error;
}

extern "C" int32_t cascade_cutlass_provider_version()
{
    return 2;
}
