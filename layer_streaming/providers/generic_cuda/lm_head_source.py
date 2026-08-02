"""CUDA source for the deterministic, chunk-independent LM Head."""


LM_HEAD_CUDA_SOURCE = r'''
typedef unsigned short ushort;

struct BF16Codec {
    __device__ __forceinline__ static float load(ushort value) {
        return __uint_as_float(((unsigned int)value) << 16);
    }
    __device__ __forceinline__ static ushort store(float value) {
        unsigned int bits = __float_as_uint(value);
        unsigned int lsb = (bits >> 16) & 1u;
        return (ushort)((bits + 0x7fffu + lsb) >> 16);
    }
};

struct FP16Codec {
    __device__ __forceinline__ static float load(ushort value) {
        float result;
        asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
        return result;
    }
    __device__ __forceinline__ static ushort store(float value) {
        ushort result;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
        return result;
    }
};

template <typename Codec>
__device__ __forceinline__ void lm_head_body(
    ushort const* hidden,
    ushort const* weight,
    ushort* output,
    int tokens,
    int rows,
    int hidden_size,
    float* reduction)
{
    int row = (int)blockIdx.x;
    int token = (int)blockIdx.y;
    int lane = (int)threadIdx.x;
    if (row >= rows || token >= tokens) {
        return;
    }
    float partial = 0.0f;
    long long hidden_base = (long long)token * hidden_size;
    long long weight_base = (long long)row * hidden_size;
    for (int column = lane; column < hidden_size; column += (int)blockDim.x) {
        partial += Codec::load(hidden[hidden_base + column])
            * Codec::load(weight[weight_base + column]);
    }
    reduction[lane] = partial;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[(long long)token * rows + row] = Codec::store(reduction[0]);
    }
}

extern "C" __global__ void deterministic_lm_head_bf16(
    ushort const* hidden,
    ushort const* weight,
    ushort* output,
    int tokens,
    int rows,
    int hidden_size)
{
    __shared__ float reduction[256];
    lm_head_body<BF16Codec>(
        hidden, weight, output, tokens, rows, hidden_size, reduction);
}

extern "C" __global__ void deterministic_lm_head_fp16(
    ushort const* hidden,
    ushort const* weight,
    ushort* output,
    int tokens,
    int rows,
    int hidden_size)
{
    __shared__ float reduction[256];
    lm_head_body<FP16Codec>(
        hidden, weight, output, tokens, rows, hidden_size, reduction);
}
'''
