"""CUDA C source shared by Generic and architecture-qualified providers."""


PAGED_ATTENTION_CUDA_SOURCE = r'''
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

extern "C" __global__ void append_kv_u16(
    ushort const* key,
    ushort const* value,
    ushort* key_pool,
    ushort* value_pool,
    int const* slot_pages,
    int const* slot_offsets,
    int total_tokens,
    int num_kv_heads,
    int page_size,
    int head_dim)
{
    int token = (int)blockIdx.x;
    int head = (int)blockIdx.y;
    int dim = (int)threadIdx.x;
    if (token >= total_tokens || head >= num_kv_heads || dim >= head_dim) {
        return;
    }
    int page = slot_pages[token];
    int offset = slot_offsets[token];
    long long source = ((long long)token * num_kv_heads + head) * head_dim + dim;
    long long target = (
        (((long long)page * num_kv_heads + head) * page_size + offset)
        * head_dim + dim);
    key_pool[target] = key[source];
    value_pool[target] = value[source];
}

template <typename Codec>
__device__ __forceinline__ void paged_attention_body(
    ushort const* query,
    ushort const* key_pool,
    ushort const* value_pool,
    int const* flat_blocks,
    int const* logical_block_ids,
    int const* page_valid_tokens,
    int const* block_indptr,
    int const* sequence_lengths,
    int const* query_indptr,
    int const* query_positions,
    ushort* output,
    float* logsumexp,
    int batch_size,
    int num_query_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    float softmax_scale,
    int causal,
    int return_lse,
    float* reduction,
    float* scalars)
{
    int batch = (int)blockIdx.x;
    int local_query = (int)blockIdx.y;
    int query_head = (int)blockIdx.z;
    int dim = (int)threadIdx.x;
    if (batch >= batch_size || query_head >= num_query_heads) {
        return;
    }
    int query_start = query_indptr[batch];
    int query_end = query_indptr[batch + 1];
    int query_index = query_start + local_query;
    if (query_index >= query_end) {
        return;
    }
    int groups = num_query_heads / num_kv_heads;
    int kv_head = query_head / groups;
    int sequence_length = sequence_lengths[batch];
    int query_position = query_positions[query_index];
    int block_start = block_indptr[batch];
    int block_end = block_indptr[batch + 1];
    float query_value = 0.0f;
    if (dim < head_dim) {
        long long query_offset = (
            ((long long)query_index * num_query_heads + query_head)
            * head_dim + dim);
        query_value = Codec::load(query[query_offset]);
    }
    // First pass computes a global maximum.  A two-pass reduction avoids the
    // repeated output rescaling of one-pass online softmax, which otherwise
    // compounds BF16 rounding differences across deep Transformer stacks.
    if (dim == 0) {
        scalars[0] = -3.402823466e+38F;
    }
    __syncthreads();
    for (int block_index = block_start; block_index < block_end; ++block_index) {
        int logical_block = logical_block_ids[block_index];
        int page = flat_blocks[block_index];
        for (int page_token = 0; page_token < page_size; ++page_token) {
            int logical_position = logical_block * page_size + page_token;
            bool visible = page_token < page_valid_tokens[block_index]
                && logical_position < sequence_length;
            if (causal) {
                visible = visible && logical_position <= query_position;
            }
            if (!visible) {
                continue;
            }
            float partial = 0.0f;
            long long pool_offset = (
                (((long long)page * num_kv_heads + kv_head) * page_size + page_token)
                * head_dim + dim);
            if (dim < head_dim) {
                partial = query_value * Codec::load(key_pool[pool_offset]);
            }
            reduction[dim] = partial;
            __syncthreads();
            for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
                if (dim < stride) {
                    reduction[dim] += reduction[dim + stride];
                }
                __syncthreads();
            }
            if (dim == 0) {
                float score = reduction[0] * softmax_scale;
                scalars[0] = fmaxf(scalars[0], score);
            }
            __syncthreads();
        }
    }
    float accumulator = 0.0f;
    if (dim == 0) {
        scalars[1] = 0.0f;
    }
    __syncthreads();
    // Second pass uses the fixed global maximum and accumulates FP32 V.
    for (int block_index = block_start; block_index < block_end; ++block_index) {
        int logical_block = logical_block_ids[block_index];
        int page = flat_blocks[block_index];
        for (int page_token = 0; page_token < page_size; ++page_token) {
            int logical_position = logical_block * page_size + page_token;
            bool visible = page_token < page_valid_tokens[block_index]
                && logical_position < sequence_length;
            if (causal) {
                visible = visible && logical_position <= query_position;
            }
            if (!visible) {
                continue;
            }
            float partial = 0.0f;
            long long pool_offset = (
                (((long long)page * num_kv_heads + kv_head) * page_size + page_token)
                * head_dim + dim);
            if (dim < head_dim) {
                partial = query_value * Codec::load(key_pool[pool_offset]);
            }
            reduction[dim] = partial;
            __syncthreads();
            for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
                if (dim < stride) {
                    reduction[dim] += reduction[dim + stride];
                }
                __syncthreads();
            }
            if (dim == 0) {
                scalars[2] = expf(reduction[0] * softmax_scale - scalars[0]);
                scalars[1] += scalars[2];
            }
            __syncthreads();
            if (dim < head_dim) {
                accumulator += Codec::load(value_pool[pool_offset]) * scalars[2];
            }
            __syncthreads();
        }
    }
    if (dim < head_dim) {
        long long output_offset = (
            ((long long)query_index * num_query_heads + query_head)
            * head_dim + dim);
        output[output_offset] = Codec::store(accumulator / scalars[1]);
    }
    if (dim == 0 && return_lse) {
        logsumexp[(long long)query_index * num_query_heads + query_head]
            = logf(scalars[1]) + scalars[0];
    }
}

extern "C" __global__ void paged_attention_bf16(
    ushort const* query,
    ushort const* key_pool,
    ushort const* value_pool,
    int const* flat_blocks,
    int const* logical_block_ids,
    int const* page_valid_tokens,
    int const* block_indptr,
    int const* sequence_lengths,
    int const* query_indptr,
    int const* query_positions,
    ushort* output,
    float* logsumexp,
    int batch_size,
    int num_query_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    float softmax_scale,
    int causal,
    int return_lse)
{
    __shared__ float reduction[256];
    __shared__ float scalars[4];
    paged_attention_body<BF16Codec>(
        query, key_pool, value_pool, flat_blocks, logical_block_ids,
        page_valid_tokens, block_indptr,
        sequence_lengths, query_indptr, query_positions, output, logsumexp,
        batch_size, num_query_heads, num_kv_heads, page_size, head_dim,
        softmax_scale, causal, return_lse, reduction, scalars);
}

extern "C" __global__ void paged_attention_fp16(
    ushort const* query,
    ushort const* key_pool,
    ushort const* value_pool,
    int const* flat_blocks,
    int const* logical_block_ids,
    int const* page_valid_tokens,
    int const* block_indptr,
    int const* sequence_lengths,
    int const* query_indptr,
    int const* query_positions,
    ushort* output,
    float* logsumexp,
    int batch_size,
    int num_query_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    float softmax_scale,
    int causal,
    int return_lse)
{
    __shared__ float reduction[256];
    __shared__ float scalars[4];
    paged_attention_body<FP16Codec>(
        query, key_pool, value_pool, flat_blocks, logical_block_ids,
        page_valid_tokens, block_indptr,
        sequence_lengths, query_indptr, query_positions, output, logsumexp,
        batch_size, num_query_heads, num_kv_heads, page_size, head_dim,
        softmax_scale, causal, return_lse, reduction, scalars);
}
'''
