#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

/*
 * CUDA Driver API H2D microbenchmark.
 *
 * This file intentionally does not include CUDA headers and does not link
 * against libcudart or libcuda.  The CUDA Driver API is loaded at runtime from
 * libcuda.so.1 with dlopen(3).
 */

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

enum {
    CUDA_SUCCESS = 0,
    CU_STREAM_NON_BLOCKING = 1,
    CU_EVENT_DEFAULT = 0,
    CU_MEMHOSTALLOC_DEFAULT = 0,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
};

typedef struct {
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDriverGetVersion)(int *);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemHostAlloc)(void **, size_t, unsigned int);
    CUresult (*cuMemFreeHost)(void *);
    CUresult (*cuStreamCreate)(CUstream *, unsigned int);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuEventCreate)(CUevent *, unsigned int);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventSynchronize)(CUevent);
    CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*cuGetErrorName)(CUresult, const char **);
    CUresult (*cuGetErrorString)(CUresult, const char **);
} CudaApi;

typedef struct {
    double p10;
    double median;
    double p90;
} Stats;

typedef struct {
    const char *memory_name;
    const char *api_name;
    const void *src;
    int asynchronous;
} Mode;

static CudaApi g_cuda;

static void die_message(const char *message)
{
    fprintf(stderr, "fatal: %s\n", message);
    exit(1);
}

static void die_errno(const char *message)
{
    fprintf(stderr, "fatal: %s: %s\n", message, strerror(errno));
    exit(1);
}

static const char *cuda_error_name(CUresult result)
{
    const char *name = NULL;
    if (g_cuda.cuGetErrorName != NULL &&
        g_cuda.cuGetErrorName(result, &name) == CUDA_SUCCESS && name != NULL) {
        return name;
    }
    return "CUDA_ERROR_UNKNOWN";
}

static const char *cuda_error_string(CUresult result)
{
    const char *message = NULL;
    if (g_cuda.cuGetErrorString != NULL &&
        g_cuda.cuGetErrorString(result, &message) == CUDA_SUCCESS &&
        message != NULL) {
        return message;
    }
    return "unknown CUDA error";
}

static void cuda_check(CUresult result, const char *operation)
{
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "fatal: %s failed: %s (%d): %s\n", operation,
                cuda_error_name(result), result, cuda_error_string(result));
        exit(1);
    }
}

static void load_symbol(void *library, const char *name, void *destination,
                        size_t destination_size, int required)
{
    void *symbol;
    const char *error;

    dlerror();
    symbol = dlsym(library, name);
    error = dlerror();
    if (error != NULL || symbol == NULL) {
        if (required) {
            fprintf(stderr, "fatal: dlsym(%s) failed: %s\n", name,
                    error != NULL ? error : "symbol not found");
            exit(1);
        }
        memset(destination, 0, destination_size);
        return;
    }
    if (destination_size != sizeof(symbol)) {
        die_message("unexpected function-pointer size on this platform");
    }
    memcpy(destination, &symbol, sizeof(symbol));
}

static void load_symbol_fallback(void *library, const char *preferred,
                                 const char *fallback, void *destination,
                                 size_t destination_size)
{
    void *symbol;

    dlerror();
    symbol = dlsym(library, preferred);
    if (dlerror() != NULL || symbol == NULL) {
        dlerror();
        symbol = dlsym(library, fallback);
        if (dlerror() != NULL || symbol == NULL) {
            fprintf(stderr, "fatal: neither %s nor %s is exported by libcuda\n",
                    preferred, fallback);
            exit(1);
        }
    }
    if (destination_size != sizeof(symbol)) {
        die_message("unexpected function-pointer size on this platform");
    }
    memcpy(destination, &symbol, sizeof(symbol));
}

#define LOAD_REQUIRED(library, member, symbol)                                \
    load_symbol((library), (symbol), &g_cuda.member, sizeof(g_cuda.member), 1)
#define LOAD_OPTIONAL(library, member, symbol)                                \
    load_symbol((library), (symbol), &g_cuda.member, sizeof(g_cuda.member), 0)
#define LOAD_FALLBACK(library, member, preferred, fallback)                   \
    load_symbol_fallback((library), (preferred), (fallback), &g_cuda.member,  \
                         sizeof(g_cuda.member))

static void *load_cuda_driver(void)
{
    void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        fprintf(stderr, "fatal: dlopen(libcuda.so.1) failed: %s\n", dlerror());
        exit(1);
    }

    LOAD_REQUIRED(library, cuInit, "cuInit");
    LOAD_REQUIRED(library, cuDriverGetVersion, "cuDriverGetVersion");
    LOAD_REQUIRED(library, cuDeviceGetCount, "cuDeviceGetCount");
    LOAD_REQUIRED(library, cuDeviceGet, "cuDeviceGet");
    LOAD_REQUIRED(library, cuDeviceGetName, "cuDeviceGetName");
    LOAD_REQUIRED(library, cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD_FALLBACK(library, cuCtxCreate, "cuCtxCreate_v2", "cuCtxCreate");
    LOAD_FALLBACK(library, cuCtxDestroy, "cuCtxDestroy_v2", "cuCtxDestroy");
    LOAD_FALLBACK(library, cuMemAlloc, "cuMemAlloc_v2", "cuMemAlloc");
    LOAD_FALLBACK(library, cuMemFree, "cuMemFree_v2", "cuMemFree");
    LOAD_REQUIRED(library, cuMemHostAlloc, "cuMemHostAlloc");
    LOAD_REQUIRED(library, cuMemFreeHost, "cuMemFreeHost");
    LOAD_REQUIRED(library, cuStreamCreate, "cuStreamCreate");
    LOAD_FALLBACK(library, cuStreamDestroy, "cuStreamDestroy_v2",
                  "cuStreamDestroy");
    LOAD_REQUIRED(library, cuStreamSynchronize, "cuStreamSynchronize");
    LOAD_REQUIRED(library, cuEventCreate, "cuEventCreate");
    LOAD_FALLBACK(library, cuEventDestroy, "cuEventDestroy_v2",
                  "cuEventDestroy");
    LOAD_REQUIRED(library, cuEventRecord, "cuEventRecord");
    LOAD_REQUIRED(library, cuEventSynchronize, "cuEventSynchronize");
    LOAD_REQUIRED(library, cuEventElapsedTime, "cuEventElapsedTime");
    LOAD_FALLBACK(library, cuMemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    LOAD_FALLBACK(library, cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2",
                  "cuMemcpyHtoDAsync");
    LOAD_OPTIONAL(library, cuGetErrorName, "cuGetErrorName");
    LOAD_OPTIONAL(library, cuGetErrorString, "cuGetErrorString");
    return library;
}

static double monotonic_us(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        die_errno("clock_gettime(CLOCK_MONOTONIC_RAW)");
    }
    return (double)now.tv_sec * 1000000.0 + (double)now.tv_nsec / 1000.0;
}

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static Stats calculate_stats(const double *values, int count)
{
    double *copy;
    Stats stats;
    int p10_index;
    int p50_index;
    int p90_index;

    if (count <= 0) {
        die_message("calculate_stats called with no samples");
    }
    copy = (double *)malloc((size_t)count * sizeof(*copy));
    if (copy == NULL) {
        die_errno("malloc statistics scratch space");
    }
    memcpy(copy, values, (size_t)count * sizeof(*copy));
    qsort(copy, (size_t)count, sizeof(*copy), compare_double);
    p10_index = (int)llround(0.10 * (double)(count - 1));
    p50_index = (int)llround(0.50 * (double)(count - 1));
    p90_index = (int)llround(0.90 * (double)(count - 1));
    stats.p10 = copy[p10_index];
    stats.median = copy[p50_index];
    stats.p90 = copy[p90_index];
    free(copy);
    return stats;
}

static void print_json_string(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    putchar('"');
    while (*cursor != '\0') {
        switch (*cursor) {
        case '"':
            fputs("\\\"", stdout);
            break;
        case '\\':
            fputs("\\\\", stdout);
            break;
        case '\b':
            fputs("\\b", stdout);
            break;
        case '\f':
            fputs("\\f", stdout);
            break;
        case '\n':
            fputs("\\n", stdout);
            break;
        case '\r':
            fputs("\\r", stdout);
            break;
        case '\t':
            fputs("\\t", stdout);
            break;
        default:
            if (*cursor < 0x20) {
                printf("\\u%04x", (unsigned int)*cursor);
            } else {
                putchar((int)*cursor);
            }
            break;
        }
        ++cursor;
    }
    putchar('"');
}

static void print_stats(const Stats *stats)
{
    printf("{\"p10\":%.6f,\"median\":%.6f,\"p90\":%.6f}",
           stats->p10, stats->median, stats->p90);
}

static int iterations_for_size(size_t size)
{
    if (size <= 64U * 1024U) {
        return 200;
    }
    if (size <= 1024U * 1024U) {
        return 100;
    }
    if (size <= 16U * 1024U * 1024U) {
        return 50;
    }
    if (size <= 64U * 1024U * 1024U) {
        return 25;
    }
    if (size <= 116U * 1024U * 1024U) {
        return 15;
    }
    return 10;
}

static int warmups_for_size(size_t size)
{
    if (size <= 1024U * 1024U) {
        return 10;
    }
    if (size <= 64U * 1024U * 1024U) {
        return 5;
    }
    return 3;
}

static CUresult measure_one(const Mode *mode, CUdeviceptr destination,
                            size_t size, CUstream async_stream,
                            CUevent event_start, CUevent event_stop,
                            double *host_call_us, double *gpu_elapsed_us)
{
    CUresult result;
    CUstream event_stream = mode->asynchronous ? async_stream : NULL;
    double host_start;
    double host_stop;
    float elapsed_ms = 0.0f;

    result = g_cuda.cuEventRecord(event_start, event_stream);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    host_start = monotonic_us();
    if (mode->asynchronous) {
        result = g_cuda.cuMemcpyHtoDAsync(destination, mode->src, size,
                                         async_stream);
    } else {
        result = g_cuda.cuMemcpyHtoD(destination, mode->src, size);
    }
    host_stop = monotonic_us();
    if (result != CUDA_SUCCESS) {
        if (mode->asynchronous) {
            (void)g_cuda.cuStreamSynchronize(async_stream);
        }
        return result;
    }
    result = g_cuda.cuEventRecord(event_stop, event_stream);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = g_cuda.cuEventSynchronize(event_stop);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = g_cuda.cuEventElapsedTime(&elapsed_ms, event_start, event_stop);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    *host_call_us = host_stop - host_start;
    *gpu_elapsed_us = (double)elapsed_ms * 1000.0;
    return CUDA_SUCCESS;
}

static void print_transfer_result(const Mode *mode, CUdeviceptr destination,
                                  size_t size, CUstream async_stream,
                                  CUevent event_start, CUevent event_stop)
{
    const int iterations = iterations_for_size(size);
    const int warmups = warmups_for_size(size);
    double *host_samples;
    double *gpu_samples;
    double *bandwidth_samples;
    Stats host_stats;
    Stats gpu_stats;
    Stats bandwidth_stats;
    CUresult result = CUDA_SUCCESS;
    int index;

    host_samples = (double *)malloc((size_t)iterations * sizeof(*host_samples));
    gpu_samples = (double *)malloc((size_t)iterations * sizeof(*gpu_samples));
    bandwidth_samples =
        (double *)malloc((size_t)iterations * sizeof(*bandwidth_samples));
    if (host_samples == NULL || gpu_samples == NULL ||
        bandwidth_samples == NULL) {
        die_errno("malloc benchmark samples");
    }

    for (index = 0; index < warmups; ++index) {
        double ignored_host;
        double ignored_gpu;
        result = measure_one(mode, destination, size, async_stream, event_start,
                             event_stop, &ignored_host, &ignored_gpu);
        if (result != CUDA_SUCCESS) {
            break;
        }
    }
    if (result == CUDA_SUCCESS) {
        for (index = 0; index < iterations; ++index) {
            result = measure_one(mode, destination, size, async_stream,
                                 event_start, event_stop, &host_samples[index],
                                 &gpu_samples[index]);
            if (result != CUDA_SUCCESS) {
                break;
            }
            bandwidth_samples[index] =
                gpu_samples[index] > 0.0
                    ? (double)size / (gpu_samples[index] * 1000.0)
                    : 0.0;
        }
    }

    printf("{\"memory\":");
    print_json_string(mode->memory_name);
    printf(",\"api\":");
    print_json_string(mode->api_name);
    printf(",\"size_bytes\":%zu,\"iterations\":%d,\"warmups\":%d,",
           size, iterations, warmups);
    if (result != CUDA_SUCCESS) {
        printf("\"status\":\"unsupported_or_failed\",\"cuda_error_code\":%d,"
               "\"cuda_error_name\":",
               result);
        print_json_string(cuda_error_name(result));
        printf(",\"cuda_error_string\":");
        print_json_string(cuda_error_string(result));
        putchar('}');
        free(host_samples);
        free(gpu_samples);
        free(bandwidth_samples);
        return;
    }

    host_stats = calculate_stats(host_samples, iterations);
    gpu_stats = calculate_stats(gpu_samples, iterations);
    printf("\"status\":\"ok\",\"host_call_us\":");
    print_stats(&host_stats);
    printf(",\"gpu_event_us\":");
    print_stats(&gpu_stats);
    printf(",\"gpu_effective_gbps\":");
    if (size == 0) {
        fputs("null", stdout);
    } else {
        bandwidth_stats = calculate_stats(bandwidth_samples, iterations);
        print_stats(&bandwidth_stats);
    }
    putchar('}');

    free(host_samples);
    free(gpu_samples);
    free(bandwidth_samples);
}

static void linear_regression(const double *x, const double *y, int count,
                              double *intercept, double *slope, double *r2)
{
    double x_mean = 0.0;
    double y_mean = 0.0;
    double covariance = 0.0;
    double variance_x = 0.0;
    double total_y = 0.0;
    double residual_y = 0.0;
    int index;

    for (index = 0; index < count; ++index) {
        x_mean += x[index];
        y_mean += y[index];
    }
    x_mean /= (double)count;
    y_mean /= (double)count;
    for (index = 0; index < count; ++index) {
        const double dx = x[index] - x_mean;
        const double dy = y[index] - y_mean;
        covariance += dx * dy;
        variance_x += dx * dx;
    }
    *slope = covariance / variance_x;
    *intercept = y_mean - *slope * x_mean;
    for (index = 0; index < count; ++index) {
        const double predicted = *intercept + *slope * x[index];
        const double dy = y[index] - y_mean;
        const double residual = y[index] - predicted;
        total_y += dy * dy;
        residual_y += residual * residual;
    }
    *r2 = total_y > 0.0 ? 1.0 - residual_y / total_y : 1.0;
}

static CUresult measure_batch(const void *source, CUdeviceptr destination,
                              size_t size, int copies, CUstream stream,
                              CUevent event_start, CUevent event_stop,
                              double *host_total_us, double *gpu_total_us)
{
    CUresult result;
    double host_start;
    double host_stop;
    float elapsed_ms;
    int copy_index;

    result = g_cuda.cuEventRecord(event_start, stream);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    host_start = monotonic_us();
    for (copy_index = 0; copy_index < copies; ++copy_index) {
        result = g_cuda.cuMemcpyHtoDAsync(destination, source, size, stream);
        if (result != CUDA_SUCCESS) {
            (void)g_cuda.cuStreamSynchronize(stream);
            return result;
        }
    }
    host_stop = monotonic_us();
    result = g_cuda.cuEventRecord(event_stop, stream);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = g_cuda.cuEventSynchronize(event_stop);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = g_cuda.cuEventElapsedTime(&elapsed_ms, event_start, event_stop);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    *host_total_us = host_stop - host_start;
    *gpu_total_us = (double)elapsed_ms * 1000.0;
    return CUDA_SUCCESS;
}

static void print_launch_overhead_for_size(const void *pinned_source,
                                           CUdeviceptr destination, size_t size,
                                           CUstream stream,
                                           CUevent event_start,
                                           CUevent event_stop)
{
    static const int batch_sizes[] = {1,   2,   4,   8,   16,  32,
                                      64,  128, 256, 512, 1024};
    enum { BATCH_COUNT = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0])) };
    double x[BATCH_COUNT];
    double host_medians[BATCH_COUNT];
    double gpu_medians[BATCH_COUNT];
    double host_intercept;
    double host_slope;
    double host_r2;
    double gpu_intercept;
    double gpu_slope;
    double gpu_r2;
    int batch_index;
    CUresult result = CUDA_SUCCESS;

    printf("{\"size_bytes\":%zu,\"batch_points\":[", size);
    for (batch_index = 0; batch_index < BATCH_COUNT; ++batch_index) {
        const int copies = batch_sizes[batch_index];
        const int repetitions = copies <= 64 ? 50 : 30;
        double *host_samples =
            (double *)malloc((size_t)repetitions * sizeof(*host_samples));
        double *gpu_samples =
            (double *)malloc((size_t)repetitions * sizeof(*gpu_samples));
        Stats host_stats;
        Stats gpu_stats;
        int repetition;

        if (host_samples == NULL || gpu_samples == NULL) {
            die_errno("malloc batch samples");
        }
        for (repetition = 0; repetition < 5; ++repetition) {
            double ignored_host;
            double ignored_gpu;
            result = measure_batch(pinned_source, destination, size, copies,
                                   stream, event_start, event_stop,
                                   &ignored_host, &ignored_gpu);
            if (result != CUDA_SUCCESS) {
                break;
            }
        }
        if (result == CUDA_SUCCESS) {
            for (repetition = 0; repetition < repetitions; ++repetition) {
                result = measure_batch(
                    pinned_source, destination, size, copies, stream,
                    event_start, event_stop, &host_samples[repetition],
                    &gpu_samples[repetition]);
                if (result != CUDA_SUCCESS) {
                    break;
                }
            }
        }
        if (batch_index != 0) {
            putchar(',');
        }
        printf("{\"copies\":%d,\"repetitions\":%d,", copies, repetitions);
        if (result != CUDA_SUCCESS) {
            printf("\"status\":\"failed\",\"cuda_error_code\":%d,"
                   "\"cuda_error_name\":",
                   result);
            print_json_string(cuda_error_name(result));
            putchar('}');
            free(host_samples);
            free(gpu_samples);
            break;
        }
        host_stats = calculate_stats(host_samples, repetitions);
        gpu_stats = calculate_stats(gpu_samples, repetitions);
        x[batch_index] = (double)copies;
        host_medians[batch_index] = host_stats.median;
        gpu_medians[batch_index] = gpu_stats.median;
        printf("\"status\":\"ok\",\"host_batch_us\":");
        print_stats(&host_stats);
        printf(",\"gpu_event_batch_us\":");
        print_stats(&gpu_stats);
        putchar('}');
        free(host_samples);
        free(gpu_samples);
    }
    printf("],");
    if (result != CUDA_SUCCESS) {
        printf("\"status\":\"failed\"}");
        return;
    }
    linear_regression(x, host_medians, BATCH_COUNT, &host_intercept,
                      &host_slope, &host_r2);
    linear_regression(x, gpu_medians, BATCH_COUNT, &gpu_intercept, &gpu_slope,
                      &gpu_r2);
    printf("\"status\":\"ok\","
           "\"host_median_linear_fit_us\":{\"intercept\":%.9f,"
           "\"per_copy_slope\":%.9f,\"r2\":%.9f},"
           "\"gpu_event_median_linear_fit_us\":{\"intercept\":%.9f,"
           "\"per_copy_slope\":%.9f,\"r2\":%.9f}}",
           host_intercept, host_slope, host_r2, gpu_intercept, gpu_slope,
           gpu_r2);
}

typedef struct {
    int device_ordinal;
    const char *output_path;
    int large_shards;
    int llama31_8b_shards;
} Options;

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [--device ORDINAL] [--output PATH] [--large-shards] "
            "[--llama31-8b-shards]\n"
            "\n"
            "Writes one JSON result document to stdout (or PATH) and progress "
            "to stderr.  --large-shards benchmarks pinned asynchronous H2D for "
            "1/2/4/8/16 exact Llama-3.2-1B decoder-layer slabs.  "
            "--llama31-8b-shards does the same for Llama-3.1-8B.\n",
            program);
}

static Options parse_options(int argc, char **argv)
{
    Options options = {0, NULL, 0, 0};
    int index;

    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--device") == 0) {
            char *end = NULL;
            long value;
            if (index + 1 >= argc) {
                usage(argv[0]);
                exit(2);
            }
            errno = 0;
            value = strtol(argv[++index], &end, 10);
            if (errno != 0 || end == argv[index] || *end != '\0' ||
                value < 0 || value > 1000000) {
                fprintf(stderr, "invalid device ordinal: %s\n", argv[index]);
                exit(2);
            }
            options.device_ordinal = (int)value;
        } else if (strcmp(argv[index], "--output") == 0) {
            if (index + 1 >= argc || argv[index + 1][0] == '\0') {
                usage(argv[0]);
                exit(2);
            }
            options.output_path = argv[++index];
        } else if (strcmp(argv[index], "--large-shards") == 0) {
            options.large_shards = 1;
        } else if (strcmp(argv[index], "--llama31-8b-shards") == 0) {
            options.llama31_8b_shards = 1;
        } else if (strcmp(argv[index], "--help") == 0 ||
                   strcmp(argv[index], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[index]);
            usage(argv[0]);
            exit(2);
        }
    }
    if (options.large_shards && options.llama31_8b_shards) {
        fprintf(stderr,
                "--large-shards and --llama31-8b-shards are mutually "
                "exclusive\n");
        exit(2);
    }
    return options;
}

int main(int argc, char **argv)
{
    static const size_t default_sizes[] = {
        0,
        1,
        4U * 1024U,
        64U * 1024U,
        1U * 1024U * 1024U,
        2U * 1024U * 1024U,
        4U * 1024U * 1024U,
        8U * 1024U * 1024U,
        16U * 1024U * 1024U,
        20U * 1024U * 1024U,
        32U * 1024U * 1024U,
        64U * 1024U * 1024U,
        96U * 1024U * 1024U,
        116U * 1024U * 1024U,
        121643008U, /* 116 MiB matrices + two 4 KiB BF16 norms. */
        256U * 1024U * 1024U
    };
    static const size_t large_shard_sizes[] = {
        121643008ULL,
        2ULL * 121643008ULL,
        4ULL * 121643008ULL,
        8ULL * 121643008ULL,
        16ULL * 121643008ULL
    };
    static const size_t llama31_8b_shard_sizes[] = {
        436224000ULL,
        2ULL * 436224000ULL,
        4ULL * 436224000ULL,
        8ULL * 436224000ULL,
        16ULL * 436224000ULL
    };
    static const size_t launch_sizes[] = {0, 1, 4U * 1024U};
    enum {
        DEFAULT_SIZE_COUNT =
            (int)(sizeof(default_sizes) / sizeof(default_sizes[0])),
        LARGE_SHARD_SIZE_COUNT =
            (int)(sizeof(large_shard_sizes) / sizeof(large_shard_sizes[0])),
        LLAMA31_8B_SHARD_SIZE_COUNT =
            (int)(sizeof(llama31_8b_shard_sizes) /
                  sizeof(llama31_8b_shard_sizes[0])),
        LAUNCH_SIZE_COUNT =
            (int)(sizeof(launch_sizes) / sizeof(launch_sizes[0]))
    };
    Options options = parse_options(argc, argv);
    const int pinned_only_profile =
        options.large_shards || options.llama31_8b_shards;
    const size_t *sizes = options.llama31_8b_shards
                              ? llama31_8b_shard_sizes
                              : (options.large_shards ? large_shard_sizes
                                                     : default_sizes);
    const int size_count =
        options.llama31_8b_shards
            ? LLAMA31_8B_SHARD_SIZE_COUNT
            : (options.large_shards ? LARGE_SHARD_SIZE_COUNT
                                    : DEFAULT_SIZE_COUNT);
    const size_t maximum_size = sizes[size_count - 1];
    int device_ordinal = options.device_ordinal;
    void *cuda_library;
    int driver_version;
    int device_count;
    CUdevice device;
    char device_name[256] = {0};
    int async_engine_count = -1;
    int concurrent_kernels = -1;
    int pci_domain = -1;
    int pci_bus = -1;
    int pci_device = -1;
    CUcontext context = NULL;
    CUstream stream = NULL;
    CUevent event_start = NULL;
    CUevent event_stop = NULL;
    CUdeviceptr device_buffer = 0;
    void *pinned_source = NULL;
    void *pageable_source = NULL;
    Mode modes[4];
    int mode_count;
    int mode_index;
    int size_index;
    int launch_size_index;
    struct timespec realtime;
    struct tm utc;
    char timestamp[64];

    if (options.output_path != NULL &&
        freopen(options.output_path, "w", stdout) == NULL) {
        die_errno("open output file");
    }

    cuda_library = load_cuda_driver();
    cuda_check(g_cuda.cuInit(0), "cuInit");
    cuda_check(g_cuda.cuDriverGetVersion(&driver_version),
               "cuDriverGetVersion");
    cuda_check(g_cuda.cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_ordinal < 0 || device_ordinal >= device_count) {
        fprintf(stderr, "device ordinal %d is out of range; found %d devices\n",
                device_ordinal, device_count);
        return 2;
    }
    cuda_check(g_cuda.cuDeviceGet(&device, device_ordinal), "cuDeviceGet");
    cuda_check(g_cuda.cuDeviceGetName(device_name, (int)sizeof(device_name),
                                     device),
               "cuDeviceGetName");
    (void)g_cuda.cuDeviceGetAttribute(
        &async_engine_count, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device);
    (void)g_cuda.cuDeviceGetAttribute(
        &concurrent_kernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device);
    (void)g_cuda.cuDeviceGetAttribute(
        &pci_domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);
    (void)g_cuda.cuDeviceGetAttribute(&pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                     device);
    (void)g_cuda.cuDeviceGetAttribute(
        &pci_device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);

    fprintf(stderr,
            "device %d/%d: %s; allocating %.3f MiB device and pinned buffers\n",
            device_ordinal, device_count, device_name,
            (double)maximum_size / (1024.0 * 1024.0));
    cuda_check(g_cuda.cuCtxCreate(&context, 0, device), "cuCtxCreate");
    cuda_check(g_cuda.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate");
    cuda_check(g_cuda.cuEventCreate(&event_start, CU_EVENT_DEFAULT),
               "cuEventCreate(start)");
    cuda_check(g_cuda.cuEventCreate(&event_stop, CU_EVENT_DEFAULT),
               "cuEventCreate(stop)");
    cuda_check(g_cuda.cuMemAlloc(&device_buffer, maximum_size),
               "cuMemAlloc(maximum transfer)");
    cuda_check(g_cuda.cuMemHostAlloc(&pinned_source, maximum_size,
                                    CU_MEMHOSTALLOC_DEFAULT),
               "cuMemHostAlloc(maximum transfer)");
    if (!pinned_only_profile) {
        const int allocation_error =
            posix_memalign(&pageable_source, 4096, maximum_size);
        if (allocation_error != 0) {
            fprintf(stderr, "fatal: posix_memalign pageable source: %s\n",
                    strerror(allocation_error));
            exit(1);
        }
    }
    memset(pinned_source, 0xa5, maximum_size);
    if (pinned_only_profile) {
        modes[0] =
            (Mode){"pinned", "cuMemcpyHtoDAsync_v2", pinned_source, 1};
        mode_count = 1;
    } else {
        memset(pageable_source, 0x5a, maximum_size);
        modes[0] =
            (Mode){"pageable", "cuMemcpyHtoD_v2", pageable_source, 0};
        modes[1] =
            (Mode){"pageable", "cuMemcpyHtoDAsync_v2", pageable_source, 1};
        modes[2] =
            (Mode){"pinned", "cuMemcpyHtoD_v2", pinned_source, 0};
        modes[3] =
            (Mode){"pinned", "cuMemcpyHtoDAsync_v2", pinned_source, 1};
        mode_count = 4;
    }

    if (clock_gettime(CLOCK_REALTIME, &realtime) != 0 ||
        gmtime_r(&realtime.tv_sec, &utc) == NULL ||
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc) ==
            0) {
        strcpy(timestamp, "unknown");
    }

    printf("{\"schema_version\":1,\"benchmark\":");
    print_json_string(
        options.llama31_8b_shards
            ? "cuda_driver_h2d_llama31_8b_shards"
            : (options.large_shards ? "cuda_driver_h2d_large_shards"
                                    : "cuda_driver_h2d"));
    printf(",\"profile\":");
    print_json_string(
        options.llama31_8b_shards
            ? "llama31_8b_exact_layer_multiples_pinned_async"
            : (options.large_shards
                   ? "exact_layer_multiples_pinned_async"
                   : "default_all_memory_and_api_modes"));
    printf(",\"timestamp_utc\":");
    print_json_string(timestamp);
    printf(",\"implementation\":{\"cuda_headers\":false,"
           "\"cuda_toolkit_link\":false,\"cuda_runtime_link\":false,"
           "\"driver_loading\":\"dlopen(libcuda.so.1)\","
           "\"clock\":\"CLOCK_MONOTONIC_RAW\","
           "\"percentile_method\":\"nearest-rank index round(p*(n-1))\","
           "\"bandwidth_units\":\"decimal GB/s\"},"
           "\"driver\":{\"api_version\":%d},"
           "\"device\":{\"ordinal\":%d,\"visible_device_count\":%d,\"name\":",
           driver_version, device_ordinal, device_count);
    print_json_string(device_name);
    printf(",\"pci_bdf\":\"%04x:%02x:%02x.0\","
           "\"async_engine_count\":%d,\"concurrent_kernels\":%s},"
           "\"transfer_results\":[",
           pci_domain, pci_bus, pci_device, async_engine_count,
           concurrent_kernels == 1 ? "true" : "false");

    for (mode_index = 0; mode_index < mode_count; ++mode_index) {
        fprintf(stderr, "  mode: %s / %s\n", modes[mode_index].memory_name,
                modes[mode_index].api_name);
        for (size_index = 0; size_index < size_count; ++size_index) {
            if (mode_index != 0 || size_index != 0) {
                putchar(',');
            }
            print_transfer_result(&modes[mode_index], device_buffer,
                                  sizes[size_index], stream, event_start,
                                  event_stop);
        }
    }
    printf("],\"async_launch_overhead\":{\"memory\":\"pinned\","
           "\"api\":\"cuMemcpyHtoDAsync_v2\","
           "\"fit_input\":\"median batch total versus copy count\","
           "\"sizes\":[");
    fprintf(stderr, "  batched async launch-overhead fit\n");
    for (launch_size_index = 0; launch_size_index < LAUNCH_SIZE_COUNT;
         ++launch_size_index) {
        if (launch_size_index != 0) {
            putchar(',');
        }
        print_launch_overhead_for_size(
            pinned_source, device_buffer, launch_sizes[launch_size_index],
            stream, event_start, event_stop);
    }
    printf("]}}\n");

    cuda_check(g_cuda.cuStreamSynchronize(stream), "cuStreamSynchronize");
    cuda_check(g_cuda.cuMemFreeHost(pinned_source), "cuMemFreeHost");
    free(pageable_source);
    cuda_check(g_cuda.cuMemFree(device_buffer), "cuMemFree");
    cuda_check(g_cuda.cuEventDestroy(event_stop), "cuEventDestroy(stop)");
    cuda_check(g_cuda.cuEventDestroy(event_start), "cuEventDestroy(start)");
    cuda_check(g_cuda.cuStreamDestroy(stream), "cuStreamDestroy");
    cuda_check(g_cuda.cuCtxDestroy(context), "cuCtxDestroy");
    if (dlclose(cuda_library) != 0) {
        fprintf(stderr, "warning: dlclose(libcuda.so.1) failed: %s\n",
                dlerror());
    }
    return 0;
}
