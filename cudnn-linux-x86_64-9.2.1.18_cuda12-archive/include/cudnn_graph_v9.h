/*
 * Copyright 2014-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/*
 *  cudnn_graph : cuDNN's basic definitions operations.
 */

#if !defined(CUDNN_GRAPH_H_)
#define CUDNN_GRAPH_H_

#include <cuda_runtime_api.h>
#include <library_types.h>

#include <stdint.h>

#include "cudnn_version.h"

/* These version numbers are autogenerated, do not edit manually. */
#define CUDNN_GRAPH_MAJOR 9
#define CUDNN_GRAPH_MINOR 2
#define CUDNN_GRAPH_PATCH 1

#if (CUDNN_GRAPH_MAJOR != CUDNN_MAJOR) || (CUDNN_GRAPH_MINOR != CUDNN_MINOR) || (CUDNN_GRAPH_PATCH != CUDNN_PATCHLEVEL)
#error Version mismatch in cuDNN GRAPH!!!
#endif

#ifndef CUDNNWINAPI
#ifdef _WIN32
#define CUDNNWINAPI __stdcall
#else
#define CUDNNWINAPI
#endif
#endif

/* Warnings for deprecated API-s are enabled using the CUDNN_WARN_DEPRECATED macro */
#if defined(CUDNN_WARN_DEPRECATED) && (defined(__GNUC__) || defined(__clang__))
/* GCC, Intel C/C++, Cray C/C++, CLANG, IBM XL C/C++ little endian */
#define CUDNN_DEPRECATED __attribute__((deprecated))
#define CUDNN_DEPRECATED_ENUM __attribute__((deprecated))
#elif defined(CUDNN_WARN_DEPRECATED) && defined(_MSC_VER)
/* Microsoft Visual C++ */
#define CUDNN_DEPRECATED __declspec(deprecated)
#define CUDNN_DEPRECATED_ENUM __declspec(deprecated)
#elif defined(CUDNN_WARN_DEPRECATED) && (__cplusplus >= 201402L)
/* C++14 compilers */
#define CUDNN_DEPRECATED [[deprecated]]
#define CUDNN_DEPRECATED_ENUM [[deprecated]]
#else
/* No support for the deprecated attribute */
#define CUDNN_DEPRECATED
#define CUDNN_DEPRECATED_ENUM
#endif

#if defined(__cplusplus)
extern "C" {
#endif

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

size_t CUDNNWINAPI
cudnnGetVersion(void);

size_t CUDNNWINAPI
cudnnGetMaxDeviceVersion(void);

/* Returns CUDA Runtime version statically linked against cudnn */
size_t CUDNNWINAPI
cudnnGetCudartVersion(void);

/*
 * CUDNN return codes
 */
typedef enum {
    CUDNN_STATUS_SUCCESS = 0,

    /* Uncategorized errors */
    CUDNN_STATUS_NOT_INITIALIZED                = 1001,
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH    = 1002,
    CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH = 1003,
    CUDNN_STATUS_DEPRECATED                     = 1004,
    CUDNN_STATUS_LICENSE_ERROR                  = 1005,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS            = 1006,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW            = 1007,
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED      = 1008,

    CUDNN_STATUS_BAD_PARAM                    = 2000,
    CUDNN_STATUS_BAD_PARAM_NULL_POINTER       = 2002,
    CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER = 2003,
    CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED      = 2004,
    CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND       = 2005,
    CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT  = 2006,
    CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH    = 2007,
    CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH     = 2008,
    CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES = 2009,
    CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE     = 2010,

    CUDNN_STATUS_NOT_SUPPORTED                              = 3000,
    CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN                = 3001,
    CUDNN_STATUS_NOT_SUPPORTED_SHAPE                        = 3002,
    CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE                    = 3003,
    CUDNN_STATUS_NOT_SUPPORTED_LAYOUT                       = 3004,
    CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER     = 3005,
    CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART          = 3006,
    CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH                = 3007,
    CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING = 3008,
    CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE       = 3009,
    CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT   = 3010,
    CUDNN_STATUS_NOT_SUPPORTED_PADDING                      = 3011,
    CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM             = 3012,

    CUDNN_STATUS_INTERNAL_ERROR                          = 4000,
    CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED       = 4001,
    CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE         = 4002,
    CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED   = 4003,
    CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED = 4004,
    CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM         = 4005,
    CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED  = 4006,

    CUDNN_STATUS_EXECUTION_FAILED             = 5000,
    CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER = 5001,
    CUDNN_STATUS_EXECUTION_FAILED_CUBLAS      = 5002,
    CUDNN_STATUS_EXECUTION_FAILED_CUDART      = 5003,
    CUDNN_STATUS_EXECUTION_FAILED_CURAND      = 5004,

    CUDNN_STATUS_ALLOC_FAILED CUDNN_DEPRECATED_ENUM  = CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED,
    CUDNN_STATUS_INVALID_VALUE CUDNN_DEPRECATED_ENUM = 2001 /* please transition to CUDNN_STATUS_BAD_PARAM instead */,
    CUDNN_STATUS_ARCH_MISMATCH CUDNN_DEPRECATED_ENUM = CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH,
    CUDNN_STATUS_MAPPING_ERROR CUDNN_DEPRECATED_ENUM = CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING CUDNN_DEPRECATED_ENUM =
        CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING,
    CUDNN_STATUS_VERSION_MISMATCH CUDNN_DEPRECATED_ENUM = CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH,
} cudnnStatus_t;

#define CUDNN_STATUS_FULL_ERROR_CODE(category, specific_err) ((cudnnStatus_t)(0 + (category) + (specific_err)))
#define CUDNN_STATUS_CATEGORY(full_error_code) ((full_error_code) / 1000 * 1000)
#define CUDNN_STATUS_SPECIFIC_ERROR(full_error_code) ((full_error_code) % 1000)

/* human-readable error messages */
const char *CUDNNWINAPI
cudnnGetErrorString(cudnnStatus_t status);

void CUDNNWINAPI
cudnnGetLastErrorString(char *message, size_t max_size);

/* Forward definition in this version only */
typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t CUDNN_DEPRECATED;

typedef enum {
    CUDNN_ERRQUERY_RAWCODE     = 0,
    CUDNN_ERRQUERY_NONBLOCKING = 1,
    CUDNN_ERRQUERY_BLOCKING    = 2,
} cudnnErrQueryMode_t;

CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI
cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag);

cudnnStatus_t CUDNNWINAPI
cudnnGetProperty(libraryPropertyType type, int *value);

cudnnStatus_t CUDNNWINAPI
cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t CUDNNWINAPI
cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t CUDNNWINAPI
cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t CUDNNWINAPI
cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);
/*
 * CUDNN data type
 */
typedef enum {
    CUDNN_DATA_FLOAT                         = 0,
    CUDNN_DATA_DOUBLE                        = 1,
    CUDNN_DATA_HALF                          = 2,
    CUDNN_DATA_INT8                          = 3,
    CUDNN_DATA_INT32                         = 4,
    CUDNN_DATA_INT8x4 CUDNN_DEPRECATED_ENUM  = 5,
    CUDNN_DATA_UINT8                         = 6,
    CUDNN_DATA_UINT8x4 CUDNN_DEPRECATED_ENUM = 7,
    CUDNN_DATA_INT8x32 CUDNN_DEPRECATED_ENUM = 8,
    CUDNN_DATA_BFLOAT16                      = 9,
    CUDNN_DATA_INT64                         = 10,
    CUDNN_DATA_BOOLEAN                       = 11,
    CUDNN_DATA_FP8_E4M3                      = 12,
    CUDNN_DATA_FP8_E5M2                      = 13,
    CUDNN_DATA_FAST_FLOAT_FOR_FP8            = 14,
} cudnnDataType_t;

/*
 * CUDNN math type
 */
typedef enum {
    CUDNN_DEFAULT_MATH                    = 0,
    CUDNN_TENSOR_OP_MATH                  = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    CUDNN_FMA_MATH                        = 3,
} cudnnMathType_t;

/*
 * CUDNN propagate Nan
 */
typedef enum {
    CUDNN_NOT_PROPAGATE_NAN CUDNN_DEPRECATED_ENUM = 0,
    CUDNN_PROPAGATE_NAN CUDNN_DEPRECATED_ENUM     = 1,
} cudnnNanPropagation_t;

/*
 * Behavior for OOB samples. OOB samples are samples where L+R > T is encountered during the gradient calculation. If
 * gradMode is set to CUDNN_CTC_SKIP_OOB_GRADIENTS, then the CTC loss function does not write to the gradient buffer for
 * that sample. Instead, the current values, even not finite, are retained. If gradMode is set to
 * CUDNN_CTC_ZERO_OOB_GRADIENTS, then the gradient for that sample is set to zero. This guarantees a finite gradient.
 */
typedef enum {
    CUDNN_CTC_ZERO_OOB_GRADIENTS = 0,
    CUDNN_CTC_SKIP_OOB_GRADIENTS = 1,
} cudnnCTCGradMode_t;

typedef enum {
    CUDNN_TENSOR_NCHW        = 0, /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC        = 1, /* feature maps interleaved ( cStride = 1 )*/
    CUDNN_TENSOR_NCHW_VECT_C = 2, /* each image point is vector of element of C, vector length in data type */
} cudnnTensorFormat_t;

/*
 * CUDNN ReduceTensor op type
 */
typedef enum {
    CUDNN_REDUCE_TENSOR_ADD          = 0,
    CUDNN_REDUCE_TENSOR_MUL          = 1,
    CUDNN_REDUCE_TENSOR_MIN          = 2,
    CUDNN_REDUCE_TENSOR_MAX          = 3,
    CUDNN_REDUCE_TENSOR_AMAX         = 4,
    CUDNN_REDUCE_TENSOR_AVG          = 5,
    CUDNN_REDUCE_TENSOR_NORM1        = 6,
    CUDNN_REDUCE_TENSOR_NORM2        = 7,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} cudnnReduceTensorOp_t;

/*
 * activation mode
 */
typedef enum {
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4,
    CUDNN_ACTIVATION_IDENTITY     = 5,
    CUDNN_ACTIVATION_SWISH        = 6
} cudnnActivationMode_t CUDNN_DEPRECATED;

typedef enum {
    CUDNN_SEV_FATAL   = 0,
    CUDNN_SEV_ERROR   = 1,
    CUDNN_SEV_WARNING = 2,
    CUDNN_SEV_INFO    = 3,
} cudnnSeverity_t;

/* Message masks to be used with cudnnSetCallback() */
#define CUDNN_SEV_ERROR_EN (1U << CUDNN_SEV_ERROR)
#define CUDNN_SEV_WARNING_EN (1U << CUDNN_SEV_WARNING)
#define CUDNN_SEV_INFO_EN (1U << CUDNN_SEV_INFO)

/* struct containing useful informaiton for each API call */
typedef struct cudnnDebugStruct {
    unsigned cudnn_version;
    cudnnStatus_t cudnnStatus;
    unsigned time_sec;      /* epoch time in seconds */
    unsigned time_usec;     /* microseconds part of epoch time */
    unsigned time_delta;    /* time since start in seconds */
    cudnnHandle_t handle;   /* cudnn handle */
    cudaStream_t stream;    /* cuda stream ID */
    unsigned long long pid; /* process ID */
    unsigned long long tid; /* thread ID */
    int cudaDeviceId;       /* CUDA device ID */
    int reserved[15];       /* reserved for future use */
} cudnnDebug_t;

typedef void (*cudnnCallback_t)(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);

cudnnStatus_t CUDNNWINAPI
cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr);

cudnnStatus_t CUDNNWINAPI
cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr);

/*
 * \brief Cross-library version checker.
 * This function is implemented differently in each sub-library. Each sublib
 * checks whether its own version matches that of its dependencies.
 * \returns CUDNN_STATUS_SUCCESS if the version check passes,
 *          CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH if the versions are inconsistent.
 */
cudnnStatus_t CUDNNWINAPI
cudnnGraphVersionCheck(void);

/* Maximum supported number of tensor dimensions */
#define CUDNN_DIM_MAX 8

/*
 *  convolution mode
 */
typedef enum { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;

/*
 * CUDNN Reorder
 */
typedef enum {
    CUDNN_DEFAULT_REORDER = 0,
    CUDNN_NO_REORDER      = 1,
} cudnnReorderType_t CUDNN_DEPRECATED;

typedef void *cudnnBackendDescriptor_t;

typedef struct cudnnFractionStruct {
    int64_t numerator;
    int64_t denominator;
} cudnnFraction_t;

typedef enum {
    CUDNN_POINTWISE_ADD        = 0,
    CUDNN_POINTWISE_ADD_SQUARE = 5,
    CUDNN_POINTWISE_DIV        = 6,
    CUDNN_POINTWISE_MAX        = 3,
    CUDNN_POINTWISE_MIN        = 2,
    CUDNN_POINTWISE_MOD        = 7,
    CUDNN_POINTWISE_MUL        = 1,
    CUDNN_POINTWISE_POW        = 8,
    CUDNN_POINTWISE_SUB        = 9,

    CUDNN_POINTWISE_ABS        = 10,
    CUDNN_POINTWISE_CEIL       = 11,
    CUDNN_POINTWISE_COS        = 12,
    CUDNN_POINTWISE_EXP        = 13,
    CUDNN_POINTWISE_FLOOR      = 14,
    CUDNN_POINTWISE_LOG        = 15,
    CUDNN_POINTWISE_NEG        = 16,
    CUDNN_POINTWISE_RSQRT      = 17,
    CUDNN_POINTWISE_SIN        = 18,
    CUDNN_POINTWISE_SQRT       = 4,
    CUDNN_POINTWISE_TAN        = 19,
    CUDNN_POINTWISE_ERF        = 20,
    CUDNN_POINTWISE_IDENTITY   = 21,
    CUDNN_POINTWISE_RECIPROCAL = 22,
    CUDNN_POINTWISE_ATAN2      = 23,

    CUDNN_POINTWISE_RELU_FWD             = 100,
    CUDNN_POINTWISE_TANH_FWD             = 101,
    CUDNN_POINTWISE_SIGMOID_FWD          = 102,
    CUDNN_POINTWISE_ELU_FWD              = 103,
    CUDNN_POINTWISE_GELU_FWD             = 104,
    CUDNN_POINTWISE_SOFTPLUS_FWD         = 105,
    CUDNN_POINTWISE_SWISH_FWD            = 106,
    CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = 107,

    CUDNN_POINTWISE_RELU_BWD             = 200,
    CUDNN_POINTWISE_TANH_BWD             = 201,
    CUDNN_POINTWISE_SIGMOID_BWD          = 202,
    CUDNN_POINTWISE_ELU_BWD              = 203,
    CUDNN_POINTWISE_GELU_BWD             = 204,
    CUDNN_POINTWISE_SOFTPLUS_BWD         = 205,
    CUDNN_POINTWISE_SWISH_BWD            = 206,
    CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = 207,

    CUDNN_POINTWISE_CMP_EQ  = 300,
    CUDNN_POINTWISE_CMP_NEQ = 301,
    CUDNN_POINTWISE_CMP_GT  = 302,
    CUDNN_POINTWISE_CMP_GE  = 303,
    CUDNN_POINTWISE_CMP_LT  = 304,
    CUDNN_POINTWISE_CMP_LE  = 305,

    CUDNN_POINTWISE_LOGICAL_AND = 400,
    CUDNN_POINTWISE_LOGICAL_OR  = 401,
    CUDNN_POINTWISE_LOGICAL_NOT = 402,

    CUDNN_POINTWISE_GEN_INDEX = 501,

    CUDNN_POINTWISE_BINARY_SELECT = 601,
} cudnnPointwiseMode_t;

typedef enum {
    CUDNN_RESAMPLE_NEAREST                 = 0,
    CUDNN_RESAMPLE_BILINEAR                = 1,
    CUDNN_RESAMPLE_AVGPOOL                 = 2,
    CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING = 2,
    CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING = 4,
    CUDNN_RESAMPLE_MAXPOOL                 = 3,
} cudnnResampleMode_t;

typedef enum {
    CUDNN_SIGNAL_SET  = 0,
    CUDNN_SIGNAL_WAIT = 1,
} cudnnSignalMode_t;

typedef enum {
    CUDNN_GENSTATS_SUM_SQSUM = 0,
} cudnnGenStatsMode_t;

typedef enum {
    CUDNN_BN_FINALIZE_STATISTICS_TRAINING  = 0,
    CUDNN_BN_FINALIZE_STATISTICS_INFERENCE = 1,
} cudnnBnFinalizeStatsMode_t;

typedef enum {
    CUDNN_RNG_DISTRIBUTION_BERNOULLI,
    CUDNN_RNG_DISTRIBUTION_UNIFORM,
    CUDNN_RNG_DISTRIBUTION_NORMAL,
} cudnnRngDistribution_t;

typedef enum {
    CUDNN_ATTR_POINTWISE_MODE                                  = 0,
    CUDNN_ATTR_POINTWISE_MATH_PREC                             = 1,
    CUDNN_ATTR_POINTWISE_NAN_PROPAGATION CUDNN_DEPRECATED_ENUM = 2,
    CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP                       = 3,
    CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP                       = 4,
    CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE                 = 5,
    CUDNN_ATTR_POINTWISE_ELU_ALPHA                             = 6,
    CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA                         = 7,
    CUDNN_ATTR_POINTWISE_SWISH_BETA                            = 8,
    CUDNN_ATTR_POINTWISE_AXIS                                  = 9,

    CUDNN_ATTR_CONVOLUTION_COMP_TYPE      = 100,
    CUDNN_ATTR_CONVOLUTION_CONV_MODE      = 101,
    CUDNN_ATTR_CONVOLUTION_DILATIONS      = 102,
    CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 103,
    CUDNN_ATTR_CONVOLUTION_POST_PADDINGS  = 104,
    CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS   = 105,
    CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS   = 106,

    CUDNN_ATTR_ENGINEHEUR_MODE            = 200,
    CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201,
    CUDNN_ATTR_ENGINEHEUR_RESULTS         = 202,
    CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET = 203,

    CUDNN_ATTR_ENGINECFG_ENGINE             = 300,
    CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO  = 301,
    CUDNN_ATTR_ENGINECFG_KNOB_CHOICES       = 302,
    CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE     = 303,
    CUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED = 304,

    CUDNN_ATTR_EXECUTION_PLAN_HANDLE                     = 400,
    CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG              = 401,
    CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE             = 402,
    CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 403,
    CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 404,
    CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION        = 405,

    CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID            = 500,
    CUDNN_ATTR_INTERMEDIATE_INFO_SIZE                 = 501,
    CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS  = 502,
    CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 503,

    CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE  = 600,
    CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = 601,

    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA        = 700,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA         = 701,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC    = 702,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W            = 703,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X            = 704,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y            = 705,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA       = 706,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA        = 707,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC   = 708,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W           = 709,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX          = 710,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY          = 711,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA     = 712,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA      = 713,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW        = 715,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X         = 716,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY        = 717,

    CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750,
    CUDNN_ATTR_OPERATION_POINTWISE_XDESC         = 751,
    CUDNN_ATTR_OPERATION_POINTWISE_BDESC         = 752,
    CUDNN_ATTR_OPERATION_POINTWISE_YDESC         = 753,
    CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1        = 754,
    CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2        = 755,
    CUDNN_ATTR_OPERATION_POINTWISE_DXDESC        = 756,
    CUDNN_ATTR_OPERATION_POINTWISE_DYDESC        = 757,
    CUDNN_ATTR_OPERATION_POINTWISE_TDESC         = 758,

    CUDNN_ATTR_OPERATION_GENSTATS_MODE      = 770,
    CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771,
    CUDNN_ATTR_OPERATION_GENSTATS_XDESC     = 772,
    CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC   = 773,
    CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774,

    CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE                = 780,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC                 = 781,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC                = 782,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC             = 783,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC                = 784,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC                 = 785,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC    = 786,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC     = 787,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = 788,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC  = 789,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC           = 790,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC        = 791,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC             = 792,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC              = 793,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC          = 794,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC              = 795,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC   = 796,

    CUDNN_ATTR_OPERATIONGRAPH_HANDLE              = 800,
    CUDNN_ATTR_OPERATIONGRAPH_OPS                 = 801,
    CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802,

    CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT       = 900,
    CUDNN_ATTR_TENSOR_DATA_TYPE            = 901,
    CUDNN_ATTR_TENSOR_DIMENSIONS           = 902,
    CUDNN_ATTR_TENSOR_STRIDES              = 903,
    CUDNN_ATTR_TENSOR_VECTOR_COUNT         = 904,
    CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905,
    CUDNN_ATTR_TENSOR_UNIQUE_ID            = 906,
    CUDNN_ATTR_TENSOR_IS_VIRTUAL           = 907,
    CUDNN_ATTR_TENSOR_IS_BY_VALUE          = 908,
    CUDNN_ATTR_TENSOR_REORDERING_MODE      = 909,
    CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC   = 913,

    CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS    = 1000,
    CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001,
    CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 1002,
    CUDNN_ATTR_VARIANT_PACK_WORKSPACE     = 1003,

    CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID = 1100,
    CUDNN_ATTR_LAYOUT_INFO_TYPES      = 1101,

    CUDNN_ATTR_KNOB_INFO_TYPE          = 1200,
    CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201,
    CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202,
    CUDNN_ATTR_KNOB_INFO_STRIDE        = 1203,

    CUDNN_ATTR_ENGINE_OPERATION_GRAPH = 1300,
    CUDNN_ATTR_ENGINE_GLOBAL_INDEX    = 1301,
    CUDNN_ATTR_ENGINE_KNOB_INFO       = 1302,
    CUDNN_ATTR_ENGINE_NUMERICAL_NOTE  = 1303,
    CUDNN_ATTR_ENGINE_LAYOUT_INFO     = 1304,
    CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE   = 1305,
    CUDNN_ATTR_ENGINE_SM_COUNT_TARGET = 1306,

    CUDNN_ATTR_MATMUL_COMP_TYPE     = 1500,
    CUDNN_ATTR_MATMUL_PADDING_VALUE = 1503,

    CUDNN_ATTR_OPERATION_MATMUL_ADESC                                                 = 1520,
    CUDNN_ATTR_OPERATION_MATMUL_BDESC                                                 = 1521,
    CUDNN_ATTR_OPERATION_MATMUL_CDESC                                                 = 1522,
    CUDNN_ATTR_OPERATION_MATMUL_DESC                                                  = 1523,
    CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT CUDNN_DEPRECATED_ENUM = 1524,
    CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC                                  = 1525,
    CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC                                  = 1526,
    CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC                                  = 1527,

    CUDNN_ATTR_REDUCTION_OPERATOR  = 1600,
    CUDNN_ATTR_REDUCTION_COMP_TYPE = 1601,

    CUDNN_ATTR_OPERATION_REDUCTION_XDESC = 1610,
    CUDNN_ATTR_OPERATION_REDUCTION_YDESC = 1611,
    CUDNN_ATTR_OPERATION_REDUCTION_DESC  = 1612,

    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC        = 1620,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC        = 1621,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC      = 1622,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC    = 1623,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC           = 1624,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC          = 1625,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC   = 1626,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC    = 1627,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = 1628,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC  = 1629,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS          = 1630,

    CUDNN_ATTR_RESAMPLE_MODE            = 1700,
    CUDNN_ATTR_RESAMPLE_COMP_TYPE       = 1701,
    CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS    = 1702,
    CUDNN_ATTR_RESAMPLE_POST_PADDINGS   = 1703,
    CUDNN_ATTR_RESAMPLE_PRE_PADDINGS    = 1704,
    CUDNN_ATTR_RESAMPLE_STRIDES         = 1705,
    CUDNN_ATTR_RESAMPLE_WINDOW_DIMS     = 1706,
    CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION = 1707,
    CUDNN_ATTR_RESAMPLE_PADDING_MODE    = 1708,

    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC                       = 1710,
    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC                       = 1711,
    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC                     = 1712,
    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA CUDNN_DEPRECATED_ENUM = 1713,
    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA CUDNN_DEPRECATED_ENUM  = 1714,
    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC                        = 1716,

    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC                      = 1720,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC                      = 1721,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC                     = 1722,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA CUDNN_DEPRECATED_ENUM = 1723,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA CUDNN_DEPRECATED_ENUM  = 1724,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC                        = 1725,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC                       = 1726,
    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC                       = 1727,

    CUDNN_ATTR_OPERATION_CONCAT_AXIS          = 1800,
    CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS   = 1801,
    CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = 1802,
    CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC   = 1803,

    CUDNN_ATTR_OPERATION_SIGNAL_MODE     = 1900,
    CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = 1901,
    CUDNN_ATTR_OPERATION_SIGNAL_VALUE    = 1902,
    CUDNN_ATTR_OPERATION_SIGNAL_XDESC    = 1903,
    CUDNN_ATTR_OPERATION_SIGNAL_YDESC    = 1904,

    CUDNN_ATTR_OPERATION_NORM_FWD_MODE                     = 2000,
    CUDNN_ATTR_OPERATION_NORM_FWD_PHASE                    = 2001,
    CUDNN_ATTR_OPERATION_NORM_FWD_XDESC                    = 2002,
    CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC                = 2003,
    CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC        = 2004,
    CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC               = 2005,
    CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC                = 2006,
    CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC             = 2007,
    CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC      = 2008,
    CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC  = 2009,
    CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC   = 2010,
    CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = 2011,
    CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC  = 2012,
    CUDNN_ATTR_OPERATION_NORM_FWD_YDESC                    = 2013,
    CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS          = 2014,

    CUDNN_ATTR_OPERATION_NORM_BWD_MODE              = 2100,
    CUDNN_ATTR_OPERATION_NORM_BWD_XDESC             = 2101,
    CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC         = 2102,
    CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = 2103,
    CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC            = 2104,
    CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC        = 2105,
    CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC      = 2106,
    CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC       = 2107,
    CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC        = 2108,
    CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC            = 2109,
    CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS   = 2110,

    CUDNN_ATTR_OPERATION_RESHAPE_XDESC = 2200,
    CUDNN_ATTR_OPERATION_RESHAPE_YDESC = 2201,

    CUDNN_ATTR_RNG_DISTRIBUTION                   = 2300,
    CUDNN_ATTR_RNG_NORMAL_DIST_MEAN               = 2301,
    CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = 2302,
    CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM           = 2303,
    CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM           = 2304,
    CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY     = 2305,

    CUDNN_ATTR_OPERATION_RNG_YDESC       = 2310,
    CUDNN_ATTR_OPERATION_RNG_SEED        = 2311,
    CUDNN_ATTR_OPERATION_RNG_DESC        = 2312,
    CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC = 2313,
} cudnnBackendAttributeName_t;

typedef enum {
    CUDNN_TYPE_HANDLE = 0,
    CUDNN_TYPE_DATA_TYPE,
    CUDNN_TYPE_BOOLEAN,
    CUDNN_TYPE_INT64,
    CUDNN_TYPE_FLOAT,
    CUDNN_TYPE_DOUBLE,
    CUDNN_TYPE_VOID_PTR,
    CUDNN_TYPE_CONVOLUTION_MODE,
    CUDNN_TYPE_HEUR_MODE,
    CUDNN_TYPE_KNOB_TYPE,
    CUDNN_TYPE_NAN_PROPOGATION CUDNN_DEPRECATED_ENUM,
    CUDNN_TYPE_NUMERICAL_NOTE,
    CUDNN_TYPE_LAYOUT_TYPE,
    CUDNN_TYPE_ATTRIB_NAME,
    CUDNN_TYPE_POINTWISE_MODE,
    CUDNN_TYPE_BACKEND_DESCRIPTOR,
    CUDNN_TYPE_GENSTATS_MODE,
    CUDNN_TYPE_BN_FINALIZE_STATS_MODE,
    CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
    CUDNN_TYPE_BEHAVIOR_NOTE,
    CUDNN_TYPE_TENSOR_REORDERING_MODE,
    CUDNN_TYPE_RESAMPLE_MODE,
    CUDNN_TYPE_PADDING_MODE,
    CUDNN_TYPE_INT32,
    CUDNN_TYPE_CHAR,
    CUDNN_TYPE_SIGNAL_MODE,
    CUDNN_TYPE_FRACTION,
    CUDNN_TYPE_NORM_MODE,
    CUDNN_TYPE_NORM_FWD_PHASE,
    CUDNN_TYPE_RNG_DISTRIBUTION
} cudnnBackendAttributeType_t;

typedef enum {
    CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0,
    CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
    CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
    CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
    CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,
    CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
    CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
    CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
    CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR,
    CUDNN_BACKEND_MATMUL_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
    CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
    CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
    CUDNN_BACKEND_RNG_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
} cudnnBackendDescriptorType_t;

typedef enum {
    CUDNN_NUMERICAL_NOTE_TENSOR_CORE = 0,
    CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS,
    CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION,
    CUDNN_NUMERICAL_NOTE_FFT,
    CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC,
    CUDNN_NUMERICAL_NOTE_WINOGRAD,
    CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4,
    CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6,
    CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13,
    CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP,
    CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
} cudnnBackendNumericalNote_t;

typedef enum {
    CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION             = 0,
    CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER = 1,
    CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER   = 2,
    CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
} cudnnBackendBehaviorNote_t;

typedef enum {
    CUDNN_KNOB_TYPE_SPLIT_K CUDNN_DEPRECATED_ENUM          = 0,
    CUDNN_KNOB_TYPE_SWIZZLE                                = 1,
    CUDNN_KNOB_TYPE_TILE_SIZE                              = 2,
    CUDNN_KNOB_TYPE_USE_TEX CUDNN_DEPRECATED_ENUM          = 3,
    CUDNN_KNOB_TYPE_EDGE                                   = 4,
    CUDNN_KNOB_TYPE_KBLOCK CUDNN_DEPRECATED_ENUM           = 5,
    CUDNN_KNOB_TYPE_LDGA CUDNN_DEPRECATED_ENUM             = 6,
    CUDNN_KNOB_TYPE_LDGB CUDNN_DEPRECATED_ENUM             = 7,
    CUDNN_KNOB_TYPE_CHUNK_K CUDNN_DEPRECATED_ENUM          = 8,
    CUDNN_KNOB_TYPE_SPLIT_H CUDNN_DEPRECATED_ENUM          = 9,
    CUDNN_KNOB_TYPE_WINO_TILE CUDNN_DEPRECATED_ENUM        = 10,
    CUDNN_KNOB_TYPE_MULTIPLY                               = 11,
    CUDNN_KNOB_TYPE_SPLIT_K_BUF                            = 12,
    CUDNN_KNOB_TYPE_TILEK                                  = 13,
    CUDNN_KNOB_TYPE_STAGES                                 = 14,
    CUDNN_KNOB_TYPE_REDUCTION_MODE                         = 15,
    CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE CUDNN_DEPRECATED_ENUM = 16,
    CUDNN_KNOB_TYPE_SPLIT_K_SLC                            = 17,
    CUDNN_KNOB_TYPE_IDX_MODE CUDNN_DEPRECATED_ENUM         = 18,
    CUDNN_KNOB_TYPE_SLICED CUDNN_DEPRECATED_ENUM           = 19,
    CUDNN_KNOB_TYPE_SPLIT_RS CUDNN_DEPRECATED_ENUM         = 20,
    CUDNN_KNOB_TYPE_SINGLEBUFFER CUDNN_DEPRECATED_ENUM     = 21,
    CUDNN_KNOB_TYPE_LDGC CUDNN_DEPRECATED_ENUM             = 22,
    CUDNN_KNOB_TYPE_SPECFILT                               = 23,
    CUDNN_KNOB_TYPE_KERNEL_CFG                             = 24,
    CUDNN_KNOB_TYPE_WORKSPACE                              = 25,
    CUDNN_KNOB_TYPE_TILE_CGA CUDNN_DEPRECATED_ENUM         = 26,
    CUDNN_KNOB_TYPE_TILE_CGA_M                             = 27,
    CUDNN_KNOB_TYPE_TILE_CGA_N                             = 28,
    CUDNN_KNOB_TYPE_BLOCK_SIZE                             = 29,
    CUDNN_KNOB_TYPE_OCCUPANCY                              = 30,
    CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD                  = 31,
    CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK CUDNN_DEPRECATED_ENUM  = 32,
    CUDNN_KNOB_TYPE_SPLIT_COLS                             = 33,
    CUDNN_KNOB_TYPE_TILE_ROWS                              = 34,
    CUDNN_KNOB_TYPE_TILE_COLS                              = 35,
    CUDNN_KNOB_TYPE_LOAD_SIZE                              = 36,
    CUDNN_KNOB_TYPE_COUNTS,
} cudnnBackendKnobType_t;

typedef enum {
    CUDNN_LAYOUT_TYPE_PREFERRED_NCHW   = 0,
    CUDNN_LAYOUT_TYPE_PREFERRED_NHWC   = 1,
    CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK = 2,
    CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK = 3,
    CUDNN_LAYOUT_TYPE_COUNT            = 4,
} cudnnBackendLayoutType_t;

typedef enum {
    CUDNN_HEUR_MODE_INSTANT  = 0,
    CUDNN_HEUR_MODE_B        = 1,
    CUDNN_HEUR_MODE_FALLBACK = 2,
    CUDNN_HEUR_MODE_A        = 3,
    CUDNN_HEUR_MODES_COUNT   = 4,
} cudnnBackendHeurMode_t;

typedef enum {
    CUDNN_TENSOR_REORDERING_NONE    = 0,
    CUDNN_TENSOR_REORDERING_INT8x32 = 1,
    CUDNN_TENSOR_REORDERING_F16x16  = 2,
} cudnnBackendTensorReordering_t;

typedef enum {
    CUDNN_ZERO_PAD     = 0,
    CUDNN_NEG_INF_PAD  = 1,
    CUDNN_EDGE_VAL_PAD = 2,
} cudnnPaddingMode_t;

typedef enum {
    CUDNN_LAYER_NORM    = 0,
    CUDNN_INSTANCE_NORM = 1,
    CUDNN_BATCH_NORM    = 2,
    CUDNN_GROUP_NORM    = 3,
    CUDNN_RMS_NORM      = 4,
} cudnnBackendNormMode_t;

typedef enum {
    CUDNN_NORM_FWD_INFERENCE = 0,
    CUDNN_NORM_FWD_TRAINING  = 1,
} cudnnBackendNormFwdPhase_t;

cudnnStatus_t CUDNNWINAPI
cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         const void *arrayOfElements);

cudnnStatus_t CUDNNWINAPI
cudnnBackendGetAttribute(cudnnBackendDescriptor_t const descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t *elementCount,
                         void *arrayOfElements);

cudnnStatus_t CUDNNWINAPI
cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack);

#if defined(__cplusplus)
}
#endif

#endif /* CUDNN_GRAPH_H_ */
