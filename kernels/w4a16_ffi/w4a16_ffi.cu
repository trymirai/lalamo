#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

constexpr int kGroupSize = 32;
constexpr int kReduceThreads = 64;
constexpr int kRowsPerCta = 2;
constexpr int kMicroPacked = 8;
constexpr int kK2048 = 2048;
constexpr int kK2048Packed = kK2048 / 2;
constexpr int kK2048Groups = kK2048 / kGroupSize;
constexpr int kBatchTile = 4;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ uint8_t unpack_nibble(uint8_t byte, int index) {
  return (byte >> ((index & 1) * 4)) & 0x0f;
}

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ uint64_t load_u64(const uint8_t* ptr) {
  return *reinterpret_cast<const uint64_t*>(ptr);
}

template <typename T>
__device__ __forceinline__ float2 load_pair_f32(const T* ptr);

template <>
__device__ __forceinline__ float2 load_pair_f32<__half>(const __half* ptr) {
  return __half22float2(*reinterpret_cast<const __half2*>(ptr));
}

template <>
__device__ __forceinline__ float2 load_pair_f32<__nv_bfloat16>(const __nv_bfloat16* ptr) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(ptr));
}

template <typename T>
__global__ void mlx_k2048_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const T* __restrict__ biases,
    T* __restrict__ output,
    int batch,
    int rows);

template <typename T>
__global__ void awq_k2048_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const uint8_t* __restrict__ packed_zero_points,
    T* __restrict__ output,
    int batch,
    int rows);

template <typename T>
__global__ void mlx_k2048_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const T* __restrict__ biases,
    T* __restrict__ output,
    int batch,
    int rows) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * kRowsPerCta + ty;
  const int batch_index = blockIdx.y;
  if (row >= rows) {
    return;
  }

  const T* x_row = x + batch_index * kK2048;
  const uint8_t* w_row = packed_weights + row * kK2048Packed;
  const T* scale_row = scales + row * kK2048Groups;
  const T* bias_row = biases + row * kK2048Groups;
  float acc = 0.0f;

#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const int chunk = block * kReduceThreads + tx;
    const int group = chunk >> 1;
    const uint64_t bytes = load_u64(w_row + chunk * kMicroPacked);
    const T* x_chunk = x_row + chunk * kMicroPacked * 2;
    float int_dot = 0.0f;
    float x_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < kMicroPacked; ++j) {
      const uint8_t byte = static_cast<uint8_t>((bytes >> (j * 8)) & 0xff);
      const float2 x_pair = load_pair_f32(x_chunk + j * 2);
      int_dot += static_cast<float>(byte & 0x0f) * x_pair.x;
      int_dot += static_cast<float>((byte >> 4) & 0x0f) * x_pair.y;
      x_sum += x_pair.x + x_pair.y;
    }

    const float scale = to_float(scale_row[group]);
    const float bias = to_float(bias_row[group]);
    acc += int_dot * scale + x_sum * bias;
  }

  const int lane = tx & 31;
  const int warp = tx >> 5;
  acc = warp_sum(acc);

  __shared__ float partials[kRowsPerCta][2];
  if (lane == 0) {
    partials[ty][warp] = acc;
  }
  __syncthreads();

  if (warp == 0) {
    acc = lane < 2 ? partials[ty][lane] : 0.0f;
    acc = warp_sum(acc);
    if (lane == 0) {
      output[batch_index * rows + row] = from_float<T>(acc);
    }
  }
}

template <typename T>
__global__ void mlx_k2048_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const T* __restrict__ biases,
    T* __restrict__ output,
    int batch,
    int rows) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * kRowsPerCta + ty;
  const int batch_base = blockIdx.y * kBatchTile;
  if (row >= rows) {
    return;
  }

  const uint8_t* w_row = packed_weights + row * kK2048Packed;
  const T* scale_row = scales + row * kK2048Groups;
  const T* bias_row = biases + row * kK2048Groups;
  float acc[kBatchTile] = {};

#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const int chunk = block * kReduceThreads + tx;
    const int group = chunk >> 1;
    const uint64_t bytes = load_u64(w_row + chunk * kMicroPacked);
    const float scale = to_float(scale_row[group]);
    const float bias = to_float(bias_row[group]);
    float int_dot[kBatchTile] = {};
    float x_sum[kBatchTile] = {};

#pragma unroll
    for (int j = 0; j < kMicroPacked; ++j) {
      const uint8_t byte = static_cast<uint8_t>((bytes >> (j * 8)) & 0xff);
      const float lo = static_cast<float>(byte & 0x0f);
      const float hi = static_cast<float>((byte >> 4) & 0x0f);

#pragma unroll
      for (int b = 0; b < kBatchTile; ++b) {
        const int batch_index = batch_base + b;
        if (batch_index < batch) {
          const T* x_chunk = x + batch_index * kK2048 + chunk * kMicroPacked * 2;
          const float2 x_pair = load_pair_f32(x_chunk + j * 2);
          int_dot[b] += lo * x_pair.x + hi * x_pair.y;
          x_sum[b] += x_pair.x + x_pair.y;
        }
      }
    }

#pragma unroll
    for (int b = 0; b < kBatchTile; ++b) {
      acc[b] += int_dot[b] * scale + x_sum[b] * bias;
    }
  }

  const int lane = tx & 31;
  const int warp = tx >> 5;
  __shared__ float partials[kRowsPerCta][kBatchTile][2];

#pragma unroll
  for (int b = 0; b < kBatchTile; ++b) {
    acc[b] = warp_sum(acc[b]);
    if (lane == 0) {
      partials[ty][b][warp] = acc[b];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int b = 0; b < kBatchTile; ++b) {
      float value = lane < 2 ? partials[ty][b][lane] : 0.0f;
      value = warp_sum(value);
      if (lane == 0 && batch_base + b < batch) {
        output[(batch_base + b) * rows + row] = from_float<T>(value);
      }
    }
  }
}

template <typename T>
__global__ void awq_k2048_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const uint8_t* __restrict__ packed_zero_points,
    T* __restrict__ output,
    int batch,
    int rows) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * kRowsPerCta + ty;
  const int batch_index = blockIdx.y;
  if (row >= rows) {
    return;
  }

  const T* x_row = x + batch_index * kK2048;
  const uint8_t* w_row = packed_weights + row * kK2048Packed;
  const T* scale_row = scales + row * kK2048Groups;
  const uint8_t* zero_row = packed_zero_points + row * (kK2048Groups / 2);
  float acc = 0.0f;

#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const int chunk = block * kReduceThreads + tx;
    const int group = chunk >> 1;
    const uint64_t bytes = load_u64(w_row + chunk * kMicroPacked);
    const T* x_chunk = x_row + chunk * kMicroPacked * 2;
    float int_dot = 0.0f;
    float x_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < kMicroPacked; ++j) {
      const uint8_t byte = static_cast<uint8_t>((bytes >> (j * 8)) & 0xff);
      const float2 x_pair = load_pair_f32(x_chunk + j * 2);
      int_dot += static_cast<float>(byte & 0x0f) * x_pair.x;
      int_dot += static_cast<float>((byte >> 4) & 0x0f) * x_pair.y;
      x_sum += x_pair.x + x_pair.y;
    }

    const uint8_t zp_byte = zero_row[group / 2];
    const float zero_point = static_cast<float>(unpack_nibble(zp_byte, group));
    acc += (int_dot - zero_point * x_sum) * to_float(scale_row[group]);
  }

  const int lane = tx & 31;
  const int warp = tx >> 5;
  acc = warp_sum(acc);

  __shared__ float partials[kRowsPerCta][2];
  if (lane == 0) {
    partials[ty][warp] = acc;
  }
  __syncthreads();

  if (warp == 0) {
    acc = lane < 2 ? partials[ty][lane] : 0.0f;
    acc = warp_sum(acc);
    if (lane == 0) {
      output[batch_index * rows + row] = from_float<T>(acc);
    }
  }
}

template <typename T>
__global__ void awq_k2048_batched_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const uint8_t* __restrict__ packed_zero_points,
    T* __restrict__ output,
    int batch,
    int rows) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * kRowsPerCta + ty;
  const int batch_base = blockIdx.y * kBatchTile;
  if (row >= rows) {
    return;
  }

  const uint8_t* w_row = packed_weights + row * kK2048Packed;
  const T* scale_row = scales + row * kK2048Groups;
  const uint8_t* zero_row = packed_zero_points + row * (kK2048Groups / 2);
  float acc[kBatchTile] = {};

#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const int chunk = block * kReduceThreads + tx;
    const int group = chunk >> 1;
    const uint64_t bytes = load_u64(w_row + chunk * kMicroPacked);
    const uint8_t zp_byte = zero_row[group / 2];
    const float zero_point = static_cast<float>(unpack_nibble(zp_byte, group));
    const float scale = to_float(scale_row[group]);
    float int_dot[kBatchTile] = {};
    float x_sum[kBatchTile] = {};

#pragma unroll
    for (int j = 0; j < kMicroPacked; ++j) {
      const uint8_t byte = static_cast<uint8_t>((bytes >> (j * 8)) & 0xff);
      const float lo = static_cast<float>(byte & 0x0f);
      const float hi = static_cast<float>((byte >> 4) & 0x0f);

#pragma unroll
      for (int b = 0; b < kBatchTile; ++b) {
        const int batch_index = batch_base + b;
        if (batch_index < batch) {
          const T* x_chunk = x + batch_index * kK2048 + chunk * kMicroPacked * 2;
          const float2 x_pair = load_pair_f32(x_chunk + j * 2);
          int_dot[b] += lo * x_pair.x + hi * x_pair.y;
          x_sum[b] += x_pair.x + x_pair.y;
        }
      }
    }

#pragma unroll
    for (int b = 0; b < kBatchTile; ++b) {
      acc[b] += (int_dot[b] - zero_point * x_sum[b]) * scale;
    }
  }

  const int lane = tx & 31;
  const int warp = tx >> 5;
  __shared__ float partials[kRowsPerCta][kBatchTile][2];

#pragma unroll
  for (int b = 0; b < kBatchTile; ++b) {
    acc[b] = warp_sum(acc[b]);
    if (lane == 0) {
      partials[ty][b][warp] = acc[b];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int b = 0; b < kBatchTile; ++b) {
      float value = lane < 2 ? partials[ty][b][lane] : 0.0f;
      value = warp_sum(value);
      if (lane == 0 && batch_base + b < batch) {
        output[(batch_base + b) * rows + row] = from_float<T>(value);
      }
    }
  }
}

template <typename T>
__global__ void mlx_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const T* __restrict__ biases,
    T* __restrict__ output,
    int batch,
    int rows,
    int channels) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * kRowsPerCta + ty;
  const int batch_index = blockIdx.y;
  if (row >= rows) {
    return;
  }

  const int groups = channels / kGroupSize;
  const int packed_channels = channels / 2;
  const T* x_row = x + batch_index * channels;
  float acc = 0.0f;

  for (int packed_base = tx * kMicroPacked; packed_base < packed_channels; packed_base += kReduceThreads * kMicroPacked) {
    const int group = (packed_base * 2) / kGroupSize;
    const float scale = to_float(scales[row * groups + group]);
    const float bias = to_float(biases[row * groups + group]);
    float int_dot = 0.0f;
    float x_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < kMicroPacked; ++j) {
      const int packed_col = packed_base + j;
      const uint8_t byte = packed_weights[row * packed_channels + packed_col];
      const float x0 = to_float(x_row[packed_col * 2]);
      const float x1 = to_float(x_row[packed_col * 2 + 1]);
      int_dot += static_cast<float>(byte & 0x0f) * x0;
      int_dot += static_cast<float>((byte >> 4) & 0x0f) * x1;
      x_sum += x0 + x1;
    }

    acc += int_dot * scale + x_sum * bias;
  }

  const int lane = tx & 31;
  const int warp = tx >> 5;
  acc = warp_sum(acc);

  __shared__ float partials[kRowsPerCta][2];
  if (lane == 0) {
    partials[ty][warp] = acc;
  }
  __syncthreads();

  if (warp == 0) {
    acc = lane < 2 ? partials[ty][lane] : 0.0f;
    acc = warp_sum(acc);
    if (lane == 0) {
      output[batch_index * rows + row] = from_float<T>(acc);
    }
  }
}

template <typename T>
__global__ void awq_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const T* __restrict__ scales,
    const uint8_t* __restrict__ packed_zero_points,
    T* __restrict__ output,
    int batch,
    int rows,
    int channels) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * kRowsPerCta + ty;
  const int batch_index = blockIdx.y;
  if (row >= rows) {
    return;
  }

  const int groups = channels / kGroupSize;
  const int packed_channels = channels / 2;
  const int packed_groups = groups / 2;
  const T* x_row = x + batch_index * channels;
  float acc = 0.0f;

  for (int packed_base = tx * kMicroPacked; packed_base < packed_channels; packed_base += kReduceThreads * kMicroPacked) {
    const int group = (packed_base * 2) / kGroupSize;
    const float scale = to_float(scales[row * groups + group]);
    const uint8_t zp_byte = packed_zero_points[row * packed_groups + group / 2];
    const float zero_point = static_cast<float>(unpack_nibble(zp_byte, group));
    float int_dot = 0.0f;
    float x_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < kMicroPacked; ++j) {
      const int packed_col = packed_base + j;
      const uint8_t byte = packed_weights[row * packed_channels + packed_col];
      const float x0 = to_float(x_row[packed_col * 2]);
      const float x1 = to_float(x_row[packed_col * 2 + 1]);
      int_dot += static_cast<float>(byte & 0x0f) * x0;
      int_dot += static_cast<float>((byte >> 4) & 0x0f) * x1;
      x_sum += x0 + x1;
    }

    acc += (int_dot - zero_point * x_sum) * scale;
  }

  const int lane = tx & 31;
  const int warp = tx >> 5;
  acc = warp_sum(acc);

  __shared__ float partials[kRowsPerCta][2];
  if (lane == 0) {
    partials[ty][warp] = acc;
  }
  __syncthreads();

  if (warp == 0) {
    acc = lane < 2 ? partials[ty][lane] : 0.0f;
    acc = warp_sum(acc);
    if (lane == 0) {
      output[batch_index * rows + row] = from_float<T>(acc);
    }
  }
}

template <typename ScaleBuffer>
ffi::Error check_common(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ScaleBuffer scales,
    ffi::AnyBuffer output,
    ffi::DataType dtype) {
  if (vector.element_type() != dtype || output.element_type() != dtype) {
    return ffi::Error::InvalidArgument("vector and output dtypes must match the FFI target dtype");
  }
  const auto vector_dims = vector.dimensions();
  const auto weight_dims = packed_weights.dimensions();
  const auto scale_dims = scales.dimensions();
  const auto output_dims = output.dimensions();
  const int vector_rank = vector_dims.size();
  if (vector_rank != 1 && vector_rank != 2) {
    return ffi::Error::InvalidArgument("vector must have rank 1 or 2");
  }
  if (scale_dims[0] != weight_dims[0]) {
    return ffi::Error::InvalidArgument("scales rows must match packed weight rows");
  }
  if (scale_dims[1] * kGroupSize != weight_dims[1] * 2) {
    return ffi::Error::InvalidArgument("scales groups must match group size 32");
  }
  if (vector_dims[vector_rank - 1] != weight_dims[1] * 2) {
    return ffi::Error::InvalidArgument("vector channels must match packed weight channels");
  }
  if (output_dims.size() != vector_rank || output_dims[output_dims.size() - 1] != weight_dims[0]) {
    return ffi::Error::InvalidArgument("output shape must be vector batch shape plus weight rows");
  }
  return ffi::Error::Success();
}

template <typename T, typename ScaleBuffer, typename BiasBuffer>
ffi::Error launch_mlx(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ScaleBuffer scales,
    BiasBuffer biases,
    ffi::Result<ffi::AnyBuffer> output,
    cudaStream_t stream,
    ffi::DataType dtype) {
  if (auto error = check_common(vector, packed_weights, scales, *output, dtype); error.failure()) {
    return error;
  }
  if (biases.dimensions()[0] != scales.dimensions()[0] || biases.dimensions()[1] != scales.dimensions()[1]) {
    return ffi::Error::InvalidArgument("biases shape must match scales shape");
  }

  const auto vector_dims = vector.dimensions();
  const int batch = vector_dims.size() == 1 ? 1 : vector_dims[0];
  const int rows = packed_weights.dimensions()[0];
  const int channels = packed_weights.dimensions()[1] * 2;
  const dim3 grid((rows + kRowsPerCta - 1) / kRowsPerCta, batch, 1);
  const dim3 block(kReduceThreads, kRowsPerCta, 1);
  if (channels == kK2048) {
    if (batch > 1) {
      mlx_k2048_batched_kernel<T><<<dim3(grid.x, (batch + kBatchTile - 1) / kBatchTile, 1), block, 0, stream>>>(
          reinterpret_cast<const T*>(vector.untyped_data()),
          packed_weights.typed_data(),
          reinterpret_cast<const T*>(scales.typed_data()),
          reinterpret_cast<const T*>(biases.typed_data()),
          reinterpret_cast<T*>((*output).untyped_data()),
          batch,
          rows);
      return ffi::Error::Success();
    }

    mlx_k2048_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(vector.untyped_data()),
        packed_weights.typed_data(),
        reinterpret_cast<const T*>(scales.typed_data()),
        reinterpret_cast<const T*>(biases.typed_data()),
        reinterpret_cast<T*>((*output).untyped_data()),
        batch,
        rows);
    return ffi::Error::Success();
  }

  mlx_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(vector.untyped_data()),
      packed_weights.typed_data(),
      reinterpret_cast<const T*>(scales.typed_data()),
      reinterpret_cast<const T*>(biases.typed_data()),
      reinterpret_cast<T*>((*output).untyped_data()),
      batch,
      rows,
      channels);
  return ffi::Error::Success();
}

template <typename T, typename ScaleBuffer>
ffi::Error launch_awq(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ScaleBuffer scales,
    ffi::BufferR2<ffi::U8> packed_zero_points,
    ffi::Result<ffi::AnyBuffer> output,
    cudaStream_t stream,
    ffi::DataType dtype) {
  if (auto error = check_common(vector, packed_weights, scales, *output, dtype); error.failure()) {
    return error;
  }
  const int groups = scales.dimensions()[1];
  if (packed_zero_points.dimensions()[0] != scales.dimensions()[0] || packed_zero_points.dimensions()[1] != groups / 2) {
    return ffi::Error::InvalidArgument("packed zero-point shape must be rows by packed groups");
  }

  const auto vector_dims = vector.dimensions();
  const int batch = vector_dims.size() == 1 ? 1 : vector_dims[0];
  const int rows = packed_weights.dimensions()[0];
  const int channels = packed_weights.dimensions()[1] * 2;
  const dim3 grid((rows + kRowsPerCta - 1) / kRowsPerCta, batch, 1);
  const dim3 block(kReduceThreads, kRowsPerCta, 1);
  if (channels == kK2048) {
    if (batch > 1) {
      awq_k2048_batched_kernel<T><<<dim3(grid.x, (batch + kBatchTile - 1) / kBatchTile, 1), block, 0, stream>>>(
          reinterpret_cast<const T*>(vector.untyped_data()),
          packed_weights.typed_data(),
          reinterpret_cast<const T*>(scales.typed_data()),
          packed_zero_points.typed_data(),
          reinterpret_cast<T*>((*output).untyped_data()),
          batch,
          rows);
      return ffi::Error::Success();
    }

    awq_k2048_kernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(vector.untyped_data()),
        packed_weights.typed_data(),
        reinterpret_cast<const T*>(scales.typed_data()),
        packed_zero_points.typed_data(),
        reinterpret_cast<T*>((*output).untyped_data()),
        batch,
        rows);
    return ffi::Error::Success();
  }

  awq_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(vector.untyped_data()),
      packed_weights.typed_data(),
      reinterpret_cast<const T*>(scales.typed_data()),
      packed_zero_points.typed_data(),
      reinterpret_cast<T*>((*output).untyped_data()),
      batch,
      rows,
      channels);
  return ffi::Error::Success();
}

ffi::Error mlx_f16(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ffi::BufferR2<ffi::F16> scales,
    ffi::BufferR2<ffi::F16> biases,
    ffi::Result<ffi::AnyBuffer> output,
    cudaStream_t stream) {
  return launch_mlx<__half>(vector, packed_weights, scales, biases, output, stream, ffi::F16);
}

ffi::Error mlx_bf16(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ffi::BufferR2<ffi::BF16> scales,
    ffi::BufferR2<ffi::BF16> biases,
    ffi::Result<ffi::AnyBuffer> output,
    cudaStream_t stream) {
  return launch_mlx<__nv_bfloat16>(vector, packed_weights, scales, biases, output, stream, ffi::BF16);
}

ffi::Error awq_f16(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ffi::BufferR2<ffi::F16> scales,
    ffi::BufferR2<ffi::U8> packed_zero_points,
    ffi::Result<ffi::AnyBuffer> output,
    cudaStream_t stream) {
  return launch_awq<__half>(vector, packed_weights, scales, packed_zero_points, output, stream, ffi::F16);
}

ffi::Error awq_bf16(
    ffi::AnyBuffer vector,
    ffi::BufferR2<ffi::U8> packed_weights,
    ffi::BufferR2<ffi::BF16> scales,
    ffi::BufferR2<ffi::U8> packed_zero_points,
    ffi::Result<ffi::AnyBuffer> output,
    cudaStream_t stream) {
  return launch_awq<__nv_bfloat16>(vector, packed_weights, scales, packed_zero_points, output, stream, ffi::BF16);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lalamo_w4a16_mlx_f16,
    mlx_f16,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::BufferR2<ffi::U8>>()
        .Arg<ffi::BufferR2<ffi::F16>>()
        .Arg<ffi::BufferR2<ffi::F16>>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lalamo_w4a16_mlx_bf16,
    mlx_bf16,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::BufferR2<ffi::U8>>()
        .Arg<ffi::BufferR2<ffi::BF16>>()
        .Arg<ffi::BufferR2<ffi::BF16>>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lalamo_w4a16_awq_f16,
    awq_f16,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::BufferR2<ffi::U8>>()
        .Arg<ffi::BufferR2<ffi::F16>>()
        .Arg<ffi::BufferR2<ffi::U8>>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lalamo_w4a16_awq_bf16,
    awq_bf16,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::BufferR2<ffi::U8>>()
        .Arg<ffi::BufferR2<ffi::BF16>>()
        .Arg<ffi::BufferR2<ffi::U8>>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
