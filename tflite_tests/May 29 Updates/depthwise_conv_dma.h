/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DEPTHWISE_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DEPTHWISE_CONV_H_

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

#include <stdint.h>

#define DMA_BASE 0x00010000 //defined in vivado

#define CH_OFFSET(n)   ((n) * 0x20)
#define REG_SRC        0x00
#define REG_DST        0x04
#define REG_LEN        0x08
#define REG_CTRL       0x0C
#define REG_STATUS     0x10
#define REG_SPAD_SEL   0x14

// SPAD select encoding — must match top.sv and the DMA RTL
#define SPAD_WEIGHTS   0b000
#define SPAD_IFMAPS    0b001
#define SPAD_BIAS      0b010
#define SPAD_SCALE     0b011
#define SPAD_SHIFT     0b100

#define CSR_BASE       0x00001000

#define SPAD_BASE      0xC0000000 //spad write base
#define SPAD_W_BASE    0xC0000000 //weight spad
#define SPAD_I_BASE    0xC0002000 //input spad
#define SPAD_B_BASE    0xC0004000 //bias spad
#define SPAD_M_BASE    0xC0006000 //mult spad
#define SPAD_S_BASE    0xC0008000 //shift spad
#define SPAD_O_BASE    0x20000000 //output spad

// Macro to write any DMA register
#define DMA_REG(ch, reg) \
    (*((volatile uint32_t *)(DMA_BASE + CH_OFFSET(ch) + (reg))))

#define DMA_STATUS(ch)  DMA_REG(ch, REG_STATUS)
#define DONE_BIT        (1 << 1)

// Please follow the order of SRC, DST, LEN, SPAD SEL, and last should be CTRL
static void dma_load_weights(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
    DMA_REG(0, REG_SRC)      = dram_src;
    DMA_REG(0, REG_DST)      = spad_dst;
    DMA_REG(0, REG_LEN)      = len;
    DMA_REG(0, REG_SPAD_SEL) = SPAD_WEIGHTS;  // 0b000
    DMA_REG(0, REG_CTRL)     = 0x1;           // START — must be last
    uint32_t timeout = 10000000u;
    while (!(DMA_STATUS(1) & DONE_BIT)){
        if (!timeout){
            MicroPrintf("Timeout.");
        }
        timeout--;
    };
    DMA_STATUS(0) = DONE_BIT;                 // W1C clear
}

static void dma_load_inputs(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
    DMA_REG(1, REG_SRC)      = dram_src;
    DMA_REG(1, REG_DST)      = spad_dst;
    DMA_REG(1, REG_LEN)      = len;
    DMA_REG(1, REG_SPAD_SEL) = SPAD_IFMAPS;   // 0b001
    DMA_REG(1, REG_CTRL)     = 0x1;
    uint32_t timeout = 10000000u;
    while (!(DMA_STATUS(1) & DONE_BIT)){
        if (!timeout){
            MicroPrintf("Timeout.");
        }
        timeout--;
    };
    DMA_STATUS(1) = DONE_BIT;
}

static void dma_load_bias(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
    DMA_REG(2, REG_SRC)      = dram_src;
    DMA_REG(2, REG_DST)      = spad_dst;
    DMA_REG(2, REG_LEN)      = len;
    DMA_REG(2, REG_SPAD_SEL) = SPAD_BIAS;     // 0b010
    DMA_REG(2, REG_CTRL)     = 0x1;
    uint32_t timeout = 10000000u;
    while (!(DMA_STATUS(1) & DONE_BIT)){
        if (!timeout){
            MicroPrintf("Timeout.");
        }
        timeout--;
    };
    DMA_STATUS(2) = DONE_BIT;
}

static void dma_load_scale(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
    DMA_REG(3, REG_SRC)      = dram_src;
    DMA_REG(3, REG_DST)      = spad_dst;
    DMA_REG(3, REG_LEN)      = len;
    DMA_REG(3, REG_SPAD_SEL) = SPAD_SCALE;  // 0b011
    DMA_REG(3, REG_CTRL)     = 0x1;           
    uint32_t timeout = 10000000u;
    while (!(DMA_STATUS(1) & DONE_BIT)){
        if (!timeout){
            MicroPrintf("Timeout.");
        }
        timeout--;
    };
    DMA_STATUS(3) = DONE_BIT;                 
}

static void dma_load_shift(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
    DMA_REG(0, REG_SRC)      = dram_src;
    DMA_REG(0, REG_DST)      = spad_dst;
    DMA_REG(0, REG_LEN)      = len;
    DMA_REG(0, REG_SPAD_SEL) = SPAD_SHIFT;   // 0b100
    DMA_REG(0, REG_CTRL)     = 0x1;
    uint32_t timeout = 10000000u;
    while (!(DMA_STATUS(1) & DONE_BIT)){
        if (!timeout){
            MicroPrintf("Timeout.");
        }
        timeout--;
    };
    DMA_STATUS(0) = DONE_BIT;
}

static void dma_load_csr(uint32_t dram_src, uint32_t csr_dst, uint32_t len) {
    DMA_REG(1, REG_SRC)      = dram_src;
    DMA_REG(1, REG_DST)      = csr_dst;
    DMA_REG(1, REG_LEN)      = len;
    DMA_REG(1, REG_SPAD_SEL) = 0b000;
    DMA_REG(1, REG_CTRL)     = 0x1;
    uint32_t timeout = 10000000u;
    while (!(DMA_STATUS(1) & DONE_BIT)){
        if (!timeout){
            MicroPrintf("Timeout.");
        }
        timeout--;
    };
    DMA_STATUS(1) = DONE_BIT;
}

static inline void RawPutcD(char c) {
  volatile uint32_t* const uart_tx =
      reinterpret_cast<volatile uint32_t*>(0x40600000u + 0x04u);
  volatile uint32_t* const uart_status =
      reinterpret_cast<volatile uint32_t*>(0x40600000u + 0x08u);

  while ((*uart_status) & 0x08u) {
  }

  *uart_tx = static_cast<uint32_t>(static_cast<uint8_t>(c));
}

static inline void RawNewlineD() {
  RawPutcD('\r');
  RawPutcD('\n');
}

static inline void RawPutHexNibbleD(uint32_t v) {
  v &= 0xFu;
  RawPutcD((v < 10u) ? static_cast<char>('0' + v)
                    : static_cast<char>('A' + (v - 10u)));
}

static inline void RawPutHex32D(uint32_t v) {
  for (int shift = 28; shift >= 0; shift -= 4) {
    RawPutHexNibbleD(v >> shift);
  }
}

static inline void RawTagHexD(char tag, uint32_t v) {
  RawPutcD(tag);
  RawPutHex32D(v);
  RawPutcD(' ');
}


namespace tflite {
namespace reference_integer_ops {
// inline void DepthwiseConvPerChannel(
//     const DepthwiseParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
//   // Get parameters.
//   // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int depth_multiplier = params.depth_multiplier;
//   const int32_t input_offset = params.input_offset;
//   const int32_t output_offset = params.output_offset;
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Check dimensions of the tensors.
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int input_depth = input_shape.Dims(3);
//   // const int input_height = input_shape.Dims(1);
//   // volatile int input_width_v = input_shape.Dims(2);
//   // const int input_width = input_width_v;
//   // const int input_depth = input_shape.Dims(3);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   TFLITE_DCHECK_EQ(output_depth, input_shape.Dims(3) * depth_multiplier);
//   TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('S'); RawPutcD(']');
//   RawTagHexD('b', static_cast<uint32_t>(batches));
//   RawTagHexD('h', static_cast<uint32_t>(input_height));
//   RawTagHexD('w', static_cast<uint32_t>(input_width));
//   RawTagHexD('W', static_cast<uint32_t>(input_shape.Dims(2)));
//   RawTagHexD('c', static_cast<uint32_t>(input_depth));
//   RawTagHexD('H', static_cast<uint32_t>(output_height));
//   RawTagHexD('W', static_cast<uint32_t>(output_width));
//   RawTagHexD('C', static_cast<uint32_t>(output_depth));
//   RawTagHexD('o', static_cast<uint32_t>(
//                      reinterpret_cast<uintptr_t>(output_data)));
//   RawNewlineD();
//   RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('I'); RawPutcD(']');
//   RawTagHexD('0', static_cast<uint32_t>(input_shape.Dims(0)));
//   RawTagHexD('1', static_cast<uint32_t>(input_shape.Dims(1)));
//   RawTagHexD('2', static_cast<uint32_t>(input_shape.Dims(2)));
//   RawTagHexD('3', static_cast<uint32_t>(input_shape.Dims(3)));
//   RawNewlineD();

//   RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('O'); RawPutcD(']');
//   RawTagHexD('0', static_cast<uint32_t>(output_shape.Dims(0)));
//   RawTagHexD('1', static_cast<uint32_t>(output_shape.Dims(1)));
//   RawTagHexD('2', static_cast<uint32_t>(output_shape.Dims(2)));
//   RawTagHexD('3', static_cast<uint32_t>(output_shape.Dims(3)));
//   RawNewlineD();

//   RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('F'); RawPutcD(']');
//   RawTagHexD('0', static_cast<uint32_t>(filter_shape.Dims(0)));
//   RawTagHexD('1', static_cast<uint32_t>(filter_shape.Dims(1)));
//   RawTagHexD('2', static_cast<uint32_t>(filter_shape.Dims(2)));
//   RawTagHexD('3', static_cast<uint32_t>(filter_shape.Dims(3)));
//   RawNewlineD();
//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
//           for (int m = 0; m < depth_multiplier; ++m) {
//             const int output_channel = m + in_channel * depth_multiplier;
//             const int in_x_origin = (out_x * stride_width) - pad_width;
//             const int in_y_origin = (out_y * stride_height) - pad_height;
//             int32_t acc = 0;
//             for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//               for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//                 const int in_x = in_x_origin + dilation_width_factor * filter_x;
//                 const int in_y =
//                     in_y_origin + dilation_height_factor * filter_y;
//                 // Zero padding by omitting the areas outside the image.
//                 const bool is_point_inside_image =
//                     (in_x >= 0) && (in_x < input_shape.Dims(2)) && (in_y >= 0) &&
//                     (in_y < input_height);
//                 if (is_point_inside_image) {
//                   int32_t input_val = input_data[Offset(
//                       input_shape, batch, in_y, in_x, in_channel)];
//                   int32_t filter_val = filter_data[Offset(
//                       filter_shape, 0, filter_y, filter_x, output_channel)];
//                   // Accumulate with 32 bits accumulator.
//                   // In the nudging process during model quantization, we force
//                   // real value of 0.0 be represented by a quantized value. This
//                   // guarantees that the input_offset is a int8_t, even though
//                   // it is represented using int32_t. int32_t += int8_t *
//                   // (int8_t - int8_t) so the highest value we can get from each
//                   // accumulation is [-127, 127] * ([-128, 127] -
//                   // [-128, 127]), which is [-32512, 32512]. log2(32512)
//                   // = 14.98, which means we can accumulate at least 2^16
//                   // multiplications without overflow. The accumulator is
//                   // applied to a filter so the accumulation logic will hold as
//                   // long as the filter size (filter_y * filter_x * in_channel)
//                   // does not exceed 2^16, which is the case in all the models
//                   // we have seen so far.
//                   // TODO(b/174275578): Add a check to make sure the
//                   // accumulator depth is smaller than 2^16.
//                   acc += filter_val * (input_val + input_offset);
//                 }
//               }
//             }
//             if (bias_data) {
//               acc += bias_data[output_channel];
//             }
//             acc = MultiplyByQuantizedMultiplier(
//                 acc, output_multiplier[output_channel],
//                 output_shift[output_channel]);
//             acc += output_offset;
//             acc = std::max(acc, output_activation_min);
//             acc = std::min(acc, output_activation_max);
//                         const int out_offset =
//                 Offset(output_shape, batch, out_y, out_x, output_channel);

//             if (batch == 0 && out_y == 0 && out_x == 0 &&
//                 in_channel == 0 && m == 0) {
//               RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('W'); RawPutcD(']');
//               RawTagHexD('a', static_cast<uint32_t>(acc));
//               RawTagHexD('p', static_cast<uint32_t>(
//                               reinterpret_cast<uintptr_t>(&output_data[out_offset])));
//               RawNewlineD();
//             }

//             output_data[out_offset] = static_cast<int8_t>(acc);
//             // output_data[Offset(output_shape, batch, out_y, out_x,
//             //                    output_channel)] = static_cast<int8_t>(acc);
//           }
//         }
//       }
//     }
//   }
// }

// inline void DepthwiseConvPerChannel(
//     const DepthwiseParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
//   // Get parameters.
//   // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int depth_multiplier = params.depth_multiplier;
//   const int32_t input_offset = params.input_offset;
//   const int32_t output_offset = params.output_offset;
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Check dimensions of the tensors.
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int input_depth = input_shape.Dims(3);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   TFLITE_DCHECK_EQ(output_depth, input_shape.Dims(3) * depth_multiplier);
//   TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('S'); RawPutcD(']');
//   // RawTagHexD('b', static_cast<uint32_t>(batches));
//   // RawTagHexD('h', static_cast<uint32_t>(input_height));
//   // RawTagHexD('w', static_cast<uint32_t>(input_width));
//   // RawTagHexD('c', static_cast<uint32_t>(input_depth));
//   // RawTagHexD('f', static_cast<uint32_t>(filter_height));
//   // RawTagHexD('g', static_cast<uint32_t>(filter_width));
//   // RawTagHexD('H', static_cast<uint32_t>(output_height));
//   // RawTagHexD('W', static_cast<uint32_t>(output_width));
//   // RawTagHexD('C', static_cast<uint32_t>(output_depth));
//   // RawTagHexD('o', static_cast<uint32_t>(
//   //                    reinterpret_cast<uintptr_t>(output_data)));
//   // RawNewlineD();

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('I'); RawPutcD(']');
//   // RawTagHexD('0', static_cast<uint32_t>(input_shape.Dims(0)));
//   // RawTagHexD('1', static_cast<uint32_t>(input_shape.Dims(1)));
//   // RawTagHexD('2', static_cast<uint32_t>(input_shape.Dims(2)));
//   // RawTagHexD('3', static_cast<uint32_t>(input_shape.Dims(3)));
//   // RawNewlineD();

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('O'); RawPutcD(']');
//   // RawTagHexD('0', static_cast<uint32_t>(output_shape.Dims(0)));
//   // RawTagHexD('1', static_cast<uint32_t>(output_shape.Dims(1)));
//   // RawTagHexD('2', static_cast<uint32_t>(output_shape.Dims(2)));
//   // RawTagHexD('3', static_cast<uint32_t>(output_shape.Dims(3)));
//   // RawNewlineD();

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('F'); RawPutcD(']');
//   // RawTagHexD('0', static_cast<uint32_t>(filter_shape.Dims(0)));
//   // RawTagHexD('1', static_cast<uint32_t>(filter_shape.Dims(1)));
//   // RawTagHexD('2', static_cast<uint32_t>(filter_shape.Dims(2)));
//   // RawTagHexD('3', static_cast<uint32_t>(filter_shape.Dims(3)));
//   // RawNewlineD();

//   for (int batch = 0; batch < MatchingDim(input_shape, 0, output_shape, 0);
//        ++batch) {
//     for (int out_y = 0; out_y < output_shape.Dims(1); ++out_y) {
//       for (int out_x = 0; out_x < output_shape.Dims(2); ++out_x) {
//         for (int in_channel = 0; in_channel < input_shape.Dims(3);
//              ++in_channel) {
//           for (int m = 0; m < depth_multiplier; ++m) {
//             const int output_channel = m + in_channel * depth_multiplier;
//             const int in_x_origin = (out_x * stride_width) - pad_width;
//             const int in_y_origin = (out_y * stride_height) - pad_height;
//             int32_t acc = 0;
//             for (int filter_y = 0; filter_y < filter_shape.Dims(1);
//                  ++filter_y) {
//               for (int filter_x = 0; filter_x < filter_shape.Dims(2);
//                    ++filter_x) {
//                 const int in_x = in_x_origin + dilation_width_factor * filter_x;
//                 const int in_y =
//                     in_y_origin + dilation_height_factor * filter_y;
//                 // Zero padding by omitting the areas outside the image.
//                 const bool is_point_inside_image =
//                     (in_x >= 0) && (in_x < input_shape.Dims(2)) &&
//                     (in_y >= 0) && (in_y < input_shape.Dims(1));
//                 if (is_point_inside_image) {
//                   int32_t input_val = input_data[Offset(
//                       input_shape, batch, in_y, in_x, in_channel)];
//                   int32_t filter_val = filter_data[Offset(
//                       filter_shape, 0, filter_y, filter_x, output_channel)];
//                   // Accumulate with 32 bits accumulator.
//                   // In the nudging process during model quantization, we force
//                   // real value of 0.0 be represented by a quantized value. This
//                   // guarantees that the input_offset is a int8_t, even though
//                   // it is represented using int32_t. int32_t += int8_t *
//                   // (int8_t - int8_t) so the highest value we can get from each
//                   // accumulation is [-127, 127] * ([-128, 127] -
//                   // [-128, 127]), which is [-32512, 32512]. log2(32512)
//                   // = 14.98, which means we can accumulate at least 2^16
//                   // multiplications without overflow. The accumulator is
//                   // applied to a filter so the accumulation logic will hold as
//                   // long as the filter size (filter_y * filter_x * in_channel)
//                   // does not exceed 2^16, which is the case in all the models
//                   // we have seen so far.
//                   // TODO(b/174275578): Add a check to make sure the
//                   // accumulator depth is smaller than 2^16.
//                   acc += filter_val * (input_val + input_offset);
//                 }
//               }
//             }
//             if (bias_data) {
//               acc += bias_data[output_channel];
//             }
//             acc = MultiplyByQuantizedMultiplier(
//                 acc, output_multiplier[output_channel],
//                 output_shift[output_channel]);
//             acc += output_offset;
//             acc = std::max(acc, output_activation_min);
//             acc = std::min(acc, output_activation_max);

//             const int out_offset =
//                 Offset(output_shape, batch, out_y, out_x, output_channel);

//             if (batch == 0 && out_y == 0 && out_x == 0 &&
//                 in_channel == 0 && m == 0) {
//               RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('W'); RawPutcD(']');
//               RawTagHexD('a', static_cast<uint32_t>(acc));
//               RawTagHexD('p', static_cast<uint32_t>(
//                                   reinterpret_cast<uintptr_t>(
//                                       &output_data[out_offset])));
//               RawNewlineD();
//             }

//             output_data[out_offset] = static_cast<int8_t>(acc);

//             if (batch == 0 && out_y == 0 && out_x == 0 &&
//                 in_channel == 0 && m == 0) {
//               RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('A'); RawPutcD(']');
//               RawTagHexD('v', static_cast<uint32_t>(
//                                   static_cast<uint8_t>(
//                                       output_data[out_offset])));
//               RawNewlineD();
//             }

//             // output_data[Offset(output_shape, batch, out_y, out_x,
//             //                    output_channel)] = static_cast<int8_t>(acc);
//           }
//         }
//       }
//     }
//   }
// }

// inline void DepthwiseConvPerChannel(
//     const DepthwiseParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
//   // Get parameters.
//   // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int depth_multiplier = params.depth_multiplier;
//   const int32_t input_offset = params.input_offset;
//   const int32_t output_offset = params.output_offset;
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Check dimensions of the tensors.
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int input_depth = input_shape.Dims(3);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   TFLITE_DCHECK_EQ(output_depth, input_shape.Dims(3) * depth_multiplier);
//   TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('S'); RawPutcD(']');
//   // RawTagHexD('b', static_cast<uint32_t>(batches));
//   // RawTagHexD('h', static_cast<uint32_t>(input_height));
//   // RawTagHexD('w', static_cast<uint32_t>(input_width));
//   // RawTagHexD('c', static_cast<uint32_t>(input_depth));
//   // RawTagHexD('f', static_cast<uint32_t>(filter_height));
//   // RawTagHexD('g', static_cast<uint32_t>(filter_width));
//   // RawTagHexD('m', static_cast<uint32_t>(depth_multiplier));
//   // RawTagHexD('H', static_cast<uint32_t>(output_height));
//   // RawTagHexD('W', static_cast<uint32_t>(output_width));
//   // RawTagHexD('C', static_cast<uint32_t>(output_depth));
//   // RawTagHexD('o', static_cast<uint32_t>(
//   //                    reinterpret_cast<uintptr_t>(output_data)));
//   // RawNewlineD();

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('I'); RawPutcD(']');
//   // RawTagHexD('0', static_cast<uint32_t>(input_shape.Dims(0)));
//   // RawTagHexD('1', static_cast<uint32_t>(input_shape.Dims(1)));
//   // RawTagHexD('2', static_cast<uint32_t>(input_shape.Dims(2)));
//   RawTagHexD('3', static_cast<uint32_t>(input_shape.Dims(3)));
//   // RawNewlineD();

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('O'); RawPutcD(']');
//   // RawTagHexD('0', static_cast<uint32_t>(output_shape.Dims(0)));
//   // RawTagHexD('1', static_cast<uint32_t>(output_shape.Dims(1)));
//   // RawTagHexD('2', static_cast<uint32_t>(output_shape.Dims(2)));
//   RawTagHexD('3', static_cast<uint32_t>(output_shape.Dims(3)));
//   // RawNewlineD();

//   // RawPutcD('['); RawPutcD('D'); RawPutcD('S'); RawPutcD('F'); RawPutcD(']');
//   // RawTagHexD('0', static_cast<uint32_t>(filter_shape.Dims(0)));
//   // RawTagHexD('1', static_cast<uint32_t>(filter_shape.Dims(1)));
//   // RawTagHexD('2', static_cast<uint32_t>(filter_shape.Dims(2)));
//   RawTagHexD('3', static_cast<uint32_t>(filter_shape.Dims(3)));
//   // RawNewlineD();

//   for (int batch = 0; batch < MatchingDim(input_shape, 0, output_shape, 0);
//        ++batch) {
//     for (int out_y = 0; out_y < output_shape.Dims(1); ++out_y) {
//       for (int out_x = 0; out_x < output_shape.Dims(2); ++out_x) {
//         for (int in_channel = 0; in_channel < input_shape.Dims(3);
//              ++in_channel) {
//           for (int m = 0; m < depth_multiplier; ++m) {
//             const int output_channel = m + in_channel * depth_multiplier;
//             const int in_x_origin = (out_x * stride_width) - pad_width;
//             const int in_y_origin = (out_y * stride_height) - pad_height;
//             int32_t acc = 0;

//             const bool dump_dw =
//                 (batch == 0 && out_y == 0 && out_x == 0 &&
//                  in_channel < 4 && m == 0);

//             if (dump_dw) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('0'); RawPutcD(']');
//               // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//               // RawTagHexD('m', static_cast<uint32_t>(m));
//               // RawTagHexD('o', static_cast<uint32_t>(output_channel));
//               // RawTagHexD('x', static_cast<uint32_t>(out_x));
//               RawTagHexD('y', static_cast<uint32_t>(out_y));
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawNewlineD();
//             }

//             for (int filter_y = 0; filter_y < filter_shape.Dims(1);
//                  ++filter_y) {
//               for (int filter_x = 0; filter_x < filter_shape.Dims(2);
//                    ++filter_x) {
//                 const int in_x = in_x_origin + dilation_width_factor * filter_x;
//                 const int in_y =
//                     in_y_origin + dilation_height_factor * filter_y;

//                 // Zero padding by omitting the areas outside the image.
//                 const bool is_point_inside_image =
//                     (in_x >= 0) && (in_x < input_shape.Dims(2)) &&
//                     (in_y >= 0) && (in_y < input_shape.Dims(1));

//                 if (is_point_inside_image) {
//                   const int input_offset_index =
//                       Offset(input_shape, batch, in_y, in_x, in_channel);
//                   const int filter_offset_index =
//                       Offset(filter_shape, 0, filter_y, filter_x,
//                              output_channel);

//                   int32_t input_val = input_data[input_offset_index];
//                   int32_t filter_val = filter_data[filter_offset_index];

//                   const int32_t input_plus_offset = input_val + input_offset;
//                   const int32_t product = filter_val * input_plus_offset;

//                   acc += product;

//                   if (dump_dw) {
//                     // RawPutcD('['); RawPutcD('D'); RawPutcD('1'); RawPutcD(']');
//                     // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//                     // RawTagHexD('f', static_cast<uint32_t>(filter_y));
//                     // RawTagHexD('g', static_cast<uint32_t>(filter_x));
//                     // RawTagHexD('y', static_cast<uint32_t>(in_y));
//                     // RawTagHexD('x', static_cast<uint32_t>(in_x));
//                     // RawTagHexD('I', static_cast<uint32_t>(input_offset_index));
//                     // RawTagHexD('F', static_cast<uint32_t>(filter_offset_index));
//                     RawTagHexD('i', static_cast<uint32_t>(input_val));
//                     // RawTagHexD('w', static_cast<uint32_t>(filter_val));
//                     // RawTagHexD('z', static_cast<uint32_t>(input_offset));
//                     // RawTagHexD('p', static_cast<uint32_t>(product));
//                     // RawTagHexD('a', static_cast<uint32_t>(acc));
//                     // RawNewlineD();
//                   }
//                 } else {
//                   if (dump_dw) {
//                     // RawPutcD('['); RawPutcD('D'); RawPutcD('P'); RawPutcD(']');
//                     // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//                     // RawTagHexD('f', static_cast<uint32_t>(filter_y));
//                     RawTagHexD('g', static_cast<uint32_t>(filter_x));
//                     // RawTagHexD('y', static_cast<uint32_t>(in_y));
//                     // RawTagHexD('x', static_cast<uint32_t>(in_x));
//                     // RawNewlineD();
//                   }
//                 }
//               }
//             }

//             if (dump_dw) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('2'); RawPutcD(']');
//               // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawNewlineD();
//             }

//             if (bias_data) {
//               const int32_t bias = bias_data[output_channel];

//               if (dump_dw) {
//                 // RawPutcD('['); RawPutcD('D'); RawPutcD('3'); RawPutcD(']');
//                 // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//                 // RawTagHexD('b', static_cast<uint32_t>(bias));
//                 // RawTagHexD('a', static_cast<uint32_t>(acc));
//                 // RawNewlineD();
//               }

//               acc += bias;

//               if (dump_dw) {
//                 // RawPutcD('['); RawPutcD('D'); RawPutcD('4'); RawPutcD(']');
//                 // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//                 // RawTagHexD('a', static_cast<uint32_t>(acc));
//                 // RawNewlineD();
//               }
//             }

//             const int32_t mult = output_multiplier[output_channel];
//             const int32_t shift = output_shift[output_channel];

//             if (dump_dw) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('5'); RawPutcD(']');
//               // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawTagHexD('m', static_cast<uint32_t>(mult));
//               // RawTagHexD('s', static_cast<uint32_t>(shift));
//               // RawNewlineD();
//             }

//             acc = MultiplyByQuantizedMultiplier(acc, mult, shift);

//             if (dump_dw) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('6'); RawPutcD(']');
//               // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawTagHexD('o', static_cast<uint32_t>(output_offset));
//               // RawNewlineD();
//             }

//             acc += output_offset;

//             if (dump_dw) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('7'); RawPutcD(']');
//               // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawTagHexD('n', static_cast<uint32_t>(output_activation_min));
//               // RawTagHexD('x', static_cast<uint32_t>(output_activation_max));
//               // RawNewlineD();
//             }

//             acc = std::max(acc, output_activation_min);
//             acc = std::min(acc, output_activation_max);

//             if (dump_dw) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('8'); RawPutcD(']');
//               // RawTagHexD('c', static_cast<uint32_t>(in_channel));
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawNewlineD();
//             }

//             const int out_offset =
//                 Offset(output_shape, batch, out_y, out_x, output_channel);

//             if (batch == 0 && out_y == 0 && out_x == 0 &&
//                 in_channel == 0 && m == 0) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('W'); RawPutcD(']');
//               // RawTagHexD('a', static_cast<uint32_t>(acc));
//               // RawTagHexD('p', static_cast<uint32_t>(
//               //                     reinterpret_cast<uintptr_t>(
//               //                         &output_data[out_offset])));
//               // RawNewlineD();
//             }

//             output_data[out_offset] = static_cast<int8_t>(acc);

//             if (batch == 0 && out_y == 0 && out_x == 0 &&
//                 in_channel == 0 && m == 0) {
//               // RawPutcD('['); RawPutcD('D'); RawPutcD('W'); RawPutcD('A'); RawPutcD(']');
//               // RawTagHexD('v', static_cast<uint32_t>(
//               //                     static_cast<uint8_t>(
//               //                         output_data[out_offset])));
//               // RawNewlineD();
//             }
//           }
//         }
//       }
//     }
//   }
// }



inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32_t* output_mult,
    const int32_t* output_shift, const RuntimeShape& i_shape,
    const int8_t* i_data, const RuntimeShape& w_shape,
    const int8_t* w_data, const RuntimeShape& b_shape,
    const int32_t* b_data, const RuntimeShape& o_shape,
    int8_t* o_data) {
    static int dw_call_count = 0;
    const int this_dw_call = dw_call_count++;

    const int stride_w = params.stride_width;
    const int stride_h = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = -(params.input_offset);
    const int32_t* output_offset = output_shift;
    const int32_t output_off = params.output_offset; 
    const int32_t* output_multiplier = output_mult;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    const int32_t zero_point=input_offset;
    const int layer_type = 0; //Pointwise=0, Depthwise=1
    const int input_shape[] = {i_shape.Dims(0), i_shape.Dims(1), i_shape.Dims(2), i_shape.Dims(3)};
    const int kernel_shape[] = {w_shape.Dims(0), w_shape.Dims(1), w_shape.Dims(2), w_shape.Dims(3)};
    const int output_shape[] = {o_shape.Dims(0), o_shape.Dims(1), o_shape.Dims(2), o_shape.Dims(3)};
    
    //SPAD Size (precalculated)
    int plm_in=8192;
    int plm_w=8192;
    int plm_out=8192;

    //Unknowns
    int p_mode=0;
    int tile_shape[4]; //input tile
    int w_tile_shape[4]; //weight tile
    int o_tile_shape[4]; //output tile
    int tile_number; //how many tiles there are
    int height_number; //how many parts height is divided into
    int width_number; //how many parts width is divided into
    int channel_number; //how many parts channel is divided into
    int batch_number; //how many batches
    int systolic_size[]={8, 8};
    int overlap_h=kernel_shape[1]-stride_h;
    int overlap_w=kernel_shape[2]-stride_w;

    const int pad_height = (output_shape[1]*stride_h)+overlap_h-input_shape[1];
    const int pad_width = (output_shape[1]*stride_h)+overlap_h-input_shape[1];

    int padding_up=0;/**/
    int padding_left=0;/**/
    int padding_down=0; /**/
    int padding_right=0; /**/
    if (pad_height%2==0){
        padding_up=(pad_height)/2;
        padding_down=(pad_height)/2;
    }
    else {
        padding_up=pad_height/2;
        padding_down=pad_height/2+1;
    }
    if (((output_shape[2]*stride_w)+overlap_w-input_shape[2])%2==0){
        padding_left=pad_width/2;
        padding_right=pad_width/2;
    }
    else {
        padding_left=pad_width/2;
        padding_right=pad_width/2+1;
    }

    int padded_input_shape[] = {input_shape[0], input_shape[1]+padding_up+padding_down, input_shape[2]+padding_left+padding_right, input_shape[3]};

    //tile batch size
    tile_shape[0]=input_shape[0];
    w_tile_shape[0]=kernel_shape[0];
    o_tile_shape[0]=output_shape[0];
    
    //tile channel size
    tile_shape[3]=input_shape[3];
    w_tile_shape[3]=kernel_shape[3];
    o_tile_shape[3]=output_shape[3];
    
    //tile width size
    tile_shape[2]=padded_input_shape[2];
    w_tile_shape[2]=kernel_shape[2];
    o_tile_shape[2]=output_shape[2];
    if (padded_input_shape[2]>=systolic_size[1]){
        tile_shape[2]=systolic_size[1];
        int stride_correction_w=kernel_shape[2];
        o_tile_shape[2]=1;
        while(stride_correction_w+stride_w<=systolic_size[1]){
            stride_correction_w=stride_correction_w+stride_w;
            o_tile_shape[2]=o_tile_shape[2]+1;
        }
        while(output_shape[2]%o_tile_shape[2]!=0){
            o_tile_shape[2]=o_tile_shape[2]-1;
            stride_correction_w=stride_correction_w-stride_w;
        }
        if (systolic_size[1]>=stride_correction_w){
            tile_shape[2]=stride_correction_w;
        }
    }
    
    //tile height size
    tile_shape[1]=padded_input_shape[1];
    w_tile_shape[1]=kernel_shape[1];
    o_tile_shape[1]=output_shape[1];
    if (padded_input_shape[1]>=systolic_size[0]){
        tile_shape[1]=systolic_size[0];
        int stride_correction_h=kernel_shape[1];
        o_tile_shape[1]=1;
        while(stride_correction_h+stride_h<=systolic_size[0]){
            stride_correction_h=stride_correction_h+stride_h;
            o_tile_shape[1]=o_tile_shape[1]+1;
        }
        while(output_shape[1]%o_tile_shape[1]!=0){
            o_tile_shape[1]=o_tile_shape[1]-1;
            stride_correction_h=stride_correction_h-stride_h;
        }
        if (systolic_size[0]>=stride_correction_h){
            tile_shape[1]=stride_correction_h;
        }
    }   

    //calculate current tile memory requirements
    int mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
    int mem_w=kernel_shape[1]*kernel_shape[2]*kernel_shape[3];
    int mem_out=o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
    
    //if (layer_type!=0){ //if depthwise
        while (tile_shape[3]>1&&(plm_w<mem_w||plm_out<mem_out)){
            //half the channel size
            tile_shape[3]=(tile_shape[3]%2==0)? (tile_shape[3]/2):(tile_shape[3]/2+1);
            mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
            mem_w=kernel_shape[1]*kernel_shape[2]*kernel_shape[3];
            mem_out=o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
        }
    /*}
    else{ //if pointwise
    
        //output channels=kernel batches
        //set the batch size to 1
        w_tile_shape[0]=1;
        o_tile_shape[3]=w_tile_shape[0];
        while (tile_shape[1]>1&&tile_shape[2]>1&&(plm_w<mem_w||plm_out<mem_out)){
                //decrement the tile height and width (to stay square)
                tile_shape[1]=tile_shape[1]-1;
                tile_shape[2]=tile_shape[2]-1;
                
                mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
                mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
                mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
    }*/

    //how many channel iterations
    
    if(input_shape[3]%tile_shape[3]==0){
        channel_number=input_shape[3]/tile_shape[3];
    }
    else{
        channel_number=input_shape[3]/tile_shape[3]+1;
    }
    //how many width iterations
    width_number=1;
    int width_iter=padded_input_shape[2]-tile_shape[2];
    while(width_iter>0){
        width_number+=1;
        width_iter=width_iter-tile_shape[2]+overlap_w;
    }
    //how many height iterations
    height_number=1; 
    int height_iter=padded_input_shape[1]-tile_shape[1];
    while(height_iter>0){
        height_number+=1;
        height_iter=height_iter-tile_shape[1]+overlap_h;
    }
    //how many batches
    batch_number=kernel_shape[0];
    //how many tiles per batch
    tile_number=channel_number*width_number*height_number;

   //how many output channel iterations
    int o_channel_number;
    if(output_shape[3]%o_tile_shape[3]==0){
        o_channel_number=output_shape[3]/o_tile_shape[3];
    }
    else{
        o_channel_number=output_shape[3]/o_tile_shape[3]+1;
    }    
    //how many output width iterations
    int o_width_number=1;
    int o_width_iter=output_shape[2]-o_tile_shape[2];
    while(o_width_iter>0){
        o_width_number+=1;
        o_width_iter=o_width_iter-o_tile_shape[2];
    }
    //how many output height iterations
    int o_height_number=1; 
    int o_height_iter=output_shape[1]-o_tile_shape[1];
    while(o_height_iter>0){
        o_height_number+=1;
        o_height_iter=o_height_iter-o_tile_shape[1];
    }

    int tile_size=tile_shape[0]*tile_shape[1]*tile_shape[2]*tile_shape[3];
    int8_t tile_data[tile_size];

    int o_tile_size=o_tile_shape[0]*o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
    int8_t o_tile_data[o_tile_size];

    int w_tile_size=w_tile_shape[0]*kernel_shape[1]*kernel_shape[2]*w_tile_shape[3];
    int8_t w_tile_data[w_tile_size];
    
    int32_t bias_data[o_tile_shape[3]];
    int32_t qmult_data[o_tile_shape[3]];
    int32_t qshift_data[o_tile_shape[3]]; 

    int o_height_count=output_shape[1];
    int o_width_count=output_shape[2];
    int o_channel_count=output_shape[3];

    //int kernel_size=kernel_shape[0]*kernel_shape[1]*kernel_shape[2]*kernel_shape[3];
    //int32_t weightsum=0;
    //for(int element=0; element<kernel_size; element++){
    //    weightsum = weightsum+w_data[element];
    //}

    for(int batch=0; batch<kernel_shape[0]; batch++){
        int tile=0;
        int height_count=padded_input_shape[1];
        int width_count=padded_input_shape[2];
        int channel_count=padded_input_shape[3];
        
        for(int real_tile=0; real_tile<tile_number; real_tile++){
            if (this_dw_call == 0) {
                RawPutcD('['); RawPutcD('D'); RawPutcD('H'); RawPutcD(']');
                RawTagHexD('k', static_cast<uint32_t>(this_dw_call));
                RawTagHexD('B', static_cast<uint32_t>(kernel_shape[0]));
                RawTagHexD('m', static_cast<uint32_t>(depth_multiplier));
                RawTagHexD('i', static_cast<uint32_t>(
                                reinterpret_cast<uintptr_t>(tile_data)));
                RawTagHexD('r', static_cast<uint32_t>(
                                reinterpret_cast<uintptr_t>(w_tile_data)));
                RawTagHexD('b', static_cast<uint32_t>(
                                reinterpret_cast<uintptr_t>(bias_data)));
                RawTagHexD('o', static_cast<uint32_t>(
                                reinterpret_cast<uintptr_t>(o_tile_data)));
                RawNewlineD();

                RawPutcD('['); RawPutcD('D'); RawPutcD('P'); RawPutcD(']');
                RawTagHexD('s', static_cast<uint32_t>(stride_h));
                RawTagHexD('S', static_cast<uint32_t>(stride_w));
                RawTagHexD('p', static_cast<uint32_t>(pad_height));
                RawTagHexD('P', static_cast<uint32_t>(pad_width));
                RawTagHexD('d', static_cast<uint32_t>(dilation_height_factor));
                RawTagHexD('D', static_cast<uint32_t>(dilation_width_factor));
                RawTagHexD('I', static_cast<uint32_t>(input_offset));
                RawTagHexD('O', static_cast<uint32_t>(output_off));
                RawTagHexD('n', static_cast<uint32_t>(output_activation_min));
                RawTagHexD('x', static_cast<uint32_t>(output_activation_max));
                RawNewlineD();
            }

            //reset control
            uint8_t reset=0;
            dma_load_csr((uint32_t)&reset, CSR_BASE+0x30, 1);

            //send information to CSR
            dma_load_csr((uint32_t)&layer_type, CSR_BASE+0x00, 4);
            dma_load_csr((uint32_t)&p_mode, CSR_BASE+0x04, 4);
            dma_load_csr((uint32_t)&tile_shape[1], CSR_BASE+0x08, 4);
            dma_load_csr((uint32_t)&tile_shape[3], CSR_BASE+0x0C, 4);
            dma_load_csr((uint32_t)&o_tile_shape[3], CSR_BASE+0x10, 4);
            dma_load_csr((uint32_t)&o_tile_shape[1], CSR_BASE+0x14, 4);
            dma_load_csr((uint32_t)&stride_h, CSR_BASE+0x18, 4);
            dma_load_csr((uint32_t)&depth_multiplier, CSR_BASE+0x1C, 4);
            dma_load_csr(SPAD_I_BASE, CSR_BASE+0x20, 4);
            dma_load_csr(SPAD_I_BASE+(tile_size)/4, CSR_BASE+0x24, 4);
            dma_load_csr(SPAD_W_BASE, CSR_BASE+0x28, 4);
            dma_load_csr(SPAD_W_BASE+(w_tile_size)/4, CSR_BASE+0x2C, 4);
            dma_load_csr((uint32_t)&output_off, CSR_BASE+0x38, 4);
            dma_load_csr((uint32_t)&input_offset, CSR_BASE+0x3C, 4);
            //dma_load_csr(&output_offset, CSR_BASE+0x38, 4);
            //dma_load_csr(&output_multiplier, CSR_BASE+0x3C, 4);
            

            //weight tiling
            if (layer_type!=0){
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=w_data[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
                        }
                    }
                }
            }
            else{
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=w_data[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
                        }
                    }
                }
            }
            

            //input tiling
            for(int height=0; height<std::min(tile_shape[1], height_count); height++){
                //std::cout<<std::endl;
                for(int width=0; width<std::min(tile_shape[2], width_count); width++){
                    for(int channel=0; channel<std::min(tile_shape[3], channel_count); channel++){
                        //the head-scratching tiling offset
                        int height_off=0;
                        int width_off=0;
                        int channel_off=(real_tile/(tile_shape[1]*tile_shape[2]))*tile_shape[3];
                        if(tile>=height_number){
                            height_off=(((tile)/height_number))*(tile_shape[1]-overlap_h);
                        }
                        if(tile%width_number!=0){
                            width_off=(tile%width_number)*(tile_shape[2]-overlap_w);
                        }
                        if((tile<height_number&&height-padding_up<0)||(tile%width_number==0&&width-padding_left<0)||(height+height_off+padding_down==padded_input_shape[1])||(width+width_off+padding_right==padded_input_shape[2])){
                            tile_data[((batch*tile_shape[1]+height)*tile_shape[2]+width)*tile_shape[3]+channel]=zero_point;
                        }
                        else{
                            tile_data[((height)*tile_shape[2]+width)*tile_shape[3]+channel]=
                            i_data[((height+height_off-padding_up)*input_shape[2]*input_shape[3])+((width+width_off-padding_left)*input_shape[3])+channel+channel_off];
                        }
                        
                    }
                }
            }

            //input edge-handling
            if(channel_count<=tile_shape[3]){
                channel_count=input_shape[3];
            }
            else{
                channel_count=channel_count-tile_shape[3];
            }
            if(width_count<=tile_shape[2]){
                width_count=padded_input_shape[2];
            }
            else{
                if(tile%width_number!=0){
                    width_count=width_count-tile_shape[2]+overlap_w;
                }
                else{
                    width_count=width_count-tile_shape[2];
                }
            }
            if ((tile+1)%height_number==0){
                if(height_count<=tile_shape[1]){
                    height_count=padded_input_shape[1];
                }
                else{
                    if(tile>height_number){
                            height_count=height_count-tile_shape[1]+overlap_h;
                    }
                    else{
                            height_count=height_count-tile_shape[1];
                    }
                }
            }

            //bias and quant tiling
            for(int channel=0; channel<output_shape[3]; channel++){
                // bias_data[channel]=b_data[channel+batch]-input_offset*weightsum;
                bias_data[channel]=b_data[channel+batch];
                qmult_data[channel]=output_mult[channel+batch];
                qshift_data[channel]=output_shift[channel+batch];
            }
            
            //send wtile to wspad
            uint32_t weight_start=(uint32_t)&w_tile_data;
            dma_load_weights(weight_start, SPAD_BASE, w_tile_size);
            
            //send tile to ispad
            uint32_t input_start=(uint32_t)&tile_data;
            dma_load_inputs(input_start, SPAD_BASE, tile_size);

            //send bias to bspad
            int bias_size=4*output_shape[3];
            uint32_t bias_start=(uint32_t)&bias_data;
            dma_load_bias(bias_start, SPAD_BASE, bias_size);

            //send quant mult to qmspad
            int qmult_size=4*output_shape[3];
            uint32_t qmult_start=(uint32_t)&qmult_data;
            dma_load_scale(qmult_start, SPAD_BASE, qmult_size);

            //send quant shift to qsspad
            int qshift_size=4*output_shape[3];
            uint32_t qshift_start=(uint32_t)&qshift_data;
            dma_load_shift(qshift_start, SPAD_BASE, qshift_size);

            //systolic start
            uint8_t start=1;
            dma_load_csr((uint32_t)&start, CSR_BASE+0x30, 1);

            //check if systolic is done
            int tile_done;
            dma_load_csr(CSR_BASE+0x34, (uint32_t)&tile_done, 4);
            while (!tile_done){
                dma_load_csr(CSR_BASE+0x34, (uint32_t)&tile_done, 4);
            }
            
            //register clear
            uint8_t clear=2;
            dma_load_csr((uint32_t)&clear, CSR_BASE+0x30, 1);

            //send ospad to dma
            uint32_t output_start=(uint32_t)&o_tile_data;
            dma_load_inputs(SPAD_O_BASE, output_start, o_tile_size);

            //output untiling
            for(int height=0; height<std::min(o_tile_shape[1], o_height_count); height++){
                for(int width=0; width<std::min(o_tile_shape[2], o_width_count); width++){
                    for(int channel=0; channel<std::min(o_tile_shape[3], o_channel_count); channel++){
                        //the head-scratching tiling offset
                        int o_height_off=0;
                        int o_width_off=0;
                        int o_channel_off=(real_tile/(tile_shape[1]*tile_shape[2]))*w_tile_shape[0];
                        if(tile>=o_height_number){
                            o_height_off=(((tile)/o_height_number))*(o_tile_shape[1]);
                        }
                        if(tile%o_width_number!=0){
                            o_width_off=(tile%o_width_number)*(o_tile_shape[2]);
                        }
                        o_data[((height+o_height_off)*output_shape[2]*output_shape[3])+((width+o_width_off)*output_shape[3])+channel+o_channel_off]=o_tile_data[((batch*o_tile_shape[1]+height)*o_tile_shape[2]+width)*o_tile_shape[3]+channel];
                    }
                }
            }
            
            //output edge handling
            if(o_channel_count<=o_tile_shape[3]){
                o_channel_count=output_shape[3];
            }
            else{
                o_channel_count=o_channel_count-o_tile_shape[3];
            }
            if(o_width_count<=o_tile_shape[2]){
                o_width_count=output_shape[2];
            }
            else{
                o_width_count=o_width_count-o_tile_shape[2];
            }
            if ((tile+1)%o_height_number==0){
                if(o_height_count<=o_tile_shape[1]){
                    o_height_count=output_shape[1];
                }
                else{
                    o_height_count=o_height_count-o_tile_shape[1];
                }
            }
            if(tile==width_number*height_number-1){
                tile=-1;
                o_height_count=output_shape[1];
                o_width_count=output_shape[2];
                o_channel_count=output_shape[3];
            }
            tile++;
        }
    }
}

inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const std::int64_t* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            std::int64_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  // Accumulate with 64 bits accumulator.
                  // We assume maximum of 2^16 accumulations as with the 8-bit
                  // case so actually the value in the accumulator should not
                  // exceed 40 bits
                  acc += static_cast<int64_t>(filter_val) *
                         static_cast<int64_t>(input_val);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[output_channel];
            }
            int32_t scaled_acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[output_channel],
                output_shift[output_channel]);
            scaled_acc = std::max(scaled_acc, output_activation_min);
            scaled_acc = std::min(scaled_acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x,
                               output_channel)] =
                static_cast<int16_t>(scaled_acc);
          }
        }
      }
    }
  }
}

inline void DepthwiseConvHybridPerChannel(
    const DepthwiseParams& params, float* scaling_factors_ptr,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const float* per_channel_scale, int32_t* input_offset) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int bias_depth = bias_shape.FlatSize();
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_depth, output_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  int32_t filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, output_channel)];
                  acc += filter_val * (input_val - input_offset[batch]);
                }
              }
            }
            float acc_float = static_cast<float>(acc);
            acc_float *=
                per_channel_scale[output_channel] * scaling_factors_ptr[batch];
            if (bias_data && output_channel < bias_depth) {
              acc_float += bias_data[output_channel];
            }
            output_data[Offset(output_shape, batch, out_y, out_x,
                               output_channel)] =
                ActivationFunctionWithMinMax(acc_float, output_activation_min,
                                             output_activation_max);
          }
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_DEPTHWISE_CONV_H_
