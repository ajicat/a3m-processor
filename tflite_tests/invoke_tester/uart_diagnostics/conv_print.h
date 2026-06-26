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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

#include <stdint.h>

namespace tflite {
namespace reference_integer_ops {
static inline void RawPutcD(char c) {
  volatile uint32_t* const uart_tx =
      reinterpret_cast<volatile uint32_t*>(0x40600000u + 0x04u);
  volatile uint32_t* const uart_status =
      reinterpret_cast<volatile uint32_t*>(0x40600000u + 0x08u);

  while ((*uart_status) & 0x08u) {
  }

  *uart_tx = static_cast<uint32_t>(static_cast<uint8_t>(c));
}

static inline void RawPutsD(const char* s) {
    while (*s) {
        RawPutcD(*s++);
    }
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

static inline void RawTagHexD(const char* tag, uint32_t v) {
  RawPutsD(tag);
  RawPutHex32D(v);
  RawPutcD(' ');
}

// ============================================================================
// ACCELERATOR INTERFACE HELPERS
// Adapted from main_aaron_x2.cc to interface with the custom systolic array.
// Note: Adjust base addresses and register mappings if your Vivado Address 
// Editor or RTL definitions have changed.
// ============================================================================
namespace accel {
    // Base addresses (from Vivado Address Editor)
    static constexpr uint32_t CSR_BASE             = 0x00001000;
    static constexpr uint32_t DMA_BASE             = 0x00010000;
    static constexpr uint32_t OUT_SPAD_BASE        = 0x20000000;
    static constexpr uint32_t SPAD_WRITE_ADAPTER   = 0xC0000000;

    // CSR register offsets
    static constexpr uint32_t CSR_CONV_MODE   = 0x00;
    static constexpr uint32_t CSR_P_MODE      = 0x04;
    static constexpr uint32_t CSR_I_SIZE      = 0x08;
    static constexpr uint32_t CSR_I_C_SIZE    = 0x0C;
    static constexpr uint32_t CSR_O_C_SIZE    = 0x10;
    static constexpr uint32_t CSR_O_SIZE      = 0x14;
    static constexpr uint32_t CSR_STRIDE      = 0x18;
    static constexpr uint32_t CSR_DEPTH_MULT  = 0x1C;
    static constexpr uint32_t CSR_I_START     = 0x20;
    static constexpr uint32_t CSR_I_END       = 0x24;
    static constexpr uint32_t CSR_W_START     = 0x28;
    static constexpr uint32_t CSR_W_END       = 0x2C;
    static constexpr uint32_t CSR_CTRL        = 0x30;
    static constexpr uint32_t CSR_STATUS      = 0x34;
    static constexpr uint32_t CSR_ZERO_POINT  = 0x38;
    static constexpr uint32_t CSR_INPUT_OFFSET= 0x3C;

    // DMA SPAD_SEL encoding
    static constexpr uint32_t SPAD_WEIGHTS  = 0;
    static constexpr uint32_t SPAD_IFMAPS   = 1;
    static constexpr uint32_t SPAD_BIAS     = 2;
    static constexpr uint32_t SPAD_SCALE    = 3;
    static constexpr uint32_t SPAD_SHIFT    = 4;
}

static inline void accel_wr32(uint32_t a, uint32_t v) { *(volatile uint32_t*)a = v; }
static inline uint32_t accel_rd32(uint32_t a) { return *(volatile uint32_t*)a; }

static inline uint32_t accel_bswap32(uint32_t v) {
    return ((v & 0x000000FFu) << 24) |
           ((v & 0x0000FF00u) << 8)  |
           ((v & 0x00FF0000u) >> 8)  |
           ((v & 0xFF000000u) >> 24);
}

static inline void accel_dma_write32_addr(uint32_t addr, uint32_t value) {
    accel_wr32(addr, accel_bswap32(value));
}
static inline uint32_t accel_dma_read32_addr(uint32_t addr) {
    return accel_bswap32(accel_rd32(addr));
}
static inline void accel_csr_write32(uint32_t off, uint32_t value) {
    accel_wr32(accel::CSR_BASE + off, accel_bswap32(value));
}
static inline uint32_t accel_csr_read32(uint32_t off) {
    return accel_bswap32(accel_rd32(accel::CSR_BASE + off));
}

static inline void accel_small_delay(void) {
    for (volatile uint32_t i = 0; i < 2000u; i++) {}
}

static inline int accel_dma_wait(int ch) {
    uint32_t timeout = 10000000u;
    uint32_t status = 0;
    uint32_t stat_reg = accel::DMA_BASE + ch*0x20 + 0x10;
    uint32_t ctrl_reg = accel::DMA_BASE + ch*0x20 + 0x0C;
    while (timeout--) {
        status = accel_dma_read32_addr(stat_reg);
        if (status & 0x2u) {
            accel_dma_write32_addr(stat_reg, 0x2u);
            accel_small_delay();
            accel_dma_write32_addr(ctrl_reg, 0x0u);
            accel_small_delay();

            RawPutsD("[DW]");
            RawTagHexD("ch=", static_cast<uint32_t>(ch));
            RawTagHexD("st=", status);
            RawNewlineD();

            return 1;
        }
    }

    RawPutsD("[DT]");
    RawTagHexD("ch=", static_cast<uint32_t>(ch));
    RawTagHexD("st=", status);
    RawNewlineD();

    return 0; // Timeout
}

static inline int accel_dma_start(int ch, uint32_t src, uint32_t dst, uint32_t len, uint32_t spad) {
    RawPutsD("[DS]");
    RawTagHexD("ch=", static_cast<uint32_t>(ch));
    RawTagHexD("src=", src);
    RawTagHexD("dst=", dst);
    RawTagHexD("len=", len);
    RawTagHexD("sp=", spad);
    RawNewlineD();

    uint32_t ch_base = accel::DMA_BASE + ch*0x20;
    accel_dma_write32_addr(ch_base + 0x0C, 0x0u);
    accel_small_delay();
    accel_dma_write32_addr(ch_base + 0x10, 0x2u);
    accel_small_delay();
    accel_dma_write32_addr(ch_base + 0x14, spad);
    accel_dma_write32_addr(ch_base + 0x00, src);
    accel_dma_write32_addr(ch_base + 0x04, dst);
    accel_dma_write32_addr(ch_base + 0x08, len);
    accel_dma_write32_addr(ch_base + 0x0C, 0x1u);
    accel_small_delay();
    return accel_dma_wait(ch);
}

static inline uint32_t ptr_to_dma_addr(const void* ptr) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}



// Fixed-point per-channel-quantization convolution reference kernel.
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
//   // Get parameters.
//   const int32_t input_offset = params.input_offset;  // r = s(q - Z)
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int32_t output_offset = params.output_offset;

//   // Set min and max value of the output.
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Consistency check.
//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   // Check dimensions of the tensors.
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int filter_input_depth = filter_shape.Dims(3);
//   const int groups = input_depth / filter_input_depth;
//   TFLITE_DCHECK_NE(groups, 0);
//   TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
//   const int filters_per_group = output_depth / groups;
//   TFLITE_DCHECK_NE(filters_per_group, 0);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       const int in_y_origin = (out_y * stride_height) - pad_height;
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         const int in_x_origin = (out_x * stride_width) - pad_width;
//         for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
//           auto group = out_channel / filters_per_group;
//           int32_t acc = 0;
//           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//             const int in_y = in_y_origin + dilation_height_factor * filter_y;
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               const int in_x = in_x_origin + dilation_width_factor * filter_x;

//               // Zero padding by omitting the areas outside the image.
//               const bool is_point_inside_image =
//                   (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
//                   (in_y < input_height);

//               if (!is_point_inside_image) {
//                 continue;
//               }

//               for (int in_channel = 0; in_channel < filter_input_depth;
//                    ++in_channel) {
//                 int32_t input_val =
//                     input_data[Offset(input_shape, batch, in_y, in_x,
//                                       in_channel + group * filter_input_depth)];
//                 int32_t filter_val = filter_data[Offset(
//                     filter_shape, out_channel, filter_y, filter_x, in_channel)];
//                 // Accumulate with 32 bits accumulator.
//                 // In the nudging process during model quantization, we force
//                 // real value of 0.0 be represented by a quantized value. This
//                 // guarantees that the input_offset is a int8_t, even though
//                 // it is represented using int32_t. int32_t += int8_t *
//                 // (int8_t - int8_t) so the highest value we can get from each
//                 // accumulation is [-127, 127] * ([-128, 127] -
//                 // [-128, 127]), which is [-32512, 32512]. log2(32512)
//                 // = 14.98, which means we can accumulate at least 2^16
//                 // multiplications without overflow. The accumulator is
//                 // applied to a filter so the accumulation logic will hold as
//                 // long as the filter size (filter_y * filter_x * in_channel)
//                 // does not exceed 2^16, which is the case in all the models
//                 // we have seen so far.
//                 // TODO(b/174275578): Add a check to make sure the
//                 // accumulator depth is smaller than 2^16.
//                 acc += filter_val * (input_val + input_offset);
//               }
//             }
//           }

//           if (bias_data) {
//             acc += bias_data[out_channel];
//           }
//           acc = MultiplyByQuantizedMultiplier(
//               acc, output_multiplier[out_channel], output_shift[out_channel]);
//           acc += output_offset;
//           acc = std::max(acc, output_activation_min);
//           acc = std::min(acc, output_activation_max);
//           output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
//               static_cast<int8_t>(acc);
//         }
//       }
//     }
//   }
// }

inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_mult,
    const int32_t* output_shift, const RuntimeShape& i_shape,
    const int8_t* i_data, const RuntimeShape& w_shape,
    const int8_t* w_data, const RuntimeShape& b_shape,
    const int32_t* b_data, const RuntimeShape& o_shape,
    int8_t* o_data) {
    
    RawPutsD("[CV0]");
    RawNewlineD();

    const int depth_multiplier=1;
    const int stride_w = params.stride_width;
    const int stride_h = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int32_t input_offset = -(params.input_offset);
    const int32_t output_offset = params.output_offset; 
    const int32_t* output_qshift = output_shift;
    const int32_t* output_multiplier = output_mult;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    const int32_t zero_point=input_offset;
    const int layer_type = 0; //Pointwise=0, Depthwise=1
    const int input_shape[] = {i_shape.Dims(0), i_shape.Dims(1), i_shape.Dims(2), i_shape.Dims(3)};
    const int kernel_shape[] = {w_shape.Dims(0), w_shape.Dims(1), w_shape.Dims(2), w_shape.Dims(3)};
    const int output_shape[] = {o_shape.Dims(0), o_shape.Dims(1), o_shape.Dims(2), o_shape.Dims(3)};
    
    RawPutsD("[CVP]");
    RawTagHexD("ioff=", static_cast<uint32_t>(input_offset));
    RawTagHexD("ooff=", static_cast<uint32_t>(output_offset));
    RawTagHexD("amin=", static_cast<uint32_t>(output_activation_min));
    RawTagHexD("amax=", static_cast<uint32_t>(output_activation_max));
    RawNewlineD();

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

    RawPutsD("[PAD]");
    RawTagHexD("upad", static_cast<uint32_t>(padding_up));
    RawTagHexD("dpad", static_cast<uint32_t>(padding_down));
    RawTagHexD("lpad", static_cast<uint32_t>(padding_left));
    RawTagHexD("rpad", static_cast<uint32_t>(padding_right));
    RawNewlineD();

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
    RawPutsD("[INP]");
    RawTagHexD("b", static_cast<uint32_t>(tile_shape[0]));
    RawTagHexD("h", static_cast<uint32_t>(tile_shape[1]));
    RawTagHexD("w", static_cast<uint32_t>(tile_shape[2]));
    RawTagHexD("c", static_cast<uint32_t>(tile_shape[3]));
    RawNewlineD();
    //calculate current tile memory requirements
    int mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
    int mem_w=kernel_shape[1]*kernel_shape[2]*kernel_shape[3];
    int mem_out=o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
    /*
    if (layer_type!=0){ //if depthwise
        while (tile_shape[3]>1&&(plm_w<mem_w||plm_out<mem_out)){
            //half the channel size
            tile_shape[3]=(tile_shape[3]%2==0)? (tile_shape[3]/2):(tile_shape[3]/2+1);
            mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
            mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
            mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
    }
    else{ //if pointwise
    */
        //output channels=kernel batches
        //set the batch size to 1
        w_tile_shape[0]=1;
        o_tile_shape[3]=w_tile_shape[0];
        while (tile_shape[1]>1&&tile_shape[2]>1&&(plm_w<mem_w||plm_out<mem_out)){
                //decrement the tile height and width (to stay square)
                tile_shape[1]=tile_shape[1]-1;
                tile_shape[2]=tile_shape[2]-1;
                
                mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
                mem_w=kernel_shape[1]*kernel_shape[2]*kernel_shape[3];
                mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
    //}
    RawPutsD("[OUT]");
    RawTagHexD("b", static_cast<uint32_t>(o_tile_shape[0]));
    RawTagHexD("h", static_cast<uint32_t>(o_tile_shape[1]));
    RawTagHexD("w", static_cast<uint32_t>(o_tile_shape[2]));
    RawTagHexD("c", static_cast<uint32_t>(o_tile_shape[3]));
    RawNewlineD();
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

    RawPutsD("[KER]");
    RawTagHexD("b", static_cast<uint32_t>(w_tile_shape[0]));
    RawTagHexD("h", static_cast<uint32_t>(w_tile_shape[1]));
    RawTagHexD("w", static_cast<uint32_t>(w_tile_shape[2]));
    RawTagHexD("c", static_cast<uint32_t>(w_tile_shape[3]));
    RawNewlineD();

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

    //int kernel_size=kernel_shape[1]*kernel_shape[2];
    //int32_t weightsum=0;

    RawPutsD("[OUT]");
    RawTagHexD("b", static_cast<uint32_t>(o_tile_shape[0]));
    RawTagHexD("h", static_cast<uint32_t>(o_tile_shape[1]));
    RawTagHexD("w", static_cast<uint32_t>(o_tile_shape[2]));
    RawTagHexD("c", static_cast<uint32_t>(o_tile_shape[3]));
    RawNewlineD();

    for(int batch=0; batch<kernel_shape[0]; batch++){
        int tile=0;
        int height_count=padded_input_shape[1];
        int width_count=padded_input_shape[2];
        int channel_count=padded_input_shape[3];
    
        for(int real_tile=0; real_tile<tile_number; real_tile++){
            /*
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
            */
            const int batches = output_shape[0];
            const int input_depth = input_shape[3];
            const int output_depth = output_shape[3];
            const int input_height = input_shape[1];
            const int input_width = input_shape[2];
            const int filter_height = kernel_shape[1];
            const int filter_width = kernel_shape[2];
            const int filter_input_depth = kernel_shape[3];
            const int groups = input_depth / filter_input_depth;
            const int filters_per_group = output_depth / groups;
            const int output_height = output_shape[1];
            const int output_width = output_shape[2];

            RawPutsD("[CVS]");
            RawTagHexD("b=", static_cast<uint32_t>(batches));
            RawTagHexD("ih=", static_cast<uint32_t>(input_height));
            RawTagHexD("iw=", static_cast<uint32_t>(input_width));
            RawTagHexD("ic=", static_cast<uint32_t>(input_depth));
            RawNewlineD();

            //weight tiling
            /*if (layer_type!=0){
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=w_data[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
                        }
                    }
                }
            }*/
            //else{
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=w_data[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
                        }
                    }
                }
            //}
            
            RawPutsD("[CVK]");
            RawTagHexD("fh=", static_cast<uint32_t>(filter_height));
            RawTagHexD("fw=", static_cast<uint32_t>(filter_width));
            RawTagHexD("oh=", static_cast<uint32_t>(output_height));
            RawTagHexD("ow=", static_cast<uint32_t>(output_width));
            RawTagHexD("oc=", static_cast<uint32_t>(output_depth));
            RawNewlineD();

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
            RawPutsD("[CVK]");
            RawTagHexD("sw=", static_cast<uint32_t>(stride_w));
            RawTagHexD("sh=", static_cast<uint32_t>(stride_h));
            RawTagHexD("pw=", static_cast<uint32_t>(pad_width));
            RawTagHexD("ph=", static_cast<uint32_t>(pad_height));
            RawNewlineD();
            //input edge-handling
            if(channel_count<=tile_shape[3]){
                channel_count=input_shape[3];
            }
            else{
                channel_count=channel_count-tile_shape[3];
            }
            RawPutsD("[CCO]");
            RawTagHexD("c=", static_cast<uint32_t>(channel_count));
            RawNewlineD();
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
            
            RawPutsD("[WCO]");
            RawTagHexD("w=", static_cast<uint32_t>(width_count));
            RawNewlineD();
            
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
            RawPutsD("[HCO]");
            RawTagHexD("h=", static_cast<uint32_t>(height_count));
            RawNewlineD();
            RawPutsD("[IWO]");
            RawTagHexD("i", static_cast<uint32_t>(tile_size));
            RawTagHexD("w", static_cast<uint32_t>(w_tile_size));
            RawTagHexD("o", static_cast<uint32_t>(o_tile_size));
            RawNewlineD();
            //bias and quant tiling
            for(int channel=0; channel<o_tile_shape[3]; channel++){
                //for(int element=0; element<kernel_size; element++){
                //    weightsum = weightsum+w_data[element*w_tile_shape[3]+channel];
                //}
                //bias_data[channel]=b_data[channel+batch]-input_offset*weightsum;
                bias_data[channel]=b_data[channel+batch];
                qmult_data[channel]=output_mult[channel+batch];
                qshift_data[channel]=output_shift[channel+batch];
            }

            RawPutsD("[CVT]");
            RawTagHexD("in=", ptr_to_dma_addr(tile_data));
            RawTagHexD("fil=", ptr_to_dma_addr(w_tile_data));
            RawTagHexD("bias=", ptr_to_dma_addr(bias_data));
            RawTagHexD("mul=", ptr_to_dma_addr(qmult_data));
            RawTagHexD("sh=", ptr_to_dma_addr(qshift_data));
            RawTagHexD("out=", ptr_to_dma_addr(o_tile_data));
            RawNewlineD();

            // =========================================================================
            // ACCELERATOR OFFLOAD START
            // =========================================================================
            
            RawPutsD("[CVA]");
            RawNewlineD();

            // 1. Clear CSR
            RawPutsD("[CVC]");
            RawNewlineD();

            accel_csr_write32(accel::CSR_CTRL, 0x2);
            accel_small_delay();
            accel_csr_write32(accel::CSR_CTRL, 0x0);
            accel_small_delay();

            RawPutsD("[CVC1]");
            RawNewlineD();

            //send wtile to wspad
            //uint32_t weight_start=(uint32_t)&w_tile_data;
            //dma_load_weights(weight_start, SPAD_BASE, w_tile_size);
            // 2. DMA Load Weights (Filter)
            // Note: With 32b SPAD words, the DMA controller handles packing 32-bit AXI beats 
            // into the SPAD. We just pass the total byte length.
            RawPutsD("[DWF]");
            RawTagHexD("bytes=", w_tile_size);
            RawNewlineD();

            if (!accel_dma_start(0, ptr_to_dma_addr(w_tile_data), accel::SPAD_WRITE_ADAPTER, w_tile_size, accel::SPAD_WEIGHTS)) {
                RawPutsD("[EWF]");
                RawNewlineD();
                return;
            }

            //send tile to ispad
            //uint32_t input_start=(uint32_t)&tile_data;
            //dma_load_inputs(input_start, SPAD_BASE, tile_size);
            // 3. DMA Load Ifmaps (Input)
            RawPutsD("[DIF]");
            RawTagHexD("bytes=", tile_size);
            RawNewlineD();

            if (!accel_dma_start(0, ptr_to_dma_addr(tile_data), accel::SPAD_WRITE_ADAPTER, tile_size, accel::SPAD_IFMAPS)) {
                RawPutsD("[EIF]");
                RawNewlineD();
                return;
            }

            //send bias to bspad
            //int bias_size=4*output_shape[3];
            //uint32_t bias_start=(uint32_t)&bias_data;
            //dma_load_bias(bias_start, SPAD_BASE, bias_size);
            // 4. DMA Load Biases
            if (bias_data) {
                uint32_t bias_size = o_tile_shape[3] * sizeof(int32_t);

                RawPutsD("[DBI]");
                RawTagHexD("bytes=", bias_size);
                RawNewlineD();

                if (!accel_dma_start(0, ptr_to_dma_addr(bias_data), accel::SPAD_WRITE_ADAPTER, bias_size, accel::SPAD_BIAS)) {
                    RawPutsD("[EBI]");
                    RawNewlineD();
                    return;
                }
            } else {
                RawPutsD("[NBI]");
                RawNewlineD();
            }

            //send quant mult to qmspad
            //int qmult_size=4*output_shape[3];
            //uint32_t qmult_start=(uint32_t)&qmult_data;
            //dma_load_scale(qmult_start, SPAD_BASE, qmult_size);
            // 5. DMA Load Scales (output_multiplier)
            uint32_t qmult_size = o_tile_shape[3] * sizeof(int32_t);

            RawPutsD("[DSC]");
            RawTagHexD("bytes=", qmult_size);
            RawNewlineD();

            if (!accel_dma_start(0, ptr_to_dma_addr(qmult_data), accel::SPAD_WRITE_ADAPTER, qmult_size, accel::SPAD_SCALE)) {
                RawPutsD("[ESC]");
                RawNewlineD();
                return;
            }

            //send quant shift to qsspad
            //int qshift_size=4*output_shape[3];
            //uint32_t qshift_start=(uint32_t)&qshift_data;
            //dma_load_shift(qshift_start, SPAD_BASE, qshift_size);
            // 6. DMA Load Shifts (output_shift)
            uint32_t qshift_size = o_tile_shape[3] * sizeof(int32_t);

            RawPutsD("[DSH]");
            RawTagHexD("bytes=", qshift_size);
            RawNewlineD();

            if (!accel_dma_start(0, ptr_to_dma_addr(qshift_data), accel::SPAD_WRITE_ADAPTER, qshift_size, accel::SPAD_SHIFT)) {
                RawPutsD("[ESH]");
                RawNewlineD();
                return;
            }

            //systolic start
            //uint8_t start=1;
            //dma_load_csr((uint32_t)&start, CSR_BASE+0x30, 1);

            //check if systolic is done
            //int tile_done;
            //dma_load_csr(CSR_BASE+0x34, (uint32_t)&tile_done, 4);
            //while (!tile_done){
                //dma_load_csr(CSR_BASE+0x34, (uint32_t)&tile_done, 4);
            //}
            
            //register clear
            //uint8_t clear=2;
            //dma_load_csr((uint32_t)&clear, CSR_BASE+0x30, 1);
            // 7. Configure CSR
            // Assuming 0 = Pointwise.
            RawPutsD("[CSR0]");
            RawNewlineD();

            accel_csr_write32(accel::CSR_CONV_MODE, 0);
            accel_csr_write32(accel::CSR_P_MODE, 0); 
            
            // Spatial dimensions (Adjust if your RTL expects total spatial size vs width)
            accel_csr_write32(accel::CSR_I_SIZE, input_width); 
            accel_csr_write32(accel::CSR_I_C_SIZE, input_depth);
            accel_csr_write32(accel::CSR_O_C_SIZE, output_depth);
            accel_csr_write32(accel::CSR_O_SIZE, output_width);
            accel_csr_write32(accel::CSR_STRIDE, stride_w);
            
            // Depth multiplier
            accel_csr_write32(accel::CSR_DEPTH_MULT, 1); // Set to 1 for pointwise conv.
            
            // Loop bounds / Address offsets
            int i_end = (tile_size-1) >> 2;      // Assuming 32-bit words for addressing
            int w_end = (w_tile_size-1) >> 2;     // Inclusive end address

            RawPutsD("[CSR1]");
            RawTagHexD("iend=", static_cast<uint32_t>(i_end));
            RawTagHexD("wend=", static_cast<uint32_t>(w_end));
            RawNewlineD();

            accel_csr_write32(accel::CSR_I_START, 0);
            accel_csr_write32(accel::CSR_I_END, i_end);
            accel_csr_write32(accel::CSR_W_START, 0);
            accel_csr_write32(accel::CSR_W_END, w_end);
            
            // Zero points and offsets
            accel_csr_write32(accel::CSR_ZERO_POINT, output_offset);    
            accel_csr_write32(accel::CSR_INPUT_OFFSET, -input_offset);  // RTL subtracts this from input, so negate the offset here.

            RawPutsD("[CSR2]");
            RawNewlineD();

            // 8. Start Systolic Array
            RawPutsD("[RUN]");
            RawNewlineD();

            accel_csr_write32(accel::CSR_CTRL, 0x1);

            // 9. Poll for done
            uint32_t status = 0;
            uint32_t accel_timeout = 10000000u;
            do {
                status = accel_csr_read32(accel::CSR_STATUS);
                if (--accel_timeout == 0u) {
                    RawPutsD("[ETO]");
                    RawTagHexD("st=", status);
                    RawNewlineD();
                    return;
                }
            } while ((status & 0x1u) == 0);

            RawPutsD("[DON]");
            RawTagHexD("st=", status);
            RawNewlineD();

            // 10. Stop accelerator
            accel_csr_write32(accel::CSR_CTRL, 0x0);
            accel_small_delay();

            RawPutsD("[STP]");
            RawNewlineD();

            //send ospad to dma
            //uint32_t output_start=(uint32_t)&o_tile_data;
            //dma_load_inputs(SPAD_O_BASE, output_start, o_tile_size);
            // 11. DMA Read Output
            RawPutsD("[DOUT]");
            RawTagHexD("bytes=", o_tile_size);
            RawNewlineD();

            if (!accel_dma_start(0, accel::OUT_SPAD_BASE, ptr_to_dma_addr(o_tile_data), o_tile_size, 0)) {
                RawPutsD("[EOUT]");
                RawNewlineD();
                return;
            }

            RawPutsD("[CV1]");
            RawNewlineD();

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
            
            //The Almighty Edge-Handler
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


// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
