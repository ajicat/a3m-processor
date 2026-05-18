#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

/*
#define DMA_BASE 0x40000000UL //defined in vivado

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

#define CSR_BASE       0x4000_1000 

#define SPAD_W_BASE    0xC000_0000 //weight spad
#define SPAD_I_BASE    0xC000_2000 //input spad
#define SPAD_B_BASE    0xC000_4000 //bias spad
#define SPAD_M_BASE    0xC000_6000 //mult spad
#define SPAD_S_BASE    0xC000_8000 //shift spad
#define SPAD_O_BASE    0x2000_0000 //output spad
*/
namespace tflite {
namespace reference_ops {
/*
    #define DMA_REG(ch, reg) \
        (*((volatile uint32_t *)(DMA_BASE + CH_OFFSET(ch) + (reg))))

    #define DMA_STATUS(ch)  DMA_REG(ch, REG_STATUS)
    #define DONE_BIT        (1 << 1)

    // Please follow the order of SRC, DST, LEN, SPAD SEL, and last should be CTRL
    void dma_load_weights(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
        DMA_REG(0, REG_SRC)      = dram_src;
        DMA_REG(0, REG_DST)      = spad_dst;
        DMA_REG(0, REG_LEN)      = len;
        DMA_REG(0, REG_SPAD_SEL) = SPAD_WEIGHTS;  // 0b000
        DMA_REG(0, REG_CTRL)     = 0x1;           // START — must be last
        while (!(DMA_STATUS(0) & DONE_BIT));
        DMA_STATUS(0) = DONE_BIT;                 // W1C clear
    }

    void dma_load_inputs(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
        DMA_REG(1, REG_SRC)      = dram_src;
        DMA_REG(1, REG_DST)      = spad_dst;
        DMA_REG(1, REG_LEN)      = len;
        DMA_REG(1, REG_SPAD_SEL) = SPAD_IFMAPS;   // 0b001
        DMA_REG(1, REG_CTRL)     = 0x1;
        while (!(DMA_STATUS(1) & DONE_BIT));
        DMA_STATUS(1) = DONE_BIT;
    }

    void dma_load_bias(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
        DMA_REG(2, REG_SRC)      = dram_src;
        DMA_REG(2, REG_DST)      = spad_dst;
        DMA_REG(2, REG_LEN)      = len;
        DMA_REG(2, REG_SPAD_SEL) = SPAD_BIAS;     // 0b010
        DMA_REG(2, REG_CTRL)     = 0x1;
        while (!(DMA_STATUS(2) & DONE_BIT));
        DMA_STATUS(2) = DONE_BIT;
    }

    void dma_load_scale(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
        DMA_REG(3, REG_SRC)      = dram_src;
        DMA_REG(3, REG_DST)      = spad_dst;
        DMA_REG(3, REG_LEN)      = len;
        DMA_REG(3, REG_SPAD_SEL) = SPAD_SCALE;  // 0b011
        DMA_REG(3, REG_CTRL)     = 0x1;           
        while (!(DMA_STATUS(3) & DONE_BIT));
        DMA_STATUS(3) = DONE_BIT;                 
    }

    void dma_load_shift(uint32_t dram_src, uint32_t spad_dst, uint32_t len) {
        DMA_REG(0, REG_SRC)      = dram_src;
        DMA_REG(0, REG_DST)      = spad_dst;
        DMA_REG(0, REG_LEN)      = len;
        DMA_REG(0, REG_SPAD_SEL) = SPAD_SHIFT;   // 0b100
        DMA_REG(0, REG_CTRL)     = 0x1;
        while (!(DMA_STATUS(0) & DONE_BIT));
        DMA_STATUS(0) = DONE_BIT;
    }

    void dma_load_csr(uint32_t dram_src, uint32_t csr_dst, uint32_t len) {
        DMA_REG(1, REG_SRC)      = dram_src;
        DMA_REG(1, REG_DST)      = csr_dst;
        DMA_REG(1, REG_LEN)      = len;
        DMA_REG(1, REG_SPAD_SEL) = 0b000;
        DMA_REG(1, REG_CTRL)     = 0x1;
        while (!(DMA_STATUS(1) & DONE_BIT));
        DMA_STATUS(1) = DONE_BIT;
    }
*/

inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_mult,
    const int32_t* output_shift, const RuntimeShape& i_shape,
    const int8_t* i_data, const RuntimeShape& w_shape,
    const int8_t* w_data, const RuntimeShape& b_shape,
    const int32_t* b_data, const RuntimeShape& o_shape,
    int8_t* o_data) {

    const int stride_w = params.stride_width;
    const int stride_h = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int32_t input_offset = params.input_offset;
    const int32_t output_off = params.output_offset; 
    const int32_t output_offset = &output_shift;
    const int32_t output_multiplier = &output_mult;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    const int32_t zero_point=-input_offset;
    const int layer_type = 0; //Pointwise=0, Depthwise=1
    const int input_shape[] = {i_shape.Dims(0), i_shape.Dims(1), i_shape.Dims(2), i_shape.Dims(3)};
    const int kernel_shape[] = {w_shape.Dims(0), w_shape.Dims(1), w_shape.Dims(2), w_shape.Dims(3)};
    const int output_shape[] = {o_shape.Dims(0), o_shape.Dims(1), o_shape.Dims(2), o_shape.Dims(3)};

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

    //SPAD Size (precalculated)
    int plm_in=8192;
    int plm_w=8192;
    int plm_out=8192;

    //Unknowns
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
    int mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
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
                mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
                mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
    //}

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
    uint32_t o_tile_data[o_tile_size];

    int w_tile_size=w_tile_shape[0]*kernel_shape[1]*kernel_shape[2]*w_tile_shape[3];
    int8_t w_tile_data[w_tile_size];
    
    int32_t bias_data[o_tile_shape[3]];
    int8_t qmult_data[o_tile_shape[3]];
    int8_t qshift_data[o_tile_shape[3]]; 

    int o_height_count=output_shape[1];
    int o_width_count=output_shape[2];
    int o_channel_count=output_shape[3];

    for(int batch=0; batch<kernel_shape[0]; batch++){
        int tile=0;
        int height_count=padded_input_shape[1];
        int width_count=padded_input_shape[2];
        int channel_count=padded_input_shape[3];
    
        for(int real_tile=0; real_tile<tile_number; real_tile++){
            /*
            //send information to CSR
            dma_load_csr(&layer_type, CSR_BASE+0x00, 4);
            dma_load_csr(&tile_shape[1], CSR_BASE+0x08, 4);
            dma_load_csr(&tile_shape[3], CSR_BASE+0x0C, 4);
            dma_load_csr(&o_tile_shape[3], CSR_BASE+0x10, 4);
            dma_load_csr(&o_tile_shape[1], CSR_BASE+0x14, 4);
            dma_load_csr(&stride_h, CSR_BASE+0x18, 4);
            //dma_load_csr(&depth_multiplier, CSR_BASE+0x1C, 4);
            //pointwise doesn't have depth multiplier?
            dma_load_csr(SPAD_I_BASE, CSR_BASE+0x20, 4);
            dma_load_csr(SPAD_I_BASE+(tile_size), CSR_BASE+0x24, 4);
            dma_load_csr(SPAD_W_BASE, CSR_BASE+0x28, 4);
            dma_load_csr(SPAD_W_BASE+(w_tile_size), CSR_BASE+0x2C, 4);
            dma_load_csr(&output_offset, CSR_BASE+0x38, 4);
            dma_load_csr(&output_multiplier, CSR_BASE+0x3C, 4);
            */

            //weight tiling
            if (layer_type!=0){
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=w_data[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
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
                        int pixel;
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
            
            // bias and quant tiling
            for(int channel=0; channel<o_tile_shape[3]; channel++){
                bias_data[channel]=b_data[real_tile/(tile_shape[1]*tile_shape[2])+channel];
                qmult_data[channel]=output_mult[real_tile/(tile_shape[1]*tile_shape[2])+channel];
                qshift_data[channel]=output_shift[real_tile/(tile_shape[1]*tile_shape[2])+channel];
            }

            /*
            //send wtile to wspad
            uint32_t weight_start=&w_tile_data;
            dma_load_weights(weight_start, SPAD_W_BASE, w_tile_size);
            
            //send tile to ispad
            uint32_t input_start=&tile_data;
            dma_load_inputs(input_start, SPAD_I_BASE, tile_size);

            //send bias to bspad
            int bias_size=4*output_shape[3];
            uint32_t bias_start=&bias_data;
            dma_load_bias(bias_start, SPAD_B_BASE, bias_size);

            //send quant mult to qmspad
            int qmult_size=output_shape[3];
            uint32_t qmult_start=&qmult_data;
            dma_load_scale(qmult_start, SPAD_M_BASE, qmult_size);

            //send quant shift to qsspad
            int qshift_size=output_shape[3];
            uint32_t qshift_start=&qshift_data;
            dma_load_shift(qshift_start, SPAD_S_BASE, qshift_size);

            //systolic start
            uint8_t start=1;
            dma_load_csr(&start, CSR_BASE+0x30, 1);

            //check if systolic is done
            //int tile_done;
            //while (!tile_done){
            //    dma_load_csr(CSR_BASE+0x34, &tile_done, 4);
            //}
            */
            //SURROGATE ACCELERATOR
            const int batches = o_tile_shape[0];
            const int input_depth = tile_shape[3];
            const int output_depth = o_tile_shape[3];
            const int input_height = tile_shape[1];
            const int input_width = tile_shape[2];
            const int filter_height = w_tile_shape[1];
            const int filter_width = w_tile_shape[2];
            const int filter_input_depth = w_tile_shape[3];
            const int groups = input_depth / filter_input_depth;
            const int filters_per_group = output_depth / groups;
            const int output_height = o_tile_shape[1];
            const int output_width = o_tile_shape[2];
            for (int batch = 0; batch < batches; ++batch) {
                for (int out_y = 0; out_y < output_height; ++out_y) {
                const int in_y_origin = (out_y * stride_h);
                for (int out_x = 0; out_x < output_width; ++out_x) {
                    const int in_x_origin = (out_x * stride_w);
                    for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    auto group = out_channel / filters_per_group;
                    int32_t acc = 0;
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
                        //((i0 * shape.Dims(1) + i1) * shape.Dims(2) + i2) * shape.Dims(3) + i3
                        for (int in_channel = 0; in_channel < filter_input_depth;
                            ++in_channel) {
                            int32_t input_val =
                                tile_data[((batch * input_height + in_y) * input_width + in_x) * input_depth + in_channel + group * filter_input_depth];
                            int32_t filter_val = w_tile_data[((out_channel * filter_height + filter_y) * filter_width + filter_x) * filter_input_depth + in_channel];
                            acc += filter_val * (input_val + input_offset);
                        }
                        }
                    }

                    if (bias_data) {
                        acc += bias_data[out_channel];
                    }
                    acc = MultiplyByQuantizedMultiplier(
                        acc, qmult_data[out_channel], qshift_data[out_channel]);
                    acc += output_off;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    o_tile_data[((batch * output_height + out_y) * output_width + out_x) * output_height + out_channel] =
                        static_cast<int8_t>(acc);
                    }
                }
                }
            }
            //send ospad to dma
            //uint32_t output_start=&o_tile_data;
            //dma_load_inputs(SPAD_O_BASE, output_start, o_tile_size);

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

}
}
