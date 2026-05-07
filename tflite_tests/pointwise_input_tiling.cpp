//pointwise conv
//TILING PALANG PALA di ko pa na-combine ang main func
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sample.h> //Contains sample image
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
//  #define SYSTOLIC_BASE 0x40000000
namespace tflite {
namespace reference_ops {
    //eto yung inputs ng Real conv function so kinopya ko lang
inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8_t* input_data, const RuntimeShape& filter_shape,
                 const uint8_t* filter_data, const RuntimeShape& bias_shape,
                 const int32_t* bias_data, const RuntimeShape& output_shape,
                 uint8_t* output_data, const RuntimeShape& im2col_shape,
                 uint8_t* im2col_data, void* cpu_backend_context) {
    const int stride_w = params.stride_width;
    const int stride_h = params.stride_height;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    /*Padding
    int padding_up=0;
    int padding_left=0;
    int padding_down=0;
    int padding_right=0;
    if (pad_width%2==0){
        padding_up=((output_shape[1]*stride_h)+overlap_h-input_shape[1])/2;
        padding_down=((output_shape[1]*stride_h)+overlap_h-input_shape[1])/2;
    }
    else {
        padding_up=((output_shape[1]*stride_h)+overlap_h-input_shape[1])/2+1;
        padding_down=((output_shape[1]*stride_h)+overlap_h-input_shape[1])/2;
    }
    if (pad_height)%2==0){
        padding_left=((output_shape[2]*stride_w)+overlap_w-input_shape[2])/2;
        padding_right=((output_shape[2]*stride_w)+overlap_w-input_shape[2])/2;
    }
    else {
        padding_left=((output_shape[2]*stride_w)+overlap_w-input_shape[2])/2+1;
        padding_right=((output_shape[2]*stride_w)+overlap_w-input_shape[2])/2;
    }
    }
    */
    const int layer_type = 1; //Depthwise=0, Pointwise=1
    const int input_shape[] = {input_shape.Dims(0), input_shape.Dims(1), input_shape.Dims(2), input_shape.Dims(3)};
    const int kernel_shape[] = {filter_shape.Dims(0), filter_shape.Dims(1), filter_shape.Dims(2), filter_shape.Dims(3)};
    const int output_shape[] = {output_shape.Dims(0), output_shape.Dims(1), output_shape.Dims(2), output_shape.Dims(3)};
    
    //SPAD Size (precalculated?)
    int plm_in=1024;
    int plm_w=1024;
    int plm_out=1024;

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
    
    //tile batch size
    tile_shape[0]=input_shape[0];
    w_tile_shape[0]=kernel_shape[0];
    o_tile_shape[0]=output_shape[0];
    
    //tile channel size
    tile_shape[3]=input_shape[3];
    w_tile_shape[3]=kernel_shape[3];
    o_tile_shape[3]=output_shape[3];
    
    //tile width size
    tile_shape[2]=input_shape[2];
    w_tile_shape[2]=kernel_shape[2];
    o_tile_shape[2]=output_shape[2];
    if (input_shape[2]>=systolic_size[1]){
        tile_shape[2]=systolic_size[1];
        int stride_correction_w=kernel_shape[2];
        o_tile_shape[2]=1;
        while(stride_correction_w+stride_w<=systolic_size[1]){
            stride_correction_w=stride_correction_w+stride_w;
            o_tile_shape[2]=o_tile_shape[2]+1;
        }
        if (systolic_size[1]>=stride_correction_w){
            tile_shape[2]=stride_correction_w;
        }
    }

    //tile height size
    tile_shape[1]=input_shape[1];
    w_tile_shape[1]=kernel_shape[1];
    o_tile_shape[1]=output_shape[1];
    if (input_shape[1]>=systolic_size[0]){
        tile_shape[1]=systolic_size[0];
        int stride_correction_h=kernel_shape[1];
        o_tile_shape[1]=1;
        while(stride_correction_h+stride_h<=systolic_size[0]){
            stride_correction_h=stride_correction_h+stride_h;
            o_tile_shape[1]=o_tile_shape[1]+1;
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
    if (layer_type!=1){ //if depthwise
        while (tile_shape[3]>1&&(plm_w<mem_w||plm_out<mem_out)){
            //half the channel size
            tile_shape[3]=(tile_shape[3]%2==0)? (tile_shape[3]/2):(tile_shape[3]/2+1);
            //kernel and output channels follow
            w_tile_shape[3]=tile_shape[3];
            o_tile_shape[3]=tile_shape[3];
            mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
            mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
            mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
    }
    else{ //if pointwise
    */
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
    channel_number=(tile_shape[3]%input_shape[3]==0)? (tile_shape[3]/input_shape[3]):(tile_shape[3]/input_shape[3]+1);
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
    //how many tiles
    tile_number=channel_number*width_number*height_number*batch_number;

    //tiled addresses stored in an array
    uint8_t *tile_address;

    int height_count=input_shape[1]+1;
    int width_count=input_shape[2]+1;
    int channel_count=input_shape[3]+1;

    for(int batch=0; batch<kernel_shape[0]; batch++){
        for(int tile=0; tile<tile_number; tile++){
            int ttile=tile+1;
            for(int height=0; height<std::min(tile_shape[1], height_count+1); height++){
                for(int width=0; width<std::min(tile_shape[2], width_count+1); width++){
                    for(int channel=0; channel<std::min(tile_shape[3], channel_count+1); channel++){
                        //the head-scratching tiling offset
                        int height_off=0;
                        int width_off=0;
                        int pixel;
                        if(tile>=height_number){
                            height_off=(((tile)/height_number))*(tile_shape[1]-overlap_h);
                        }
                        if(tile%width_number!=0){
                            width_off=(tile%width_number)*(tile_shape[2]-overlap_w);
                        }
                        *tile_addressfilter_data[(((batch*tile_shape[1]+height)*tile_shape[1]+width)*tile_shape[2]+channel)*tile_shape[3]]=
                            input_data[(((batch*input_shape[1]*input_shape[2]*input_shape[3]+height+height_off)*v[2]*input_shape[3])+((width+width_off)*input_shape[3])+channel)];
                    }
                }
            }
        //The Edge-Handler
        if(channel_count<=tile_shape[3]){
            channel_count=padded_input_shape[3];
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
        }
    }
    }
}
}
/*
reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                   weights_comp_td,
                                                   data.weights_scratch_index),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(
                  micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
MicroPrintf("Tile size is %d x %d\n",
            , );
*/