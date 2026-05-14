
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "sample.h"

//  #define SYSTOLIC_BASE 0x40000000

inline void DepthwiseConvPerChannel(
    const int stride_w, const int stride_h,
    const int depth_multiplier,
    const int input_shape[], const uint8_t image[],
    const int kernel_shape[], const uint8_t kernel_0[],
    const int bias_shape, const uint32_t bias_0[],
    const int output_shape[], uint8_t output_0[]) {

  const int batches = input_shape[0];
  const int output_depth = output_shape[3];
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];
  const int filter_height = kernel_shape[1];
  const int filter_width = kernel_shape[2];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_w);
            const int in_y_origin = (out_y * stride_h);
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + filter_x;
                const int in_y =
                    in_y_origin + filter_y;  
                  int32_t input_val = image[((batch * input_shape[1] + in_y) * input_shape[2] + in_x) * input_shape[3] + in_channel];
                  int32_t filter_val = kernel_0[((batch * kernel_shape[1] + filter_y) * kernel_shape[2] + filter_x) * kernel_shape[0] + output_channel];
                  acc += filter_val * (input_val);
              }
            }
            if (bias_0) {
              acc += bias_0[output_channel];
            }
            output_0[((batch * output_shape[1] + out_y) * output_shape[2] + out_x) * output_shape[3] + output_channel]
             = static_cast<int>(acc);
          }
        }
      }
    }
  }
}

inline void ConvPerChannel(
    const int stride_w, const int stride_h,
    const int input_shape[], const uint8_t image[],
    const int kernel_shape[], const uint8_t kernel_2[],
    const int bias_shape, const uint32_t bias_2[],
    const int output_shape[], uint8_t output_2[]) {

  const int batches = input_shape[0];
  const int output_depth = output_shape[3];
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];
  const int filter_height = kernel_shape[1];
  const int filter_width = kernel_shape[2];
  const int filter_input_depth = kernel_shape[3];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];

  const int groups = input_depth / filter_input_depth;
  const int filters_per_group = output_depth / groups;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_h);
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_w);
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + filter_x;

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
                    image[((batch * input_shape[1] + in_y) * input_shape[2] + in_x) * input_shape[3] + (in_channel + group * filter_input_depth)];
                int32_t filter_val = kernel_2[((out_channel * kernel_shape[1] + filter_y) * kernel_shape[2] + filter_x) * kernel_shape[3] + in_channel];
                acc += filter_val * input_val;
              }
            }
          }

          if (bias_2) {
            acc += bias_2[out_channel];
          }
          output_2[((batch * output_shape[1] + out_y) * output_shape[2] + out_x) * output_shape[3] + out_channel] =
              static_cast<int>(acc);
        }
      }
    }
  }
}


int main(){
    int tile_shape[4]; //batch, height, width, channel
    int tile_number; //how many tiles there are per batch
    int height_number;
    int width_number;
    int channel_number;
    int systolic_size[]={8, 8};
    
    int w_tile_shape[4]; //weight tile
    int batch_number;
    
    int o_tile_shape[4]; //output tile
    
    //Dependent on the SPAD
    int plm_in=512;
    int plm_w=1024;
    int plm_out=1024;

    //check
    //Parameters will be provided per layer, variables set for testing only.
    int layer_type = 0; //Pointwise=0, Depthwise=1
    int input_shape[] = {1, 48, 48, 8};
    int kernel_shape[] = {16, 1, 1, 8};
    int output_shape[] = {1, 48, 48, 16};
    int stride_h=1; //check
    int stride_w=1; //check
    int zero_point=0;
    
    //Calculated Parameters
    int overlap_h=kernel_shape[1]-stride_h;
    int overlap_w=kernel_shape[2]-stride_w;
    const int pad_height = (output_shape[1]*stride_h)+overlap_h-input_shape[1];
    const int pad_width = (output_shape[1]*stride_h)+overlap_h-input_shape[1];
    
    int padding_up=0;
    int padding_left=0;
    int padding_down=0; 
    int padding_right=0; 
    if (pad_height%2==0){
        padding_up=(pad_height)/2;
        padding_down=(pad_height)/2;
    }
    else {
        padding_up=pad_height/2+1;
        padding_down=pad_height/2;
    }
    if (((output_shape[2]*stride_w)+overlap_w-input_shape[2])%2==0){
        padding_left=pad_width/2;
        padding_right=pad_width/2;
    }
    else {
        padding_left=pad_width/2+1;
        padding_right=pad_width/2;
    }
    //std::cout<<"Padding Up: "<<padding_up<<std::endl<<"Padding Left: "<<padding_left<<std::endl<<"Padding Down: "<<padding_down<<std::endl<<"Padding Right: "<<padding_right<<std::endl;
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
        if (systolic_size[0]>=stride_correction_h){
            tile_shape[1]=stride_correction_h;
        }
    }
    
    int mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
    int mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
    int mem_out=o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
    
    if (layer_type!=0){ //if depthwise
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
        //output channels=kernel batches
        //set the batch size to 1
        w_tile_shape[0]=1;
        o_tile_shape[3]=w_tile_shape[0];
        while (tile_shape[1]>1&&tile_shape[2]>1&&(plm_w<mem_w||plm_out<mem_out)){
                //decrement the tile height and width (to stay square)
                tile_shape[1]=tile_shape[1]-1;
                tile_shape[2]=tile_shape[2]-1;
                
                w_tile_shape[0]=(w_tile_shape[0]%2==0)? (w_tile_shape[0]/2):(w_tile_shape[0]/2+1);
                mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
                mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
                mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
        o_tile_shape[3]=w_tile_shape[0];
    }
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

    std::cout << "Tiles for Layer 1: "<<tile_number<<std::endl<<"Input Tile Size: "<<tile_shape[0]<<"x"<<tile_shape[1]<<"x"<<tile_shape[2]<<"x"<<tile_shape[3]<<std::endl<<"Output Tile Size: "<<o_tile_shape[0]<<"x"<<o_tile_shape[1]<<"x"<<o_tile_shape[2]<<"x"<<o_tile_shape[3]<<std::endl;
    std::cout <<"Weight Tile Size: "<<w_tile_shape[0]<<"x"<<w_tile_shape[1]<<"x"<<w_tile_shape[2]<<"x"<<w_tile_shape[3]<<std::endl;

    //how many output channel iterations
    int o_channel_number;
    if(output_shape[3]%o_tile_shape[3]==0){
        o_channel_number=output_shape[3]/o_tile_shape[3];
    }
    else{
        o_channel_number=output_shape[3]/o_tile_shape[3]+1;
    }    //how many output width iterations
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
    uint8_t tile_data[tile_size];

    int output_size=output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3];
    uint8_t output_full[output_size];

    int o_tile_size=o_tile_shape[0]*o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
    uint8_t o_tile_data[o_tile_size];

    int w_tile_size=w_tile_shape[0]*kernel_shape[1]*kernel_shape[2]*w_tile_shape[3];
    uint8_t w_tile_data[w_tile_size];
    
    uint32_t bias_c[o_tile_shape[3]];

    int o_height_count=output_shape[1];
    int o_width_count=output_shape[2];
    int o_channel_count=output_shape[3];

    for(int batch=0; batch<kernel_shape[0]; batch++){
        int tile=0;
        int height_count=padded_input_shape[1]+1;
        int width_count=padded_input_shape[2]+1;
        int channel_count=padded_input_shape[3];
        
        for(int real_tile=0; real_tile<tile_number; real_tile++){
            int ttile=real_tile+1;
            if (layer_type!=0){
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((batch*w_tile_shape[1]+height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=kernel_0[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
                        }
                    }
                }
            }
            else{
                for(int height=0; height<kernel_shape[1]; height++){
                    for(int width=0; width<kernel_shape[2]; width++){
                        for(int channel=0; channel<w_tile_shape[3]; channel++){
                            w_tile_data[((batch*w_tile_shape[1]+height)*kernel_shape[2]+width)*kernel_shape[3]+channel]=kernel_2[((batch*kernel_shape[1]+height)*kernel_shape[2]+width)*w_tile_shape[3]+channel];
                        }
                    }
                }
            }
            for(int height=0; height<std::min(tile_shape[1], height_count+1); height++){
                //std::cout<<std::endl;
                for(int width=0; width<std::min(tile_shape[2], width_count+1); width++){
                    for(int channel=0; channel<std::min(tile_shape[3], channel_count+1); channel++){
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
                            image[((height+height_off-padding_up)*input_shape[2]*input_shape[3])+((width+width_off-padding_left)*input_shape[3])+channel+channel_off];
                        //std::cout<<std::uppercase<<std::hex<<pixel<<" ";
                        //uint8_t* input_address[pixel]=&image[(((batch*kernel_shape[1]*kernel_shape[2]*kernel_shape[3]+height+height_off)*input_shape[2]*input_shape[3])+((width+width_off)*input_shape[3])+channel)];
                        //pixel++;
                        }
                    }
                }
            }
            //The Almighty Edge-Handler
            if(channel_count<=tile_shape[3]){
                channel_count=input_shape[3];
            }
            else{
                channel_count=channel_count-tile_shape[3];
            }
            if(width_count<=tile_shape[2]){
                width_count=padded_input_shape[2]+1;
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
                    height_count=padded_input_shape[1]+1;
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

            if (layer_type!=0){
                for(int channel=0; channel<o_tile_shape[3]; channel++){
                    bias_c[channel]=bias_0[real_tile/(tile_shape[1]*tile_shape[2])+channel];
                }
                DepthwiseConvPerChannel(stride_w, stride_h, (o_tile_shape[3]/tile_shape[3]),
                tile_shape, tile_data, w_tile_shape, w_tile_data,
                o_tile_shape[3], bias_c, o_tile_shape, o_tile_data);
            }
            else{
                for(int channel=0; channel<o_tile_shape[3]; channel++){
                    bias_c[channel]=bias_2[real_tile/(tile_shape[1]*tile_shape[2])+channel];
                }
                ConvPerChannel(stride_w, stride_h, tile_shape, tile_data,
                w_tile_shape, w_tile_data, o_tile_shape[3], bias_c,
                o_tile_shape, o_tile_data);
            }
            //std::cout<<std::endl<<"Output Tile "<<ttile<<":";
            for(int height=0; height<std::min(o_tile_shape[1], o_height_count); height++){
                //std::cout<<std::endl;
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
                        output_full[((height+o_height_off)*output_shape[2]*output_shape[3])+((width+o_width_off)*output_shape[3])+channel+o_channel_off]=o_tile_data[((batch*o_tile_shape[1]+height)*o_tile_shape[2]+width)*o_tile_shape[3]+channel];
                        int pixel=image[((height+o_height_off)*output_shape[2]*output_shape[3])+((width+o_width_off)*output_shape[3])+channel+o_channel_off];
                        //std::cout<<std::uppercase<<std::hex<<pixel<<" ";
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
    
    int padded_input_size=input_shape[0]*(input_shape[1]+pad_height)*(input_shape[2]+pad_width)*input_shape[3];
    //std::cout<<input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]<<std::endl<<padded_input_size;
    uint8_t padded_image[padded_input_size];
    //std::cout << input_shape[0] <<std::endl<< padded_input_shape[1] <<std::endl<< padded_input_shape[2] <<std::endl<< input_shape[3] <<std::endl;
    //std::cout << "Padded Input" <<std::endl;
    /*
    for(int batch=0; batch<kernel_shape[0]; batch++){
        for(int height=0; height<padded_input_shape[1]; height++){
            //std::cout<<std::endl;
            for(int width=0; width<padded_input_shape[2]; width++){
                for(int channel=0; channel<input_shape[3]; channel++){
                    if((height-padding_up<0)||(width-padding_left<0)||(height+padding_down==padded_input_shape[1])||(width+padding_right==padded_input_shape[2])){
                        padded_image[((height) * padded_input_shape[2] + width) * input_shape[3] + channel]=zero_point;
                    }
                    else{
                        padded_image[((height) * padded_input_shape[2] + width) * input_shape[3] + channel]
                        =image[((batch*input_shape[1]+(height-padding_up))*input_shape[2]+(width-padding_left))*input_shape[3]+channel];
                    }
                    int pixel=padded_image[((height) * padded_input_shape[2] + width) * input_shape[3] + channel];
                    
                    //std::cout<<std::uppercase<<std::hex<<pixel<<" ";
                }
            }
        }
    }
        */
    
    /*if (layer_type!=0){
        std::cout << "Depthwise Test (Layer 0) Padded Tiled" <<std::endl;
        for(int batch=0; batch<output_shape[0]; batch++){
            for(int height=0; height<output_shape[1]; height++){
                std::cout<<std::endl;
                for(int width=0; width<output_shape[2]; width++){
                    for(int channel=0; channel<output_shape[3]; channel++){
                        int pixel=output_full[((batch * output_shape[1] + height) * output_shape[2] + width) * output_shape[3] + channel];
                        std::cout<<std::uppercase<<std::hex<<std::setfill('0')<<std::setw(2)<<pixel<<" ";
                    }
                }
            }
        }
    }
    else{
        std::cout << "Pointwise Test (Layer 2) Padded Tiled" <<std::endl;
        for(int batch=0; batch<output_shape[0]; batch++){
            for(int height=0; height<output_shape[1]; height++){
                std::cout<<std::endl;
                for(int width=0; width<output_shape[2]; width++){
                    for(int channel=0; channel<output_shape[3]; channel++){
                        int pixel=output_full[((batch * output_shape[1] + height) * output_shape[2] + width) * output_shape[3] + channel];
                        std::cout<<std::uppercase<<std::hex<<std::setfill('0')<<std::setw(2)<<pixel<<" ";
                    }
                }
            }
        }

    }*/
    
    
    if (layer_type!=0){
        uint8_t output_0[output_size];

        std::cout << "Depthwise Test (Layer 0) Padded Tiled" <<std::endl;
        DepthwiseConvPerChannel(stride_w, stride_h, (output_shape[3]/input_shape[3]),
        padded_input_shape, image, kernel_shape, kernel_0,
        output_shape[3], bias_0, output_shape, output_0);
        for(int batch=0; batch<kernel_shape[0]; batch++){
            for(int height=0; height<output_shape[1]; height++){
                std::cout<<std::endl;
                for(int width=0; width<output_shape[2]; width++){
                    for(int channel=0; channel<output_shape[3]; channel++){
                        int pixel=output_0[((batch * output_shape[1] + height) * output_shape[2] + width) * output_shape[3] + channel];
                        std::cout<<std::uppercase<<std::hex<<std::setfill('0')<<std::setw(2)<<pixel<<" ";
                    }
                }
            }
        }
    }
    else{
        uint8_t output_2[output_size];
        std::cout << "Pointwise Test (Layer 2) Padded Tiled" <<std::endl;
        ConvPerChannel(stride_w, stride_h, input_shape, image,
        kernel_shape, kernel_2, output_shape[3], bias_2,
        output_shape, output_2);
        for(int batch=0; batch<output_shape[0]; batch++){
            for(int height=0; height<output_shape[1]; height++){
                std::cout<<std::endl;
                for(int width=0; width<output_shape[2]; width++){
                    for(int channel=0; channel<output_shape[3]; channel++){
                        int pixel=output_2[((batch * output_shape[1] + height) * output_shape[2] + width) * output_shape[3] + channel];
                        std::cout<<std::uppercase<<std::hex<<std::setfill('0')<<std::setw(2)<<pixel<<" ";
                    }
                }
            }
        }

    }
    
}