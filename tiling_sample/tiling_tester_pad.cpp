#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sample.h> //Contains sample image

//  #define SYSTOLIC_BASE 0x40000000

int main(){ //yes i know i should turn this into a function
    int tile_shape[4]; //batch, height, width, channel
    int tile_number; //how many tiles there are
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
    int layer_type = 0; //Depthwise=0, Pointwise=1
    int input_shape[] = {1, 96, 96, 1};
    int kernel_shape[] = {1, 3, 3, 8};
    int output_shape[] = {1, 96, 96, 8};
    int stride_h=1; //check
    int stride_w=1; //check
    int zero_point=128;
    
    //Calculated Parameters
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
    std::cout<<"Padding Up: "<<padding_up<<std::endl<<"Padding Left: "<<padding_left<<std::endl<<"Padding Down: "<<padding_down<<std::endl<<"Padding Right: "<<padding_right<<std::endl;
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
        //output channels=kernel batches
        o_tile_shape[3]=w_tile_shape[0];
        while (tile_shape[1]>1&&tile_shape[2]>1&&(plm_w<mem_w||plm_out<mem_out)){
                //decrement the tile height and width (to stay square)
                tile_shape[1]=tile_shape[1]-1;
                tile_shape[2]=tile_shape[2]-1;
                
                mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
                mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
                mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
    }
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
    //how many tiles per batch
    tile_number=channel_number*width_number*height_number;

    std::cout << "Tiles for Layer 1: "<<tile_number<<std::endl<<"Input Tile Size: "<<tile_shape[0]<<"x"<<tile_shape[1]<<"x"<<tile_shape[2]<<"x"<<tile_shape[3]<<std::endl<<"Output Tile Size: "<<o_tile_shape[0]<<"x"<<o_tile_shape[1]<<"x"<<o_tile_shape[2]<<"x"<<o_tile_shape[3]<<std::endl;

    int height_count=padded_input_shape[1]+1;
    int width_count=padded_input_shape[2]+1;
    int channel_count=padded_input_shape[3]+1;
    int tile_size=tile_shape[0]*tile_shape[1]*tile_shape[2]*tile_shape[3];
    
    //how many output channel iterations
    int o_channel_number=(o_tile_shape[3]%output_shape[3]==0)? (o_tile_shape[3]/output_shape[3]):(o_tile_shape[3]/output_shape[3]+1);
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
    int o_height_count=output_shape[1]+1;
    int o_width_count=output_shape[2]+1;
    int o_channel_count=output_shape[3]+1;
    int o_tile_size=o_tile_shape[0]*o_tile_shape[1]*o_tile_shape[2]*o_tile_shape[3];
    
    uint8_t output[o_tile_size];
    
    for(int batch=0; batch<kernel_shape[0]; batch++){
        for(int tile=0; tile<tile_number; tile++){
            int ttile=tile+1;
            std::cout<<std::endl<<"Input Tile "<<ttile<<":";
            for(int height=0; height<std::min(tile_shape[1], height_count+1); height++){
                std::cout<<std::endl;
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
                        if((tile<height_number&&height-padding_up<0)||(tile%width_number==0&&width-padding_left<0)||(height+height_off+padding_down==padded_input_shape[1])||(width+width_off+padding_right==padded_input_shape[2])){
                            pixel=zero_point;
                        }
                        else{
                            pixel=image[((height+height_off-padding_up)*input_shape[2]*input_shape[3])+((width+width_off-padding_left)*input_shape[3])+channel];
                        }
                        std::cout<<std::uppercase<<std::hex<<pixel<<" ";
                        //uint8_t* input_address[pixel]=&image[(((batch*kernel_shape[1]*kernel_shape[2]*kernel_shape[3]+height+height_off)*input_shape[2]*input_shape[3])+((width+width_off)*input_shape[3])+channel)];
                        //pixel++;
                    }
                    
                    
                }
            }
            //The Almighty Edge-Handler
            if(channel_count<=tile_shape[3]){
                channel_count=padded_input_shape[3];
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
            std::cout<<std::endl<<"Output Tile "<<ttile<<":";
            for(int height=0; height<std::min(o_tile_shape[1], o_height_count); height++){
                std::cout<<std::endl;
                for(int width=0; width<std::min(o_tile_shape[2], o_width_count); width++){
                    for(int channel=0; channel<std::min(o_tile_shape[3], o_channel_count); channel++){
                        //the head-scratching tiling offset
                        int o_height_off=0;
                        int o_width_off=0;
                        uint8_t* pixel;
                        if(tile>=o_height_number){
                            o_height_off=(((tile)/o_height_number))*(o_tile_shape[1]);
                        }
                        if(tile%o_width_number!=0){
                            o_width_off=(tile%o_width_number)*(o_tile_shape[2]);
                        }
                        pixel=&output[((height+o_height_off)*output_shape[2]*output_shape[3])+((width+o_width_off)*output_shape[3])+channel];
                        std::cout<<pixel<<" ";
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
            
        }
    }



    
            
}