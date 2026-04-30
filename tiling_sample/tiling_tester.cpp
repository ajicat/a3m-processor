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
    
    //Dependent on the SPAD
    int plm_in=512;
    int plm_w=512;
    int plm_out=512;

    //Parameters will be provided per layer, variables set for testing only.
    int layer_type = 0; //Depthwise=0, Pointwise=1
    int input_shape[] = {1, 48, 48, 8};
    int kernel_shape[] = {1, 3, 3, 8};
    int output_shape[] = {1, 48, 48, 16};
    int stride_h=2;
    int stride_w=2;
    int overlap_h=kernel_shape[1]-stride_h;
    int overlap_w=kernel_shape[2]-stride_w;

    //tile batch size (for inputs)
    tile_shape[0]=input_shape[0];
    //for weights, one batch at a time
    //tile_shape[0]=1;

    //tile channel size
    tile_shape[3]=input_shape[3];
    if (input_shape[3]>=systolic_size[0]){
        tile_shape[3]=systolic_size[0];
    }
    //tile width size
    tile_shape[2]=input_shape[2];
    if (input_shape[2]>=systolic_size[1]){
        tile_shape[2]=systolic_size[1];
        int stride_correction_w=kernel_shape[2];
        while(stride_correction_w+stride_w<=systolic_size[1]){
            stride_correction_w=stride_correction_w+stride_w;
        }
        if (systolic_size[1]>=stride_correction_w){
            tile_shape[2]=stride_correction_w;
        }
    }
    //tile height size
    tile_shape[1]=input_shape[1];
    if (input_shape[1]>=systolic_size[0]){
        tile_shape[1]=systolic_size[0];
        int stride_correction_h=kernel_shape[1];
        while(stride_correction_h+stride_h<=systolic_size[0]){
            stride_correction_h=stride_correction_h+stride_h;
        }
        if (systolic_size[0]>=stride_correction_h){
            tile_shape[1]=stride_correction_h;
        }
    }
    
    int mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
    int mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
    int mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
    
    if (layertype!=1){
        while (tile_shape[3]>1&&(plm_w<mem_w||plm_out<mem_out)){
            tile_shape[3]=(tile_shape[3]%2==0)? (tile_shape[3]/2):(tile_shape[3]/2+1);
            mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
            mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
            mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
        }
        while (tile_shape[1]>1&&(plm_w<mem_w||plm_out<mem_out)){
                tile_shape[1]=tile_shape[1]-1;
                mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
                mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
                mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
                if (tile_shape[2]>1&&(plm_w<mem_w||plm_out<mem_out)){
                    tile_shape[2]=tile_shape[2]-1;
                    mem_in=tile_shape[1]*tile_shape[2]*tile_shape[3];
                    mem_w=kernel_shape[1]*kernel_shape[2]*tile_shape[3]*kernel_shape[3];
                    mem_out=tile_shape[1]*tile_shape[2]*kernel_shape[3];
                }
        }
    }
    //how many channel iterations
    channel_number=(tile_shape[3]%input_shape[3]==0)? (tile_shape[3]/input_shape[3]):(tile_shape[3]/input_shape[3]+1);
    //how many width iterations
    width_number=1;
    int width_iter=input_shape[2]-tile_shape[2];
    while(width_iter>0){
        width_number+=1;
        width_iter=width_iter-tile_shape[2]+overlap_w;
    }
    //how many height iterations
    height_number=1; 
    int height_iter=input_shape[1]-tile_shape[1];
    while(height_iter>0){
        height_number+=1;
        height_iter=height_iter-tile_shape[1]+overlap_h;
    }
    //for weights there will be batches here also which is just batch_number=kernel_shape[0];
    //how many tiles
    tile_number=channel_number*width_number*height_number*kernel_shape[0];
    
    std::cout << "Tiles for Layer 1: "<<tile_number<<std::endl<<"Tile Size: "<<tile_shape[0]<<"x"<<tile_shape[1]<<"x"<<tile_shape[2]<<"x"<<tile_shape[3]<<std::endl;
    int height_count=input_shape[1];
    int width_count=input_shape[2];
    int channel_count=input_shape[3];
    for(int tile=0; tile<tile_number; tile++){
        int ttile=tile+1;
        //for(int batch=0; batch<weight_shape[0], batch++)
        std::cout<<std::endl<<"Tile "<<ttile<<":";
        for(int height=0; height<std::min(tile_shape[1], height_count+1); height++){
            std::cout<<std::endl;
            for(int width=0; width<std::min(tile_shape[2], width_count+1); width++){
                for(int channel=0; channel<std::min(tile_shape[3], channel_count+1); channel++){
                    //the head-scratching tiling offset
                    int height_off=0;
                    int width_off=0;
                    if(tile>=height_number){
                        height_off=(((tile)/height_number))*(tile_shape[1]-overlap_h);
                    }
                    if(tile%width_number!=0){
                        width_off=(tile%width_number)*(tile_shape[2]-overlap_w);
                    }
                    int pixel=image[(((height+height_off)*input_shape[2]*input_shape[3])+((width+width_off)*input_shape[3])+channel)];
                    //+batch*kernel_shape[1]*kernel_shape[2]*kernel_shape[3]
                    //int input_address=SYSTOLIC_BASE+8*[(((height+height_off)*input_shape[2]*input_shape[3])+((width+width_off)*input_shape[3])+channel)];
                    std::cout<<std::uppercase<<std::hex<<pixel<<" ";//remove all these testing printouts
                     
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
        width_count=input_shape[2];
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
                height_count=input_shape[1];
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