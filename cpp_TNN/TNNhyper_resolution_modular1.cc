#include <fstream>
#include <string>
#include <vector>

#include <cmath>
#include <iostream>

#include <stdlib.h>

#include "hyper_resolutor.h"
#include "macro.h"
#include "utils/utils.h"

#include "../flags.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image_write.h"

using namespace TNN_NS;


int main(int argc, char** argv) {
    if (!ParseAndCheckCommandLine(argc, argv)) {
        ShowUsage(argv[0]);
        return -1;
    }

    auto proto_content = fdLoadFile(FLAGS_p.c_str());
    auto model_content = fdLoadFile(FLAGS_m.c_str());

    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = "";
        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
        // if enable openvino, set option compute_units to openvino
        // if enable openvino/tensorrt, set option compute_units to openvino/tensorrt
        #ifdef _CUDA_
            option->compute_units = TNN_NS::TNNComputeUnitsGPU;
        #endif
    }

    
    auto predictor = std::make_shared<HyperResolutor>();
    

    char img_buff[256];
    char *input_imgfn = img_buff;
    strncpy(input_imgfn, FLAGS_i.c_str(), 256);
    printf("Hyper resolution x2 is about to start, and the picture is %s\n",input_imgfn);

    int image_width, image_height, image_channel;
    unsigned char *data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    if (!data) {
        fprintf(stderr, "hyper_resolutor open file %s failed.\n", input_imgfn);
    }

    std::vector<int> nchw = {1, image_channel, image_height, image_width};
    
    
    float image_height_float=(float) image_height;
    float image_width_float=(float) image_width;
    
    float h_blocks_float = ceil(image_height_float/255);
    float w_blocks_float = ceil(image_width_float/255);
    
    int h_blocks=(int) h_blocks_float;
    int w_blocks=(int) w_blocks_float;
    
    int padded_height=h_blocks*255;
    int padded_width=w_blocks*255;
    
    unsigned char *data_padded = new unsigned char[padded_height*padded_width*3];
    for (int indxk = 0; indxk < padded_width * padded_height * 3; ++indxk) {
        data_padded[indxk]=0;        
    }
    
    int y=0;
    int x=0;
    
    int y_reflect=0;
    int x_reflect=0;
    
    for (y = 0; y < padded_height; ++y) {
        for (x = 0; x < padded_width; ++x) {
            
            if ((x<2*image_width) and (y<2*image_height)) {
                
                x_reflect=image_width-1-abs(x-image_width+1);
                y_reflect=image_width-1-abs(y-image_height+1);
                    
                data_padded[3*(x+y*padded_width)]   = data[3*(x_reflect+y_reflect*image_width)];
                data_padded[3*(x+y*padded_width)+1]   = data[3*(x_reflect+y_reflect*image_width)+1];
                data_padded[3*(x+y*padded_width)+2]   = data[3*(x_reflect+y_reflect*image_width)+2]; 
            }
        }
    }
    
    char buff_input[256];
    sprintf(buff_input, "%s.png", "input_image"); 
    int success_padded = stbi_write_bmp(buff_input, image_width, image_height, 3, data);
    
    char buff_padded[256];
    sprintf(buff_padded, "%s.png", "padded_image"); 
    int success_input = stbi_write_bmp(buff_padded, padded_width, padded_height, 3, data_padded);
    
    char buff_padded2[256];
    sprintf(buff_padded2, "%s.jpg", "padded_image2"); 
    int success_padded2 = stbi_write_bmp(buff_padded2, padded_width, padded_height, 3, data_padded);
    
    char buff_inpcrop_save[256];
    int success_inpcrop_save = 0;
    
    char buff_lower_reflection_input_save[256];
    int success_lower_reflection_input_save = 0;
    char buff_upper_reflection_input_save[256];
    int success_upper_reflection_input_save = 0;
    char buff_lower_reflection_output_save[256];
    int success_lower_reflection_output_save = 0;
    char buff_upper_reflection_output_save[256];
    int success_upper_reflection_output_save = 0;
    
    char buff_outcrop_save[256];
    int success_outcrop_save = 0;
    
    
    std::vector<int> nchw2 = {1, image_channel, padded_height, padded_width};
    std::vector<int> nchw255 = {1, image_channel, 255, 255};
   
    uint8_t *output_data = new uint8_t[image_width*image_height*3*4];


    
    //Init
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput(); // inside for loop or ouside for loop?

    CHECK_TNN_STATUS(predictor->Init(option));
    
    
    unsigned char *patch_input_data = new unsigned char[255*255*3];

    uint8_t *patch_output_data = new uint8_t[510*510*3];
    
    
    int special_y_padder=124;
    int special_y_cshift=6;
    //int upper_y=127;
    //int lower_y=255-upper_y;
    int orig_y_depth=200;
    
    int x2=0;
    int y2=0;
    
    int y3=0;
    int y4=0;
    int y5=0;
    
    unsigned char *patch_input_lower_reflected = new unsigned char[255*255*3];
    unsigned char *patch_input_upper_reflected = new unsigned char[255*255*3];
    unsigned char *patch_output_lower_data_reflected = new unsigned char[510*510*3];
    unsigned char *patch_output_upper_data_reflected = new unsigned char[510*510*3];
    
  
    int i=0;
    for (int j = 0; j < h_blocks; ++j) {
        for (i = 0; i < w_blocks; ++i) {
            
            for (y = 0; y < 255; ++y) {
                for (x = 0; x < 255; ++x) {
                    x2=x+i*255;
                    y2=y+j*255;

                    patch_input_data[3*(x+y*255)]   = data_padded[3*(x2+y2*padded_width)];
                    patch_input_data[3*(x+y*255)+1]   = data_padded[3*(x2+y2*padded_width)+1];
                    patch_input_data[3*(x+y*255)+2]   = data_padded[3*(x2+y2*padded_width)+2];

                }
            }
            
            for (y = 0; y < 255; ++y) {
                for (x = 0; x < 255; ++x) {
                  
                    y3=254-abs(y-orig_y_depth+1);
                    y4=abs((254-orig_y_depth)-y);

                    patch_input_lower_reflected[3*(x+y*255)]   = patch_input_data[3*(x+y3*255)];
                    patch_input_lower_reflected[3*(x+y*255)+1]   = patch_input_data[3*(x+y3*255)+1];
                    patch_input_lower_reflected[3*(x+y*255)+2]   = patch_input_data[3*(x+y3*255)+2];
                    
                    patch_input_upper_reflected[3*(x+y*255)]   = patch_input_data[3*(x+y4*255)];
                    patch_input_upper_reflected[3*(x+y*255)+1]   = patch_input_data[3*(x+y4*255)+1];
                    patch_input_upper_reflected[3*(x+y*255)+2]   = patch_input_data[3*(x+y4*255)+2];

                }
            }
            
            
            sprintf(buff_inpcrop_save, "input_i%dj%d.png", i, j); 
            success_inpcrop_save = stbi_write_bmp(buff_inpcrop_save, 255, 255, 3, patch_input_data);
            
            sprintf(buff_lower_reflection_input_save, "input_lower_reflected_i%dj%d.png", i, j); 
            success_lower_reflection_input_save = stbi_write_bmp(buff_lower_reflection_input_save, 255, 255, 3, patch_input_lower_reflected);
            
            sprintf(buff_upper_reflection_input_save, "input_upper_reflected_i%dj%d.png", i, j); 
            success_upper_reflection_input_save = stbi_write_bmp(buff_upper_reflection_input_save, 255, 255, 3, patch_input_upper_reflected);

            
            //CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));
            CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw255, patch_input_lower_reflected)), sdk_output));
            if (sdk_output && dynamic_cast<HyperResolutorOutput *>(sdk_output.get())) {
                auto SR_output = dynamic_cast<HyperResolutorOutput *>(sdk_output.get());
                patch_output_lower_data_reflected = SR_output->output_data_patch;
            }

            
            
            CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw255, patch_input_upper_reflected)), sdk_output));
            if (sdk_output && dynamic_cast<HyperResolutorOutput *>(sdk_output.get())) {
                auto SR_output = dynamic_cast<HyperResolutorOutput *>(sdk_output.get());
                patch_output_upper_data_reflected = SR_output->output_data_patch;
            }
            
            
            for (y = 0; y < 510; ++y) {
                for (x = 0; x < 510; ++x) {
                    x2=x+i*510;
                    y2=y+j*510;

                    
                    if (y>255) {
                        y5=y-(510-2*orig_y_depth);
                        
                        patch_output_data[3*(x+y*510)]=patch_output_lower_data_reflected[3*(x+y4*510)];
                        patch_output_data[3*(x+y*510)+1]=patch_output_lower_data_reflected[3*(x+y4*510)+1];
                        patch_output_data[3*(x+y*510)+2]=patch_output_lower_data_reflected[3*(x+y4*510)+2];
                    }
                    else {
                        y5=y+510-2*orig_y_depth+1;
                        
                        patch_output_data[3*(x+y*510)]=patch_output_upper_data_reflected[3*(x+y4*510)];
                        patch_output_data[3*(x+y*510)+1]=patch_output_upper_data_reflected[3*(x+y4*510)+1];
                        patch_output_data[3*(x+y*510)+2]=patch_output_upper_data_reflected[3*(x+y4*510)+2];
                    }
                    
                    
                    if ((x2<2*image_width) && (y2<2*image_height))
                    {
                        
                        output_data[3*(x2+y2*2*image_width)]   = patch_output_data[3*(x+y*510)];
                        output_data[3*(x2+y2*2*image_width)+1]   = patch_output_data[3*(x+y*510)+1];
                        output_data[3*(x2+y2*2*image_width)+2]   = patch_output_data[3*(x+y*510)+2];
                        
                    }
                }
            }
            
            sprintf(buff_outcrop_save, "output_reordered_i%dj%d.png", i, j); 
            success_outcrop_save = stbi_write_bmp(buff_outcrop_save, 510, 510, 3, patch_output_data);
            
            sprintf(buff_reflection_output_save, "output_reflected_reordered_i%dj%d.png", i, j); 
            success_reflection_output_save = stbi_write_bmp(buff_reflection_output_save, 510, 510, 3, patch_output_data_reflected);
            
        }
    }
    

    
    

    char buff[256];
    //sprintf(buff, "%s.jpg", "super_resolution"); //  save as jpg?
    sprintf(buff, "%s.png", "super_resolution"); //from TNNObjectDetector
    int success = stbi_write_bmp(buff, image_width*2, image_height*2, 3, output_data);
    if(!success) {
        printf("something went wrong with saving the image\n");
        return -1;
    }
    else {
        printf("image saved\n");
    }
    
    free(data);
    free(output_data);
    free(data_padded);
    
    return 0;
}
