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
    
    
    //padding
    //int residue_width=image_width%255;
    //int residue_height=image_height%255;
    
    float image_height_float=(float) image_height;
    float image_width_float=(float) image_width;
    
    float h_blocks_float = ceil(image_height_float/255);
    float w_blocks_float = ceil(image_width_float/255);
    
    int h_blocks=(int) h_blocks_float;
    int w_blocks=(int) w_blocks_float;
    
    int padded_height=h_blocks*255;
    int padded_width=w_blocks*255;
    
    unsigned char *data_padded = new unsigned char[padded_height*padded_width*3];
    //uint8_t *data_padded = new uint8_t[padded_height*image_width*3];
    for (int indxk = 0; indxk < padded_width * padded_height * 3; ++indxk) {
        data_padded[indxk]=0;        
    }
    
    int y=0;
    int x=0; //must be ordered in one of two ways? what about data type?
    for (y = 0; y < image_height; ++y) {
        for (x = 0; x < image_width; ++x) {
            data_padded[3*(x+y*padded_width)]   = data[3*(x+y*image_width)];
            data_padded[3*(x+y*padded_width)+1]   = data[3*(x+y*image_width)+1];
            data_padded[3*(x+y*padded_width)+2]   = data[3*(x+y*image_width)+2]; //is there a need to divide by 255?
        }
    }
    
    char buff_input[256];
    sprintf(buff_input, "%s.png", "input_image"); 
    int success_padded = stbi_write_bmp(buff_input, image_width, image_height, 3, data);
    
    char buff_padded[256];
    sprintf(buff_padded, "%s.png", "padded_image"); 
    int success_input = stbi_write_bmp(buff_padded, padded_width, padded_height, 3, data_padded);
    
    //char buff_input2[256];
    //sprintf(buff_input2, "%s.jpg", "input_image2"); 
    //int success_input2 = stbi_write_bmp(buff_input2, image_width, image_height, 3, data);
    
    char buff_padded2[256];
    sprintf(buff_padded2, "%s.jpg", "padded_image2"); 
    int success_padded2 = stbi_write_bmp(buff_padded2, padded_width, padded_height, 3, data_padded);
    
    char buff_inpcrop_save[256];
    int success_inpcrop_save = 0;
    char buff_outcrop_save[256];
    int success_outcrop_save = 0;
    
    char buff_outcrop_reordered_save[256];
    int success_outcrop_reordered_save = 0;
    
    
    std::vector<int> nchw2 = {1, image_channel, padded_height, padded_width};
    std::vector<int> nchw255 = {1, image_channel, 255, 255};
   
    uint8_t *output_data = new uint8_t[image_width*image_height*3*4];
    
    //uint8_t *output_data_mid = new uint8_t[image_width*image_height*3*4];
    //uint8_t *output_data_pre = new uint8_t[image_width*image_height*3*4];


    
    //Init
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput(); // inside for loop or ouside for loop?

    CHECK_TNN_STATUS(predictor->Init(option));
    //Predict
    /*
    auto image_mat_padded = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw2, data_padded);
    //auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
    
    
    uint8_t *blank = new uint8_t[255*255*3];
    for (int t = 0; t < 255 * 255 * 3; ++t) {
        blank[k]=0;        
    }
    
    auto patchOf255 = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw255, blank);
    */
    
    unsigned char *patch_input_data = new unsigned char[255*255*3];
    
    //uint8_t *patch_input_data = new uint8_t[255*255*3];
    uint8_t *patch_output_data = new uint8_t[510*510*3];
    
    uint8_t *patch_output_data_reordered = new uint8_t[510*510*3];
    
    
    int xshift=16; // why 16?
    //int yshift=0;
    
    
    int x2=0;
    int y2=0;
    
    int xcs=0; //x circularly shifted
    int ycs=0;
  
  
    //int y_input_shift=6;
    //int yics=0;  
  
  
    int special_y_cshift=4; //8 in 510
    int special_y_padder=124;
    int y3=0;
    int y4=0;
    
    unsigned char *patch_input_reflected = new unsigned char[255*255*3];
    unsigned char *patch_output_data_reflected = new unsigned char[510*510*3];
    unsigned char *patch_output_data_reflected_reordered = new unsigned char[510*510*3];
    
  
    int i=0;
    for (int j = 0; j < h_blocks; ++j) {
        for (i = 0; i < w_blocks; ++i) {
            
            //Status MatUtils::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue)
            for (y = 0; y < 255; ++y) {
                for (x = 0; x < 255; ++x) {
                    x2=x+i*255;
                    y2=y+j*255;
                  
                    y3=254-abs(y-special_y_cshift-special_y_padder);
                    
                    //yics=(y-y_input_shift)%255;
                    //yics=(y+y_input_shift)%255;
                    //patch_input_data[3*(x+yics*255)]   = data_padded[3*(x2+y2*padded_width)];
                  
                    patch_input_data[3*(x+y*255)]   = data_padded[3*(x2+y2*padded_width)];
                    patch_input_data[3*(x+y*255)+1]   = data_padded[3*(x2+y2*padded_width)+1];
                    patch_input_data[3*(x+y*255)+2]   = data_padded[3*(x2+y2*padded_width)+2];
                    
                    patch_input_reflected[3*(x+y*255)]   = patch_input_data[3*(x+y3*255)];
                    patch_input_reflected[3*(x+y*255)+1]   = patch_input_data[3*(x+y3*255)+1];
                    patch_input_reflected[3*(x+y*255)+2]   = patch_input_data[3*(x+y3*255)+2];
                  
                    //patch_input_data[3*(x+y*255)]   = data_padded[3*(x2+y2*padded_width)];
                    //patch_input_data[3*(x+y*255)+1]   = data_padded[3*(x2+y2*padded_width)+1];
                    //patch_input_data[3*(x+y*255)+2]   = data_padded[3*(x2+y2*padded_width)+2];
                }
            }
            
            
            sprintf(buff_inpcrop_save, "input_i%dj%d.png", i, j); 
            success_inpcrop_save = stbi_write_bmp(buff_inpcrop_save, 255, 255, 3, patch_input_data);

            
            //CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));
            CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw255, patch_input_data)), sdk_output));
            
            
            if (sdk_output && dynamic_cast<HyperResolutorOutput *>(sdk_output.get())) {
                auto SR_output = dynamic_cast<HyperResolutorOutput *>(sdk_output.get());
                patch_output_data = SR_output->output_data_patch;
            }
            
            
            sprintf(buff_outcrop_save, "output_i%dj%d.png", i, j); 
            success_outcrop_save = stbi_write_bmp(buff_outcrop_save, 510, 510, 3, patch_output_data);
            
            
            CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw255, patch_input_reflected)), sdk_output));
            if (sdk_output && dynamic_cast<HyperResolutorOutput *>(sdk_output.get())) {
                auto SR_output = dynamic_cast<HyperResolutorOutput *>(sdk_output.get());
                patch_output_data_reflected = SR_output->output_data_patch;
            }
            
            //for (y = 0; y < special_y_cshift; ++y) {
                //for (x = 0; x < 510; ++x) {
                    
                  //  y4=

                //    patch_output_data[x+y*510] = patch_output_data_reflected[x+y4*510];
                //    patch_output_data[x+y*510+510*510] = patch_output_data_reflected[x+y4*510+510*510];
               //     patch_output_data[x+y*510+2*510*510] = patch_output_data_reflected[x+y4*510+2*510*510];
              //      
             //       
             //   }
            //}
            
            
            for (y = 0; y < 510; ++y) {
                for (x = 0; x < 510; ++x) {
                    x2=x+i*510;
                    y2=y+j*510;
                    
                    xcs=(x+xshift)%510;
                    //ycs=(y+yshift)%510;
                    ycs=(y+special_y_cshift)%510;
                        
                    patch_output_data_reordered[3*(x+y*510)]   = patch_output_data[xcs+ycs*510];
                    patch_output_data_reordered[3*(x+y*510)+1]   = patch_output_data[xcs+ycs*510+510*510];
                    patch_output_data_reordered[3*(x+y*510)+2]   = patch_output_data[xcs+ycs*510+2*510*510];
                    
                    patch_output_data_reflected_reordered[3*(x+y*510)]   = patch_output_data_reflected[xcs+ycs*510];
                    patch_output_data_reflected_reordered[3*(x+y*510)+1]   = patch_output_data_reflected[xcs+ycs*510+510*510];
                    patch_output_data_reflected_reordered[3*(x+y*510)+2]   = patch_output_data_reflected[xcs+ycs*510+2*510*510];
                    
                    if (y>510-2*special_y_cshift-1) {
                        y4=2*(special_y_padder+1)+(y-(510-2*special_y_cshift-1));
                        
                        patch_output_data_reordered[3*(x+y*510)]=patch_output_data_reflected_reordered[3*(x+y4*510)];
                        patch_output_data_reordered[3*(x+y*510)+1]=patch_output_data_reflected_reordered[3*(x+y4*510)+1];
                        patch_output_data_reordered[3*(x+y*510)+2]=patch_output_data_reflected_reordered[3*(x+y4*510)+2];
                    }
                    
                    
                    if ((x2<2*image_width) && (y2<2*image_height))
                    {
                        
                        output_data[3*(x2+y2*2*image_width)]   = patch_output_data_reordered[3*(x+y*510)];
                        output_data[3*(x2+y2*2*image_width)+1]   = patch_output_data_reordered[3*(x+y*510)+1];
                        output_data[3*(x2+y2*2*image_width)+2]   = patch_output_data_reordered[3*(x+y*510)+2];
                        
                        //output_data_pre[3*(x2+y2*2*image_width)]   = patch_output_data[3*(x+y*510)];
                        //output_data_pre[3*(x2+y2*2*image_width)+1]   = patch_output_data[3*(x+y*510)+1];
                        //output_data_pre[3*(x2+y2*2*image_width)+2]   = patch_output_data[3*(x+y*510)+2];
                        
                        //output_data_mid[3*(x2+y2*2*image_width)]   = patch_output_data[x+3*y*510];
                        //output_data_mid[3*(x2+y2*2*image_width)+1]   = patch_output_data[x+510+3*y*510];
                        //output_data_mid[3*(x2+y2*2*image_width)+2]   = patch_output_data[x+2*510+3*y*510];
                        
                        //output_data[3*(x2+y2*2*image_width)]   = patch_output_data[x+y*510];
                        //output_data[3*(x2+y2*2*image_width)+1]   = patch_output_data[x+y*510+510*510];
                        //output_data[3*(x2+y2*2*image_width)+2]   = patch_output_data[x+y*510+2*510*510];
                        
                    }
                }
            }
            
            sprintf(buff_outcrop_reordered_save, "output_reordered_i%dj%d.png", i, j); 
            success_outcrop_reordered_save = stbi_write_bmp(buff_outcrop_reordered_save, 510, 510, 3, patch_output_data_reordered);
            
        }
    }
    
    //char buff_mid[256];
    //sprintf(buff2, "%s.png", "super_resolution_mid"); //from TNNObjectDetector
    //int success_mid = stbi_write_bmp(buff_mid, image_width*2, image_height*2, 3, output_data_mid);
    
    //char buff_pre[256];
    //sprintf(buff3, "%s.png", "super_resolution_pre"); //from TNNObjectDetector
    //int success_pre = stbi_write_bmp(buff_pre, image_width*2, image_height*2, 3, output_data_pre);
    
    

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
