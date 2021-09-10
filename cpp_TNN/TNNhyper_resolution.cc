#include <fstream>
#include <string>
#include <vector>


#include <cmath>//////////////////////
#include <iostream>/////////////


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

//static const char label_path_message[] = "(optional) label file path. Default is: ../../../assets/synset.txt";
//DEFINE_string(l, "../../../assets/synset.txt", label_path_message);

int main(int argc, char** argv) {
    if (!ParseAndCheckCommandLine(argc, argv)) {
        ShowUsage(argv[0]);
        //printf("\t-l, <label>    \t%s\n", label_path_message);
        return -1;
    }

    // 创建tnn实例
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

    
    auto predictor = std::make_shared<HyperResolutor>(); // ??
    

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
    int residue_width=image_width%255;
    int residue_height=image_height%255;
    
    int w_blocks=ceil(image_height/255);
    int h_blocks=ceil(image_width/255);
    
    int padded_height=h_blocks*255;
    int padded_width=w_blocks*255;
    
    uint8_t *data_padded = new uint8_t[padded_height*padded_weight*3];
    for (int k = 0; k < image_width * image_height * 3; ++k) {
        data_padded[k]=0;        
    }
    
    int y=0;
    int x=0; //must be ordered in one of two ways? what about data type?
    for (y = 0; y < image_height; ++y) {
        for (x = 0; x < image_width; ++x) {
            data_padded[3*(x+y*padded_width)]   = data[3*(x+y*image_width)];
            data_padded[3*(x+y*padded_width)+1]   = data[3*(x+y*image_width)+1];
            data_padded[3*(x+y*padded_width)+2]   = data[3*(x+y*image_width)+2];
        }
    }
    
    std::vector<int> nchw2 = {1, image_channel, padded_height, padded_width};
    std::vector<int> nchw255 = {1, image_channel, 255, 255};
    
    
    
    uint8_t *output_data = new uint8_t[image_width*image_height*3*4];

    
    
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
    
    uint8_t *patch_input_data = new uint8_t[255*255*3];
    uint8_t *patch_output_data = new uint8_t[510*510*3];
    
    int x2=0;
    int y2=0;
    int i=0;
    for (int j = 0; j < h_blocks; ++j) {
        for (i = 0; i < w_blocks; ++i) {
            
            //Status MatUtils::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue)
            for (y = 0; y < 255; ++y) {
                for (x = 0; x < 255; ++x) {
                    x2=x+i*255;
                    y2=y+j*255;
                    patch_input_data[3*(x+y*255)]   = data_padded[3*(x2+y2*padded_width)];
                    patch_input_data[3*(x+y*255)+1]   = data_padded[3*(x2+y2*padded_width)+1];
                    patch_input_data[3*(x+y*255)+2]   = data_padded[3*(x2+y2*padded_width)+2];
                }
            }
            //CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));
            CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw255, patch_input_data)), sdk_output));
            
            //dynamic_cast<HairSegmentationOutput *>(sdk_output.get())
            //(uint8_t*)(merged_image.data.get())
            //patch_output_data=(uint8_t*)(sdk_output.data.get()); ///////////////////////////////////////////change?
            //float *scores_data = (float *)output_mat_scores.get()->GetData();
            if (sdk_output && dynamic_cast<HyperResolutorOutput *>(sdk_output.get())) {
                auto SR_output = dynamic_cast<HyperResolutorOutput *>(sdk_output.get());
                patch_output_data = SR_output->output_data_patch;
            }
            
            
            for (y = 0; y < 510; ++y) {
                for (x = 0; x < 510; ++x) {
                    x2=x+i*510;
                    y2=y+j*510;
                    if ((x2<image_width) && (y2<image_height))
                    output_data[3*(x2+y2*510)]   = patch_output_data[3*(x+y*padded_width)];
                    output_data[3*(x2+y2*510)+1]   = patch_output_data[3*(x+y*padded_width)+1];
                    output_data[3*(x2+y2*510)+2]   = patch_output_data[3*(x+y*padded_width)+2];
                }
            }
            
        }
    }
    
    //CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));

    //int class_id = -1;
    //if (sdk_output && dynamic_cast<ImageClassifierOutput *>(sdk_output.get())) {
    //    auto classfy_output = dynamic_cast<ImageClassifierOutput *>(sdk_output.get());
    //    class_id = classfy_output->class_id;
    //}
    //完成计算，获取任意输出点
    
    
    //char img_buff[256];
    //char *input_imgfn = img_buff;
    //stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
   
    //int success = stbi_write_bmp(buff, image_orig_width, image_orig_height, 4, ifm_buf); //from TNNFaceDetector
    
    //stbi_write_png(*output_filename, image_width*2, image_height*2, 3, *output_data, int stride_in_bytes);
    
    
    //char img_buff[256];
    //char *input_imgfn = img_buff;
    //strncpy(input_imgfn, FLAGS_i.c_str(), 256);
    //input_imgfn
    char buff[256];  //  instead of "predictions", give it a name based on the input?
    sprintf(buff, "%s.png", "super_resolution"); //from TNNObjectDetector
    int success = stbi_write_bmp(buff, image_width*2, image_height*2, 3, output_data);
    if(!success) 
        return -1;
    //delete [] ifm_buf;
    
    //fprintf(stdout, "Hyper resolution done. Result: %sOutput argmax: %d\n", labels[class_id], class_id+1);
    //fprintf(stdout, "%s\n", predictor->GetBenchResult().Description().c_str());
    
    free(data);
    free(output_data);
    free(data_padded);
    
    return 0;
}
