#include <fstream>
#include <string>
#include <vector>

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
        printf("\t-l, <label>    \t%s\n", label_path_message);
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

    
    auto predictor = std::make_shared<ImageClassifier>();
    

    char img_buff[256];
    char *input_imgfn = img_buff;
    strncpy(input_imgfn, FLAGS_i.c_str(), 256);
    printf("Classify is about to start, and the picture is %s\n",input_imgfn);

    int image_width, image_height, image_channel;
    unsigned char *data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);
    if (!data) {
        fprintf(stderr, "hyper_resolutor open file %s failed.\n", input_imgfn);
    }

    std::vector<int> nchw = {1, image_channel, image_height, image_width};

    //Init
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput();
    CHECK_TNN_STATUS(predictor->Init(option));
    //Predict
    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_NAIVE, TNN_NS::N8UC3, nchw, data);
    CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));

    int class_id = -1;
    if (sdk_output && dynamic_cast<ImageClassifierOutput *>(sdk_output.get())) {
        auto classfy_output = dynamic_cast<ImageClassifierOutput *>(sdk_output.get());
        class_id = classfy_output->class_id;
    }
    //完成计算，获取任意输出点
    
    
    char img_buff[256];
    char *input_imgfn = img_buff;
    //stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
    stbi_write_png(*output_filename, image_width*2, image_height*2, 3, *output_data, int stride_in_bytes);
    
    fprintf(stdout, "Classify done. Result: %sOutput argmax: %d\n", labels[class_id], class_id+1);
    fprintf(stdout, "%s\n", predictor->GetBenchResult().Description().c_str());
    free(data);
    return 0;
}
