#include "image_classifier.h"
#include "sample_timer.h"
#include <cmath>

namespace TNN_NS {

ImageClassifierOutput::~ImageClassifierOutput() {}

ImageClassifier::~ImageClassifier() {}

std::shared_ptr<Mat> ImageClassifier::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    auto target_dims = GetInputShape(name);
    auto input_height = input_mat->GetHeight();
    auto input_width = input_mat->GetWidth();
    auto input_channel = input_mat->GetChannel();
    if (target_dims.size() >= 4 &&  input_channel == target_dims[1] &&
        (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(),
                                                        input_mat->GetMatType(), target_dims);
        auto status = Resize(input_mat, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGE("%s\n", status.description().c_str());
            return nullptr;
        }
    }
    return input_mat;
}


MatConvertParam ImageClassifier::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_cvt_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    return input_cvt_param;
}

std::shared_ptr<TNNSDKOutput> ImageClassifier::CreateSDKOutput() {
    return std::make_shared<ImageClassifierOutput>();
}

Status ImageClassifier::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto output = dynamic_cast<ImageClassifierOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto output_mat_scores = output->GetMat();
    RETURN_VALUE_ON_NEQ(!output_mat_scores, false,
                        Status(TNNERR_PARAM_ERR, "output_mat_scores is invalid"));
    
    int class_id           = 0;
    float *scores_data = (float *)output_mat_scores.get()->GetData();
    float max_v        = scores_data[0];
    for (int i = 1; i < output_mat_scores->GetChannel(); ++i) {
        if (max_v < scores_data[i]) {
            max_v    = scores_data[i];
            class_id = i;
        }
    }
    output->class_id = class_id;
    return status;
}

}
