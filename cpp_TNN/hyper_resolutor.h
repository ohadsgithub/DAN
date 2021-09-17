#ifndef TNN_EXAMPLES_BASE_HYPER_RESOLUTOR_H_
#define TNN_EXAMPLES_BASE_HYPER_RESOLUTOR_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <array>


#include "tnn_sdk_sample.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class HyperResolutorOutput : public TNNSDKOutput {
public:
    HyperResolutorOutput(std::shared_ptr<Mat> mat = nullptr) : TNNSDKOutput(mat) {};
    virtual ~HyperResolutorOutput(); // was HyperResolutorrOutput
    
    uint8_t *output_data_patch = new uint8_t[510*510*3];
    float *output_data_patch_real = new float[510*510*3];//float or double?
    //uint8_t *output_data_patch;
    int special_y_cshift;
    
};
    

class HyperResolutor : public TNN_NS::TNNSDKSample {
public:
    virtual ~HyperResolutor();
    virtual MatConvertParam GetConvertParamForInput(std::string tag = "");
    virtual std::shared_ptr<TNNSDKOutput> CreateSDKOutput();
    virtual Status ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output);
    virtual std::shared_ptr<TNN_NS::Mat> ProcessSDKInputMat(std::shared_ptr<TNN_NS::Mat> mat,
                                                              std::string name);
    
 
};

}

#endif // TNN_EXAMPLES_BASE_HYPER_RESOLUTOR_H_
