#include <fstream>
#include <string>

#include "tnn/core/tnn.h"

#include "tnn/core/tnn_impl.h"

namespace TNN_NS {

TNN::TNN() {}
TNN::~TNN() {
    DeInit();
}

Status TNN::Init(ModelConfig& config) {
    impl_ = TNNImplManager::GetTNNImpl(config.model_type);
    if (!impl_) {
        LOGE("Error: not support mode type: %d. If TNN is a static library, link it with option -Wl,--whole-archive tnn -Wl,--no-whole-archive on android or add -force_load on iOS\n", config.model_type);
        return Status(TNNERR_NET_ERR, "unsupported mode type, If TNN is a static library, link it with option -Wl,--whole-archive tnn -Wl,--no-whole-archive on android or add -force_load on iOS");
    }
  fprintf("model type is %d", int(config.model_type));
    return impl_->Init(config);
}

Status TNN::DeInit() {
    impl_ = nullptr;
    return TNN_OK;
}

Status TNN::AddOutput(const std::string& layer_name, int output_index) {
    // todo for output index
    if (!impl_) {
        LOGE("Error: impl_ is nil\n");
        return Status(TNNERR_NET_ERR, "tnn impl_ is nil");
    }
    return impl_->AddOutput(layer_name, output_index);
}

Status TNN::GetModelInputShapesMap(InputShapesMap& shapes_map) {
     if (!impl_) {
        LOGE("Error: impl_ is nil\n");
        return Status(TNNERR_NET_ERR, "tnn impl_ is nil");
    }
    return impl_->GetModelInputShapesMap(shapes_map);
}

std::shared_ptr<Instance> TNN::CreateInst(NetworkConfig& config, Status& status, InputShapesMap inputs_shape) {
    if (!impl_) {
        status = Status(TNNERR_NET_ERR, "tnn impl_ is nil");
        return nullptr;
    }

    return impl_->CreateInst(config, status, inputs_shape);
}

std::shared_ptr<Instance> TNN::CreateInst(NetworkConfig& config, Status& status, InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape) {
    if (!impl_) {
        status = Status(TNNERR_NET_ERR, "tnn impl_ is nil");
        return nullptr;
    }

    return impl_->CreateInst(config, status, min_inputs_shape, max_inputs_shape);
}

}  // namespace TNN_NS
