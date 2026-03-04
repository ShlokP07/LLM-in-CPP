#include <llm/optim.hpp>

namespace llm {

SGD::SGD(const std::vector<Parameter*>& params, float lr, float weight_decay)
    : params_(params), lr_(lr), weight_decay_(weight_decay) {}

void SGD::step() {
  for (Parameter* p : params_) {
    if (!p) continue;
    if (p->dtype() != DType::Float32) continue;
    std::shared_ptr<Tensor> g = p->grad();
    if (!g) continue;

    float* w = p->data_float();
    const float* gw = g->data_float();
    const int64_t n = p->numel();

    const float lr = lr_;
    const float wd = weight_decay_;
    for (int64_t i = 0; i < n; ++i) {
      float grad = gw[i];
      if (wd != 0.f)
        grad += wd * w[i];
      w[i] -= lr * grad;
    }
  }
}

void SGD::zero_grad() {
  for (Parameter* p : params_) {
    if (!p) continue;
    p->set_grad(std::shared_ptr<Tensor>());  // clear gradient
  }
}

}  // namespace llm

