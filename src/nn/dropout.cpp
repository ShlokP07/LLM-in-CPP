#include <llm/nn.hpp>
#include <llm/init.hpp>
#include <llm/autograd.hpp>

#include <cstring>
#include <memory>
#include <stdexcept>

namespace llm {

namespace {

class DropoutBackward : public AutogradNode {
public:
  DropoutBackward(std::shared_ptr<Tensor> x,
                  std::shared_ptr<Tensor> mask,
                  float scale)
      : x_(std::move(x)), mask_(std::move(mask)), scale_(scale) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {x_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!x_ || !x_->requires_grad()) return;
    NoGradGuard guard;
    Tensor gx(x_->shape(), DType::Float32, x_->device(), false);
    const float* go = grad_output->data_float();
    const float* m = mask_->data_float();
    float* pgx = gx.data_float();
    for (int64_t i = 0; i < gx.numel(); ++i) {
      pgx[i] = go[i] * m[i] * scale_;
    }
    x_->accumulate_grad(gx);
  }

private:
  std::shared_ptr<Tensor> x_;
  std::shared_ptr<Tensor> mask_;
  float scale_;
};

}  // namespace

Dropout::Dropout(float p) : p_(p) {
  if (p_ < 0.f || p_ >= 1.f)
    throw std::invalid_argument("Dropout: p must be in [0, 1)");
}

Tensor Dropout::operator()(const Tensor& x) {
  if (x.dtype() != DType::Float32)
    throw std::invalid_argument("Dropout: input must be float32");

  // Eval mode: identity (no randomness, no new node).
  if (!is_training() || p_ == 0.f) {
    return x;
  }

  const float p_keep = 1.f - p_;
  const float scale = 1.f / p_keep;

  Tensor mask(x.shape(), DType::Float32, x.device(), false);
  bernoulli_mask_(mask, p_keep);

  Tensor out(x.shape(), DType::Float32, x.device(), false);
  const float* px = x.data_float();
  const float* pm = mask.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i) {
    po[i] = px[i] * pm[i] * scale;
  }

  if (is_grad_enabled() && x.requires_grad()) {
    out.set_requires_grad(true);
    auto node = std::make_shared<DropoutBackward>(
        std::make_shared<Tensor>(x),
        std::make_shared<Tensor>(mask),
        scale);
    out.set_grad_fn(node);
  }
  return out;
}

}  // namespace llm

