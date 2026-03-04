#include <llm/optim.hpp>

#include <cmath>

namespace llm {

AdamW::AdamW(const std::vector<Parameter*>& params,
             float lr,
             float beta1,
             float beta2,
             float eps,
             float weight_decay)
    : params_(params),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      step_count_(0) {
  state_m_.reserve(params_.size());
  state_v_.reserve(params_.size());
  for (Parameter* p : params_) {
    if (!p || p->dtype() != DType::Float32) {
      state_m_.push_back({});
      state_v_.push_back({});
      continue;
    }
    const int64_t n = p->numel();
    state_m_.push_back(std::vector<float>(static_cast<size_t>(n), 0.f));
    state_v_.push_back(std::vector<float>(static_cast<size_t>(n), 0.f));
  }
}

void AdamW::step() {
  step_count_++;
  const float t = static_cast<float>(step_count_);
  const float bias1 = 1.f - std::pow(beta1_, t);
  const float bias2 = 1.f - std::pow(beta2_, t);

  for (size_t idx = 0; idx < params_.size(); ++idx) {
    Parameter* p = params_[idx];
    if (!p || p->dtype() != DType::Float32) continue;
    std::shared_ptr<Tensor> g = p->grad();
    if (!g) continue;

    float* w = p->data_float();
    const float* gw = g->data_float();
    const int64_t n = p->numel();
    std::vector<float>& m = state_m_[idx];
    std::vector<float>& v = state_v_[idx];
    if (m.size() != static_cast<size_t>(n)) continue;

    for (int64_t i = 0; i < n; ++i) {
      const float gi = gw[i];
      m[i] = beta1_ * m[i] + (1.f - beta1_) * gi;
      v[i] = beta2_ * v[i] + (1.f - beta2_) * gi * gi;
    }

    for (int64_t i = 0; i < n; ++i) {
      const float m_hat = m[i] / bias1;
      const float v_hat = v[i] / bias2;
      w[i] -= lr_ * (m_hat / (std::sqrt(v_hat) + eps_));
    }

    if (weight_decay_ != 0.f) {
      for (int64_t i = 0; i < n; ++i)
        w[i] *= (1.f - lr_ * weight_decay_);
    }
  }
}

void AdamW::zero_grad() {
  for (Parameter* p : params_) {
    if (!p) continue;
    p->set_grad(std::shared_ptr<Tensor>());
  }
}

}  // namespace llm
