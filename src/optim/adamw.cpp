#include <llm/optim.hpp>

#include <cmath>
#include <cstring>

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

Module::StateDict AdamW::state_dict() const {
  Module::StateDict state;
  Tensor step_t = Tensor::zeros({1}, DType::Int64, Device::cpu(), false);
  step_t.data_int64()[0] = step_count_;
  state.emplace("step_count", std::move(step_t));
  for (size_t i = 0; i < state_m_.size(); ++i) {
    const std::vector<float>& m = state_m_[i];
    const std::vector<float>& v = state_v_[i];
    if (m.empty()) continue;
    Tensor tm({static_cast<int64_t>(m.size())}, DType::Float32, Device::cpu(), false);
    Tensor tv({static_cast<int64_t>(v.size())}, DType::Float32, Device::cpu(), false);
    std::memcpy(tm.data_float(), m.data(), m.size() * sizeof(float));
    std::memcpy(tv.data_float(), v.data(), v.size() * sizeof(float));
    std::string key = std::to_string(i);
    state.emplace(key + "_m", std::move(tm));
    state.emplace(key + "_v", std::move(tv));
  }
  return state;
}

void AdamW::load_state_dict(const Module::StateDict& state) {
  auto it_sc = state.find("step_count");
  if (it_sc != state.end() && it_sc->second.numel() >= 1)
    step_count_ = it_sc->second.data_int64()[0];
  for (size_t i = 0; i < state_m_.size(); ++i) {
    std::string key = std::to_string(i);
    auto it_m = state.find(key + "_m");
    auto it_v = state.find(key + "_v");
    if (it_m != state.end() && it_m->second.numel() == static_cast<int64_t>(state_m_[i].size())) {
      std::memcpy(state_m_[i].data(), it_m->second.data_float(),
                  state_m_[i].size() * sizeof(float));
    }
    if (it_v != state.end() && it_v->second.numel() == static_cast<int64_t>(state_v_[i].size())) {
      std::memcpy(state_v_[i].data(), it_v->second.data_float(),
                  state_v_[i].size() * sizeof(float));
    }
  }
}

}  // namespace llm
