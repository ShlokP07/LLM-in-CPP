#include <llm/nn.hpp>
#include <llm/init.hpp>
#include <llm/autograd.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace llm {

namespace {

class LayerNormBackward : public AutogradNode {
public:
  LayerNormBackward(std::shared_ptr<Tensor> x,
                    std::shared_ptr<Tensor> gamma,
                    std::shared_ptr<Tensor> beta,
                    std::shared_ptr<Tensor> mean,
                    std::shared_ptr<Tensor> std_val,
                    std::shared_ptr<Tensor> x_norm,
                    int64_t N,
                    int64_t D)
      : x_(std::move(x)),
        gamma_(std::move(gamma)),
        beta_(std::move(beta)),
        mean_(std::move(mean)),
        std_(std::move(std_val)),
        x_norm_(std::move(x_norm)),
        N_(N),
        D_(D) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {x_, gamma_, beta_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    const float* go = grad_output->data_float();
    const float* xn = x_norm_->data_float();
    const float* gm = gamma_->data_float();
    const float* m = mean_->data_float();
    const float* s = std_->data_float();
    const float* px = x_->data_float();

    const float scale = 1.f / static_cast<float>(D_);

    // dgamma[j] = sum_i grad_output[i,j] * x_norm[i,j]
    if (gamma_ && gamma_->requires_grad()) {
      auto ggamma = std::make_shared<Tensor>(std::vector<int64_t>{D_},
                                             DType::Float32, gamma_->device(), false);
      float* pg = ggamma->data_float();
      for (int64_t j = 0; j < D_; ++j) {
        float acc = 0.f;
        for (int64_t i = 0; i < N_; ++i)
          acc += go[i * D_ + j] * xn[i * D_ + j];
        pg[j] = acc;
      }
      gamma_->accumulate_grad(*ggamma);
    }

    // dbeta[j] = sum_i grad_output[i,j]
    if (beta_ && beta_->requires_grad()) {
      auto gbeta = std::make_shared<Tensor>(std::vector<int64_t>{D_},
                                            DType::Float32, beta_->device(), false);
      float* pg = gbeta->data_float();
      for (int64_t j = 0; j < D_; ++j) {
        float acc = 0.f;
        for (int64_t i = 0; i < N_; ++i)
          acc += go[i * D_ + j];
        pg[j] = acc;
      }
      beta_->accumulate_grad(*gbeta);
    }

    // dx_norm[i,j] = grad_output[i,j] * gamma[j]
    std::vector<float> dx_norm(static_cast<size_t>(N_ * D_));
    for (int64_t i = 0; i < N_; ++i)
      for (int64_t j = 0; j < D_; ++j)
        dx_norm[static_cast<size_t>(i * D_ + j)] = go[i * D_ + j] * gm[j];

    // dmean[i] = - (1/std[i]) * sum_j dx_norm[i,j]
    std::vector<float> dmean(static_cast<size_t>(N_));
    for (int64_t i = 0; i < N_; ++i) {
      float sum_dx = 0.f;
      for (int64_t j = 0; j < D_; ++j)
        sum_dx += dx_norm[static_cast<size_t>(i * D_ + j)];
      dmean[static_cast<size_t>(i)] = -(1.f / s[i]) * sum_dx;
    }

    // dvar[i] = sum_j dx_norm[i,j] * (x[i,j]-mean[i]) * (-0.5) * (var+eps)^(-1.5)
    // var = std^2, so d(1/std)/d(var) = -0.5 * var^(-1.5) = -0.5 * (var+eps)^(-1.5)
    std::vector<float> dvar(static_cast<size_t>(N_));
    for (int64_t i = 0; i < N_; ++i) {
      float var_eps = s[i] * s[i];  // std = sqrt(var+eps), so var+eps = std^2
      float factor = -0.5f * std::pow(var_eps, -1.5f);
      float acc = 0.f;
      for (int64_t j = 0; j < D_; ++j) {
        float x_centered = px[i * D_ + j] - m[i];
        acc += dx_norm[static_cast<size_t>(i * D_ + j)] * x_centered;
      }
      dvar[static_cast<size_t>(i)] = acc * factor;
    }

    // dx[i,j] = dx_norm[i,j]/std[i] + dmean[i]/D + dvar[i] * 2*(x[i,j]-mean[i])/D
    if (x_ && x_->requires_grad()) {
      auto gx = std::make_shared<Tensor>(std::vector<int64_t>{N_, D_},
                                         DType::Float32, x_->device(), false);
      float* pg = gx->data_float();
      for (int64_t i = 0; i < N_; ++i) {
        for (int64_t j = 0; j < D_; ++j) {
          float x_centered = px[i * D_ + j] - m[i];
          pg[i * D_ + j] = dx_norm[static_cast<size_t>(i * D_ + j)] / s[i]
                           + dmean[static_cast<size_t>(i)] * scale
                           + dvar[static_cast<size_t>(i)] * 2.f * x_centered * scale;
        }
      }
      x_->accumulate_grad(*gx);
    }
  }

private:
  std::shared_ptr<Tensor> x_, gamma_, beta_;
  std::shared_ptr<Tensor> mean_, std_, x_norm_;
  int64_t N_, D_;
};

}  // namespace

LayerNorm::LayerNorm(int64_t normalized_shape, float eps)
    : normalized_shape_(normalized_shape),
      eps_(eps) {
  if (normalized_shape <= 0)
    throw std::invalid_argument("LayerNorm: normalized_shape must be positive");
  if (eps <= 0.f)
    throw std::invalid_argument("LayerNorm: eps must be positive");

  Parameter gamma = Parameter::zeros({normalized_shape});
  Parameter beta = Parameter::zeros({normalized_shape});
  zeros_(gamma);
  zeros_(beta);
  // Common init: gamma = ones, beta = zeros (so initial output = normalized)
  for (int64_t i = 0; i < normalized_shape; ++i)
    gamma.data_float()[i] = 1.f;
  register_parameter("gamma", gamma);
  register_parameter("beta", beta);
}

Tensor LayerNorm::operator()(const Tensor& x) {
  if (x.dtype() != DType::Float32)
    throw std::invalid_argument("LayerNorm: input must be float32");
  if (x.dim() != 2)
    throw std::invalid_argument("LayerNorm: only 2D input (N, D) supported");
  if (x.shape()[1] != normalized_shape_)
    throw std::invalid_argument("LayerNorm: last dim must match normalized_shape");

  const int64_t N = x.shape()[0];
  const int64_t D = x.shape()[1];
  const float* px = x.data_float();

  Parameter& gamma = parameters_.at("gamma");
  Parameter& beta = parameters_.at("beta");
  const float* pgamma = gamma.data_float();
  const float* pbeta = beta.data_float();

  // mean[i] = (1/D) * sum_j x[i,j]
  auto mean = std::make_shared<Tensor>(std::vector<int64_t>{N, 1},
                                       DType::Float32, x.device(), false);
  float* pmean = mean->data_float();
  for (int64_t i = 0; i < N; ++i) {
    float sum = 0.f;
    for (int64_t j = 0; j < D; ++j)
      sum += px[i * D + j];
    pmean[i] = sum / static_cast<float>(D);
  }

  // var[i] = (1/D) * sum_j (x[i,j] - mean[i])^2
  auto var = std::make_shared<Tensor>(std::vector<int64_t>{N, 1},
                                      DType::Float32, x.device(), false);
  float* pvar = var->data_float();
  for (int64_t i = 0; i < N; ++i) {
    float sum_sq = 0.f;
    for (int64_t j = 0; j < D; ++j) {
      float d = px[i * D + j] - pmean[i];
      sum_sq += d * d;
    }
    pvar[i] = sum_sq / static_cast<float>(D);
  }

  // std[i] = sqrt(var[i] + eps)
  auto std_val = std::make_shared<Tensor>(std::vector<int64_t>{N, 1},
                                          DType::Float32, x.device(), false);
  float* pstd = std_val->data_float();
  for (int64_t i = 0; i < N; ++i)
    pstd[i] = std::sqrt(pvar[i] + eps_);

  // x_norm[i,j] = (x[i,j] - mean[i]) / std[i]
  auto x_norm = std::make_shared<Tensor>(std::vector<int64_t>{N, D},
                                         DType::Float32, x.device(), false);
  float* pxn = x_norm->data_float();
  for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < D; ++j)
      pxn[i * D + j] = (px[i * D + j] - pmean[i]) / pstd[i];

  // out[i,j] = gamma[j] * x_norm[i,j] + beta[j]
  Tensor out({N, D}, DType::Float32, x.device(), false);
  float* pout = out.data_float();
  for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < D; ++j)
      pout[i * D + j] = pgamma[j] * pxn[i * D + j] + pbeta[j];

  bool need_grad = (x.requires_grad() || gamma.requires_grad() || beta.requires_grad());
  if (llm::is_grad_enabled() && need_grad) {
    out.set_requires_grad(true);
    auto node = std::make_shared<LayerNormBackward>(
        std::make_shared<Tensor>(x),
        std::make_shared<Tensor>(gamma),
        std::make_shared<Tensor>(beta),
        mean,
        std_val,
        x_norm,
        N,
        D);
    out.set_grad_fn(node);
  }

  return out;
}

}  // namespace llm
