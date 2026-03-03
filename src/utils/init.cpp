#include <llm/init.hpp>
#include <llm/tensor.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace llm {

namespace {

std::mt19937& get_rng() {
  static std::mt19937 rng(std::random_device{}());
  return rng;
}

}  // namespace

void seed(uint64_t s) {
  get_rng().seed(static_cast<std::mt19937::result_type>(s));
}

void uniform_(Tensor& t, float low, float high) {
  if (t.dtype() != DType::Float32)
    throw std::invalid_argument("uniform_: expected float32 tensor");
  std::uniform_real_distribution<float> dist(low, high);
  float* p = t.data_float();
  for (int64_t i = 0; i < t.numel(); ++i)
    p[i] = dist(get_rng());
}

void normal_(Tensor& t, float mean, float stddev) {
  if (t.dtype() != DType::Float32)
    throw std::invalid_argument("normal_: expected float32 tensor");
  if (stddev <= 0.f)
    throw std::invalid_argument("normal_: stddev must be positive");
  std::normal_distribution<float> dist(mean, stddev);
  float* p = t.data_float();
  for (int64_t i = 0; i < t.numel(); ++i)
    p[i] = dist(get_rng());
}

void xavier_uniform_(Tensor& t) {
  if (t.dtype() != DType::Float32)
    throw std::invalid_argument("xavier_uniform_: expected float32 tensor");
  if (t.dim() != 2)
    throw std::invalid_argument("xavier_uniform_: expected 2D tensor");
  int64_t fan_in = t.shape()[1];
  int64_t fan_out = t.shape()[0];
  float limit = std::sqrt(6.f / (static_cast<float>(fan_in) + static_cast<float>(fan_out)));
  uniform_(t, -limit, limit);
}

void zeros_(Tensor& t) {
  if (t.dtype() == DType::Float32) {
    float* p = t.data_float();
    for (int64_t i = 0; i < t.numel(); ++i)
      p[i] = 0.f;
  } else if (t.dtype() == DType::Int64) {
    int64_t* p = t.data_int64();
    for (int64_t i = 0; i < t.numel(); ++i)
      p[i] = 0;
  } else {
    throw std::invalid_argument("zeros_: expected float32 or int64 tensor");
  }
}

}  // namespace llm
