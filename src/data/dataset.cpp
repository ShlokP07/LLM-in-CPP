#include <llm/data.hpp>

#include <cstring>
#include <stdexcept>

namespace llm {

TensorDataset::TensorDataset(const Tensor& input, const Tensor& target) {
  if (input.dtype() != DType::Int64 || target.dtype() != DType::Int64)
    throw std::invalid_argument("TensorDataset: input and target must be int64");
  if (input.dim() != 2 || target.dim() != 2)
    throw std::invalid_argument("TensorDataset: input and target must be 2D (N, seq_len)");
  if (input.shape() != target.shape())
    throw std::invalid_argument("TensorDataset: input and target must have same shape");

  input_ = input;
  target_ = target;
  N_ = input.shape()[0];
  seq_len_ = input.shape()[1];
}

int64_t TensorDataset::size() const {
  return N_;
}

Sample TensorDataset::get(int64_t i) const {
  if (i < 0 || i >= N_)
    throw std::out_of_range("TensorDataset::get: index out of range");

  Tensor x({seq_len_}, DType::Int64, Device::cpu(), false);
  Tensor y({seq_len_}, DType::Int64, Device::cpu(), false);

  const int64_t* pin = input_.data_int64();
  const int64_t* ptar = target_.data_int64();
  int64_t* px = x.data_int64();
  int64_t* py = y.data_int64();

  std::memcpy(px, pin + i * seq_len_, static_cast<size_t>(seq_len_) * sizeof(int64_t));
  std::memcpy(py, ptar + i * seq_len_, static_cast<size_t>(seq_len_) * sizeof(int64_t));

  return {x, y};
}

}  // namespace llm
