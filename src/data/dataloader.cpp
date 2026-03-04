#include <llm/data.hpp>

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

namespace llm {

DataLoader::DataLoader(const Dataset* dataset,
                       int64_t batch_size,
                       bool shuffle,
                       uint64_t shuffle_seed)
    : dataset_(dataset), batch_size_(batch_size) {
  if (!dataset_)
    throw std::invalid_argument("DataLoader: dataset must be non-null");
  if (batch_size_ <= 0)
    throw std::invalid_argument("DataLoader: batch_size must be positive");

  const int64_t n = dataset_->size();
  indices_.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    indices_[static_cast<size_t>(i)] = i;

  if (shuffle && n > 0) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(shuffle_seed));
    std::shuffle(indices_.begin(), indices_.end(), rng);
  }
}

int64_t DataLoader::num_batches() const {
  const int64_t n = dataset_->size();
  if (n == 0) return 0;
  return (n + batch_size_ - 1) / batch_size_;
}

Sample DataLoader::get_batch(int64_t k) const {
  const int64_t num_b = num_batches();
  if (k < 0 || k >= num_b)
    throw std::out_of_range("DataLoader::get_batch: batch index out of range");

  const int64_t n = dataset_->size();
  const int64_t start = k * batch_size_;
  const int64_t end = std::min(start + batch_size_, n);
  const int64_t B = end - start;

  Sample first = dataset_->get(indices_[static_cast<size_t>(start)]);
  const int64_t seq_len = first.first.shape()[0];

  Tensor x_batch({B, seq_len}, DType::Int64, Device::cpu(), false);
  Tensor y_batch({B, seq_len}, DType::Int64, Device::cpu(), false);

  int64_t* px = x_batch.data_int64();
  int64_t* py = y_batch.data_int64();

  for (int64_t j = 0; j < B; ++j) {
    Sample s = dataset_->get(indices_[static_cast<size_t>(start + j)]);
    const int64_t* sx = s.first.data_int64();
    const int64_t* sy = s.second.data_int64();
    std::memcpy(px + j * seq_len, sx, static_cast<size_t>(seq_len) * sizeof(int64_t));
    std::memcpy(py + j * seq_len, sy, static_cast<size_t>(seq_len) * sizeof(int64_t));
  }

  return {x_batch, y_batch};
}

}  // namespace llm
