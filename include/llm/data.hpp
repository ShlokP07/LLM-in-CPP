#pragma once

#include <llm/tensor.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace llm {

/**
 * Sample for language modeling: input token ids (x) and target token ids (y).
 * Typically x and y are (seq_len,) int64; for next-token prediction y is often x shifted by 1.
 */
using Sample = std::pair<Tensor, Tensor>;

/**
 * Dataset interface: fixed size and indexed access to (x, y) token sequences.
 * x and y are int64 tensors of shape (seq_len,).
 */
class Dataset {
public:
  virtual ~Dataset() = default;

  /** Number of samples. */
  virtual int64_t size() const = 0;

  /** Return sample at index i (0 <= i < size()). x and y are (seq_len,) int64. */
  virtual Sample get(int64_t i) const = 0;
};

/**
 * Dataset backed by two 2D int64 tensors: input (N, seq_len) and target (N, seq_len).
 * get(i) returns (input[i], target[i]) as (seq_len,) tensors (copied).
 */
class TensorDataset : public Dataset {
public:
  /** input and target must be 2D int64, same shape (N, seq_len). */
  TensorDataset(const Tensor& input, const Tensor& target);

  int64_t size() const override;
  Sample get(int64_t i) const override;

  int64_t seq_len() const { return seq_len_; }

private:
  Tensor input_;   // (N, seq_len) int64
  Tensor target_;  // (N, seq_len) int64
  int64_t N_;
  int64_t seq_len_;
};

/**
 * DataLoader: batches over a Dataset with optional shuffle.
 * Yields batches of (x_batch, y_batch) with shapes (batch_size, seq_len) int64.
 * Last batch may be smaller if size() is not divisible by batch_size.
 */
class DataLoader {
public:
  /**
   * @param dataset  Dataset to iterate over (not owned; must outlive the loader).
   * @param batch_size  Number of samples per batch.
   * @param shuffle  If true, sample indices are shuffled at construction (using shuffle_seed).
   * @param shuffle_seed  Used when shuffle is true.
   */
  DataLoader(const Dataset* dataset,
             int64_t batch_size,
             bool shuffle = false,
             uint64_t shuffle_seed = 0);

  /** Number of batches (last batch may be smaller). */
  int64_t num_batches() const;

  /** Return the k-th batch (0 <= k < num_batches()). x and y are (B, seq_len) int64, B = batch_size (or smaller for last batch). */
  Sample get_batch(int64_t k) const;

  int64_t batch_size() const { return batch_size_; }

private:
  const Dataset* dataset_;
  int64_t batch_size_;
  std::vector<int64_t> indices_;  // order in which to take samples (0..size()-1, possibly shuffled)
};

}  // namespace llm
