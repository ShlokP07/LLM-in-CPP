#pragma once

#include <llm/module.hpp>
#include <llm/tensor.hpp>

namespace llm {

/** Fully connected layer: y = x @ W^T + b. Weight (out_features, in_features), bias (out_features). */
class Linear : public Module {
public:
  Linear(int64_t in_features, int64_t out_features, bool bias = true);

  /** Forward: x shape (batch, in_features) -> out shape (batch, out_features). */
  Tensor operator()(const Tensor& x);

  int64_t in_features() const { return in_features_; }
  int64_t out_features() const { return out_features_; }
  bool has_bias() const { return has_bias_; }

private:
  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;
};

/** Lookup table: indices (N,) int64 -> embedded (N, embedding_dim) from weight (num_embeddings, embedding_dim). */
class Embedding : public Module {
public:
  Embedding(int64_t num_embeddings, int64_t embedding_dim);

  /** Forward: indices (N,) int64 -> output (N, embedding_dim) float32. */
  Tensor operator()(const Tensor& indices);

  int64_t num_embeddings() const { return num_embeddings_; }
  int64_t embedding_dim() const { return embedding_dim_; }

private:
  int64_t num_embeddings_;
  int64_t embedding_dim_;
};

/** Dropout layer. In train mode: zero elements with probability p and scale by 1/(1-p). In eval: identity. */
class Dropout : public Module {
public:
  explicit Dropout(float p = 0.5f);

  Tensor operator()(const Tensor& x);

  float p() const { return p_; }

private:
  float p_;
};

/** Layer normalization over the last dimension. Normalize then scale + shift with gamma/beta. */
class LayerNorm : public Module {
public:
  LayerNorm(int64_t normalized_shape, float eps = 1e-5f);

  /** Forward: x shape (..., D) -> same shape. Currently 2D only: (N, D). */
  Tensor operator()(const Tensor& x);

  int64_t normalized_shape() const { return normalized_shape_; }
  float eps() const { return eps_; }

private:
  int64_t normalized_shape_;
  float eps_;
};

}  // namespace llm
