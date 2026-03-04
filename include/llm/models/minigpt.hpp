#pragma once

#include <llm/nn.hpp>

#include <cstdint>
#include <stdexcept>
#include <memory>
#include <vector>

namespace llm {
namespace models {

/** Configuration for MiniGPT (decoder-only transformer). */
struct MiniGPTConfig {
  int64_t dim = 64;           // model dimension
  int64_t num_heads = 4;      // number of attention heads
  int64_t num_layers = 2;     // number of transformer blocks
  int64_t vocab_size = 256;   // vocabulary size (embedding table)
  int64_t seq_len = 32;       // max sequence length (for reference)
  int64_t ffn_dim = 0;        // FFN hidden dim; 0 = 4 * dim

  int64_t head_dim() const {
    if (dim % num_heads != 0)
      throw std::invalid_argument("MiniGPTConfig: dim must be divisible by num_heads");
    return dim / num_heads;
  }
  int64_t ffn_hidden() const { return ffn_dim > 0 ? ffn_dim : 4 * dim; }
};

/**
 * Multi-head attention: projects to Q/K/V, runs scaled_dot_product_attention per head, concat, output projection.
 * Input/output (T, dim). Causal by default.
 */
class MultiHeadAttention : public Module {
public:
  MultiHeadAttention(int64_t dim, int64_t num_heads);

  Tensor operator()(const Tensor& x, bool causal = true);

  int64_t dim() const { return dim_; }
  int64_t num_heads() const { return num_heads_; }
  int64_t head_dim() const { return head_dim_; }

private:
  int64_t dim_;
  int64_t num_heads_;
  int64_t head_dim_;

  // Cached typed pointers to avoid per-forward map lookups/casts.
  std::shared_ptr<Linear> q_proj_;
  std::shared_ptr<Linear> k_proj_;
  std::shared_ptr<Linear> v_proj_;
  std::shared_ptr<Linear> out_proj_;

  // Cached selector matrices for head split/merge.
  // W_h: (dim, head_dim) selects the h-th head subspace from (T, dim) -> (T, head_dim)
  // U_h: (head_dim, dim) writes head output back into the h-th slice for concatenation.
  std::vector<Tensor> head_W_;
  std::vector<Tensor> head_U_;
};

/**
 * Single transformer block: pre-norm self-attention + residual, pre-norm FFN + residual.
 * FFN = Linear(dim, ffn_dim) -> GELU -> Linear(ffn_dim, dim).
 */
class TransformerBlock : public Module {
public:
  TransformerBlock(int64_t dim, int64_t num_heads, int64_t ffn_dim);

  Tensor operator()(const Tensor& x);

private:
  int64_t dim_;
  int64_t ffn_dim_;

  std::shared_ptr<LayerNorm> attn_ln_;
  std::shared_ptr<MultiHeadAttention> attn_;
  std::shared_ptr<LayerNorm> ffn_ln_;
  std::shared_ptr<Linear> ffn_1_;
  std::shared_ptr<Linear> ffn_2_;
};

/**
 * Decoder-only transformer (MiniGPT): embedding -> N x TransformerBlock -> final LayerNorm -> LM head.
 * Forward: input token ids (T,) int64 -> logits (T, vocab_size) float32.
 * Batch size 1 (sequence shape (T,)).
 */
class MiniGPT : public Module {
public:
  explicit MiniGPT(const MiniGPTConfig& config);

  /** Forward: indices (T,) int64 -> logits (T, vocab_size) float32. */
  Tensor forward(const Tensor& token_ids);

  const MiniGPTConfig& config() const { return config_; }

private:
  MiniGPTConfig config_;

  std::shared_ptr<Embedding> tok_embed_;
  std::shared_ptr<Embedding> pos_embed_;
  std::vector<std::shared_ptr<TransformerBlock>> blocks_;
  std::shared_ptr<LayerNorm> final_ln_;
  std::shared_ptr<Linear> lm_head_;
};

}  // namespace models
}  // namespace llm
