#include <llm/models/minigpt.hpp>
#include <llm/ops.hpp>
#include <llm/init.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace llm {
namespace models {

namespace {

// Head split/concat trick (no slicing op required):
// - To "slice" the h-th head from a (T, dim) tensor, we multiply by a constant selector matrix
//   W_h ∈ R^{dim×head_dim} that picks the contiguous [h*head_dim : (h+1)*head_dim) columns.
// - To "concat" head outputs back into (T, dim), we multiply by U_h ∈ R^{head_dim×dim} and sum:
//     out = Σ_h out_h @ U_h
// This keeps everything differentiable using only existing matmul/add ops.
Tensor make_head_W(int64_t dim, int64_t head_dim, int64_t h) {
  Tensor W({dim, head_dim}, DType::Float32, Device::cpu(), false);
  zeros_(W);
  float* p = W.data_float();
  for (int64_t i = 0; i < head_dim; ++i)
    p[(h * head_dim + i) * head_dim + i] = 1.f;
  return W;
}

Tensor make_head_U(int64_t dim, int64_t head_dim, int64_t h) {
  Tensor U({head_dim, dim}, DType::Float32, Device::cpu(), false);
  zeros_(U);
  float* p = U.data_float();
  for (int64_t i = 0; i < head_dim; ++i)
    p[i * dim + (h * head_dim + i)] = 1.f;
  return U;
}

}  // namespace

// --- MultiHeadAttention ---

MultiHeadAttention::MultiHeadAttention(int64_t dim, int64_t num_heads)
    : dim_(dim), num_heads_(num_heads), head_dim_(dim / num_heads) {
  if (dim <= 0 || num_heads <= 0 || dim % num_heads != 0)
    throw std::invalid_argument("MultiHeadAttention: dim must be divisible by num_heads");
  q_proj_ = std::make_shared<Linear>(dim, dim);
  k_proj_ = std::make_shared<Linear>(dim, dim);
  v_proj_ = std::make_shared<Linear>(dim, dim);
  out_proj_ = std::make_shared<Linear>(dim, dim);
  register_module("q_proj", q_proj_);
  register_module("k_proj", k_proj_);
  register_module("v_proj", v_proj_);
  register_module("out_proj", out_proj_);

  head_W_.reserve(static_cast<size_t>(num_heads_));
  head_U_.reserve(static_cast<size_t>(num_heads_));
  for (int64_t h = 0; h < num_heads_; ++h) {
    head_W_.push_back(make_head_W(dim_, head_dim_, h));
    head_U_.push_back(make_head_U(dim_, head_dim_, h));
  }
}

Tensor MultiHeadAttention::operator()(const Tensor& x, bool causal) {
  const int64_t T = x.shape()[0];
  Tensor Q = (*q_proj_)(x);
  Tensor K = (*k_proj_)(x);
  Tensor V = (*v_proj_)(x);

  Tensor out = Tensor::zeros({T, dim_}, DType::Float32, Device::cpu(), false);
  for (int64_t h = 0; h < num_heads_; ++h) {
    const Tensor& W_h = head_W_[static_cast<size_t>(h)];
    const Tensor& U_h = head_U_[static_cast<size_t>(h)];
    Tensor Q_h = matmul(Q, W_h);
    Tensor K_h = matmul(K, W_h);
    Tensor V_h = matmul(V, W_h);
    Tensor out_h = scaled_dot_product_attention(Q_h, K_h, V_h, causal);
    out = add(out, matmul(out_h, U_h));
  }
  return (*out_proj_)(out);
}

// --- TransformerBlock ---

TransformerBlock::TransformerBlock(int64_t dim, int64_t num_heads, int64_t ffn_dim)
    : dim_(dim), ffn_dim_(ffn_dim) {
  attn_ln_ = std::make_shared<LayerNorm>(dim);
  attn_ = std::make_shared<MultiHeadAttention>(dim, num_heads);
  ffn_ln_ = std::make_shared<LayerNorm>(dim);
  ffn_1_ = std::make_shared<Linear>(dim, ffn_dim);
  ffn_2_ = std::make_shared<Linear>(ffn_dim, dim);
  register_module("attn_ln", attn_ln_);
  register_module("attn", attn_);
  register_module("ffn_ln", ffn_ln_);
  register_module("ffn_1", ffn_1_);
  register_module("ffn_2", ffn_2_);
}

Tensor TransformerBlock::operator()(const Tensor& x) {
  Tensor attn_ln_out = (*attn_ln_)(x);
  Tensor attn_out = (*attn_)(attn_ln_out, true);
  Tensor x1 = add(x, attn_out);

  Tensor ffn_ln_out = (*ffn_ln_)(x1);
  Tensor ffn_h = (*ffn_1_)(ffn_ln_out);
  Tensor ffn_out = (*ffn_2_)(gelu(ffn_h));
  return add(x1, ffn_out);
}

// --- MiniGPT ---

MiniGPT::MiniGPT(const MiniGPTConfig& config) : config_(config) {
  int64_t hdim = config_.head_dim();
  int64_t ffn = config_.ffn_hidden();
  (void)hdim;

  tok_embed_ = std::make_shared<Embedding>(config_.vocab_size, config_.dim);
  pos_embed_ = std::make_shared<Embedding>(config_.seq_len, config_.dim);
  register_module("tok_embed", tok_embed_);
  register_module("pos_embed", pos_embed_);

  blocks_.reserve(static_cast<size_t>(config_.num_layers));
  for (int64_t i = 0; i < config_.num_layers; ++i) {
    auto block = std::make_shared<TransformerBlock>(config_.dim, config_.num_heads, ffn);
    blocks_.push_back(block);
    register_module("block_" + std::to_string(i), block);
  }
  final_ln_ = std::make_shared<LayerNorm>(config_.dim);
  lm_head_ = std::make_shared<Linear>(config_.dim, config_.vocab_size, false);
  register_module("final_ln", final_ln_);
  register_module("lm_head", lm_head_);
}

Tensor MiniGPT::forward(const Tensor& token_ids) {
  if (token_ids.dtype() != DType::Int64 || token_ids.dim() != 1)
    throw std::invalid_argument("MiniGPT::forward: token_ids must be 1D int64");
  const int64_t T = token_ids.shape()[0];
  if (T > config_.seq_len)
    throw std::invalid_argument("MiniGPT::forward: token_ids length exceeds config.seq_len");

  Tensor x_tok = (*tok_embed_)(token_ids);

  Tensor pos_ids({T}, DType::Int64, Device::cpu(), false);
  int64_t* pp = pos_ids.data_int64();
  for (int64_t i = 0; i < T; ++i) pp[i] = i;
  Tensor x_pos = (*pos_embed_)(pos_ids);

  Tensor x = add(x_tok, x_pos);
  for (int64_t i = 0; i < config_.num_layers; ++i) {
    x = (*blocks_[static_cast<size_t>(i)])(x);
  }
  x = (*final_ln_)(x);
  return (*lm_head_)(x);
}

}  // namespace models
}  // namespace llm

// --- Minimal train loop (built as minigpt executable) ---

#ifdef LLM_MINIGPT_MAIN

#include <llm/optim.hpp>
#include <llm/checkpoint.hpp>

int main() {
  using namespace llm;
  using namespace llm::models;

  seed(42);
  MiniGPTConfig config;
  config.dim = 32;
  config.num_heads = 2;
  config.num_layers = 2;
  config.vocab_size = 128;
  config.seq_len = 16;

  MiniGPT model(config);
  model.train();
  auto params = model.parameters();
  AdamW optimizer(params, 1e-3f, 0.9f, 0.98f, 1e-5f, 0.01f);

  const int64_t T = 8;
  Tensor token_ids({T}, DType::Int64, Device::cpu(), false);
  int64_t* pid = token_ids.data_int64();
  for (int64_t i = 0; i < T; ++i) pid[i] = i % config.vocab_size;

  Tensor target_ids({T}, DType::Int64, Device::cpu(), false);
  int64_t* pt = target_ids.data_int64();
  for (int64_t i = 0; i < T; ++i) pt[i] = (i + 1) % config.vocab_size;

  std::cout << "MiniGPT: " << config.num_layers << " layers, dim=" << config.dim
            << ", vocab=" << config.vocab_size << ", seq_len=" << T << std::endl;

  for (int step = 0; step < 5; ++step) {
    optimizer.zero_grad();
    Tensor logits = model.forward(token_ids);
    Tensor loss = cross_entropy(logits, target_ids);
    loss.backward();
    optimizer.step();
    std::cout << "  step " << step << " loss " << loss.data_float()[0] << std::endl;
  }

  std::cout << "Done." << std::endl;
  return 0;
}

#endif
