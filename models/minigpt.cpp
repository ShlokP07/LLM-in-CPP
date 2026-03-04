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
  register_module("q_proj", std::make_shared<Linear>(dim, dim));
  register_module("k_proj", std::make_shared<Linear>(dim, dim));
  register_module("v_proj", std::make_shared<Linear>(dim, dim));
  register_module("out_proj", std::make_shared<Linear>(dim, dim));
}

Tensor MultiHeadAttention::operator()(const Tensor& x, bool causal) {
  const int64_t T = x.shape()[0];
  Tensor Q = (*std::static_pointer_cast<Linear>(modules_.at("q_proj")))(x);
  Tensor K = (*std::static_pointer_cast<Linear>(modules_.at("k_proj")))(x);
  Tensor V = (*std::static_pointer_cast<Linear>(modules_.at("v_proj")))(x);

  Tensor out = Tensor::zeros({T, dim_}, DType::Float32, Device::cpu(), true);
  for (int64_t h = 0; h < num_heads_; ++h) {
    Tensor W_h = make_head_W(dim_, head_dim_, h);
    Tensor U_h = make_head_U(dim_, head_dim_, h);
    Tensor Q_h = matmul(Q, W_h);
    Tensor K_h = matmul(K, W_h);
    Tensor V_h = matmul(V, W_h);
    Tensor out_h = scaled_dot_product_attention(Q_h, K_h, V_h, causal);
    out = add(out, matmul(out_h, U_h));
  }
  return (*std::static_pointer_cast<Linear>(modules_.at("out_proj")))(out);
}

// --- TransformerBlock ---

TransformerBlock::TransformerBlock(int64_t dim, int64_t num_heads, int64_t ffn_dim)
    : dim_(dim), ffn_dim_(ffn_dim) {
  register_module("attn_ln", std::make_shared<LayerNorm>(dim));
  register_module("attn", std::make_shared<MultiHeadAttention>(dim, num_heads));
  register_module("ffn_ln", std::make_shared<LayerNorm>(dim));
  register_module("ffn_1", std::make_shared<Linear>(dim, ffn_dim));
  register_module("ffn_2", std::make_shared<Linear>(ffn_dim, dim));
}

Tensor TransformerBlock::operator()(const Tensor& x) {
  Tensor attn_ln_out = (*std::static_pointer_cast<LayerNorm>(modules_.at("attn_ln")))(x);
  Tensor attn_out = (*std::static_pointer_cast<MultiHeadAttention>(modules_.at("attn")))(attn_ln_out, true);
  Tensor x1 = add(x, attn_out);

  Tensor ffn_ln_out = (*std::static_pointer_cast<LayerNorm>(modules_.at("ffn_ln")))(x1);
  Tensor ffn_h = (*std::static_pointer_cast<Linear>(modules_.at("ffn_1")))(ffn_ln_out);
  Tensor ffn_out = (*std::static_pointer_cast<Linear>(modules_.at("ffn_2")))(gelu(ffn_h));
  return add(x1, ffn_out);
}

// --- MiniGPT ---

MiniGPT::MiniGPT(const MiniGPTConfig& config) : config_(config) {
  int64_t hdim = config_.head_dim();
  int64_t ffn = config_.ffn_hidden();

  register_module("embed", std::make_shared<Embedding>(config_.vocab_size, config_.dim));
  for (int64_t i = 0; i < config_.num_layers; ++i) {
    register_module("block_" + std::to_string(i),
                    std::make_shared<TransformerBlock>(config_.dim, config_.num_heads, ffn));
  }
  register_module("final_ln", std::make_shared<LayerNorm>(config_.dim));
  register_module("lm_head", std::make_shared<Linear>(config_.dim, config_.vocab_size, false));
}

Tensor MiniGPT::forward(const Tensor& token_ids) {
  if (token_ids.dtype() != DType::Int64 || token_ids.dim() != 1)
    throw std::invalid_argument("MiniGPT::forward: token_ids must be 1D int64");

  Tensor x = (*std::static_pointer_cast<Embedding>(modules_.at("embed")))(token_ids);
  for (int64_t i = 0; i < config_.num_layers; ++i) {
    x = (*std::static_pointer_cast<TransformerBlock>(modules_.at("block_" + std::to_string(i))))(x);
  }
  x = (*std::static_pointer_cast<LayerNorm>(modules_.at("final_ln")))(x);
  return (*std::static_pointer_cast<Linear>(modules_.at("lm_head")))(x);
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
