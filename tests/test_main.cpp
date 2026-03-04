/**
 * Test entry point for basic Tensor sanity checks and autograd gradient checks.
 *
 * Tests are grouped into: version/tensor basics, grad checks, autograd (no_grad, detach),
 * module/parameters, RNG/init, NN layers (Linear, LayerNorm, Embedding, Dropout, GELU,
 * Softmax, CrossEntropy), optimizers (SGD, AdamW, clip_grad_norm), and edge-case tests.
 * Edge-case tests cover single-element tensors, empty/null grads, detach stopping flow,
 * state_dict, and layer behavior with constant input, single row, or single class.
 */

#include <llm/llm.hpp>
#include <llm/ops.hpp>
#include <llm/autograd.hpp>
#include <llm/module.hpp>
#include <llm/init.hpp>
#include <llm/nn.hpp>
#include <llm/optim.hpp>
#include <llm/data.hpp>
#include <llm/checkpoint.hpp>

#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>

using llm::DType;
using llm::Device;
using llm::Tensor;
using llm::Module;
using llm::Parameter;
using llm::add;
using llm::mul;
using llm::sub;
using llm::div;
using llm::neg;
using llm::sum;
using llm::mean;
using llm::exp;
using llm::max;
using llm::matmul;
using llm::transpose;
using llm::gather;
using llm::seed;
using llm::uniform_;
using llm::normal_;
using llm::xavier_uniform_;
using llm::zeros_;
using llm::Linear;
using llm::Embedding;
using llm::Dropout;
using llm::GELU;
using llm::gelu;
using llm::softmax;
using llm::log_softmax;
using llm::cross_entropy;
using llm::CrossEntropyLoss;
using llm::LayerNorm;
using llm::scaled_dot_product_attention;
using llm::ScaledDotProductAttention;
using llm::SGD;
using llm::AdamW;
using llm::clip_grad_norm_;
using llm::Dataset;
using llm::TensorDataset;
using llm::DataLoader;
using llm::Sample;
using llm::save_tensor;
using llm::load_tensor;
using llm::save_state_dict;
using llm::load_state_dict;

// Verify that version() returns some non-null, non-empty string.
static void test_version() {
  const char* v = llm::version();
  assert(v != nullptr);
  assert(v[0] != '\0');
}

// Construct a small tensor and check basic metadata: shape, numel, dtype, device.
static void test_tensor_basic_shape() {
  std::vector<long long> shape = {2, 3};
  Tensor t(shape, DType::Float32, Device::cpu(), /*requires_grad=*/false);

  assert(t.dim() == 2);
  assert(t.numel() == 6);
  assert(t.dtype() == DType::Float32);
  assert(t.device().type == llm::DeviceType::CPU);

  const auto& s = t.shape();
  assert(s.size() == 2);
  assert(s[0] == 2);
  assert(s[1] == 3);
}

// Check that zeros() constructs a tensor of the requested size and type.
static void test_tensor_zeros() {
  std::vector<long long> shape = {4};
  Tensor t = Tensor::zeros(shape, DType::Int64, Device::cpu(), /*requires_grad=*/true);

  assert(t.numel() == 4);
  assert(t.dtype() == DType::Int64);
  assert(t.requires_grad());
}

// Check that from_data enforces matching shape and data size and copies data.
static void test_tensor_from_data() {
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
  std::vector<long long> shape = {4};

  Tensor t = Tensor::from_data(data, shape, /*requires_grad=*/false);
  assert(t.numel() == 4);

  const float* p = t.data_float();
  assert(p != nullptr);
  for (int i = 0; i < 4; ++i) {
    assert(p[i] == data[i]);
  }

  // Mismatched shape should throw.
  bool threw = false;
  try {
    Tensor::from_data(data, std::vector<long long>{2, 2, 2}, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

// Check that reshape preserves numel and shares storage.
static void test_tensor_reshape() {
  std::vector<float> data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<long long> shape = {2, 3};

  Tensor t = Tensor::from_data(data, shape, /*requires_grad=*/false);
  Tensor r = t.reshape({3, 2});

  assert(r.numel() == t.numel());
  assert(r.shape().size() == 2);
  assert(r.shape()[0] == 3);
  assert(r.shape()[1] == 2);

  // Reshape with incompatible total size should throw.
  bool threw = false;
  try {
    (void)t.reshape({7});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

// --- Tensor & ops edge cases (help readers see scalar, single-element, and reshape behavior) ---

// Single-element tensor behaves like a scalar: from_data one value, sum gives same value, backward fills grad.
static void test_tensor_single_element_scalar_like() {
  Tensor t = Tensor::from_data({42.f}, {1}, true);
  assert(t.numel() == 1);
  Tensor s = sum(t);
  s.backward();
  assert(t.grad() != nullptr);
  assert(std::fabs(t.grad()->data_float()[0] - 1.f) < 1e-5f);
}

// Reshape to (1, N) or (N, 1): numel preserved, strides/shape correct.
static void test_tensor_reshape_to_one_dim() {
  Tensor t = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {4}, false);
  Tensor r1 = t.reshape({1, 4});
  Tensor r2 = t.reshape({4, 1});
  assert(r1.numel() == 4 && r2.numel() == 4);
  assert(r1.shape()[0] == 1 && r1.shape()[1] == 4);
  assert(r2.shape()[0] == 4 && r2.shape()[1] == 1);
}

// Matmul of (1,1) x (1,1): scalar-like matrix multiply.
static void test_matmul_1x1_edge() {
  Tensor a = Tensor::from_data({3.f}, {1, 1}, true);
  Tensor b = Tensor::from_data({7.f}, {1, 1}, true);
  Tensor c = matmul(a, b);
  assert(c.numel() == 1);
  assert(std::fabs(c.data_float()[0] - 21.f) < 1e-5f);
  Tensor loss = sum(c);
  loss.backward();
  assert(a.grad() != nullptr && b.grad() != nullptr);
  assert(std::fabs(a.grad()->data_float()[0] - 7.f) < 1e-5f);
  assert(std::fabs(b.grad()->data_float()[0] - 3.f) < 1e-5f);
}

// Same tensor used twice in the graph: gradient accumulates (d/da (a+a) = 2).
static void test_backward_grad_accumulation() {
  Tensor a = Tensor::from_data({1.f, 2.f}, {2}, true);
  Tensor b = add(a, a);  // b = a + a
  Tensor loss = sum(b);
  loss.backward();
  assert(a.grad() != nullptr);
  assert(std::fabs(a.grad()->data_float()[0] - 2.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[1] - 2.f) < 1e-5f);
}

// Backward through detach: gradient does not flow to leaves before the detach.
// (Detached tensor has requires_grad=false, so grad may not be stored on c; the important part is a gets nothing.)
static void test_detach_stops_gradient_flow() {
  Tensor a = Tensor::from_data({1.f, 2.f}, {2}, true);
  Tensor b = add(a, Tensor::from_data({0.f, 0.f}, {2}, false));
  Tensor c = b.detach();
  Tensor loss = sum(c);
  loss.backward();
  assert(a.grad() == nullptr);  // detach breaks the chain; gradient does not reach a
}

// Mean over a single row: shape (1, D) -> mean(dim=1) -> (1,).
static void test_mean_single_row() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f}, {1, 3}, false);
  Tensor m = mean(a, 1, false);
  assert(m.numel() == 1);
  assert(std::fabs(m.data_float()[0] - 2.f) < 1e-5f);
}

// Reshape should not detach: gradients through reshape must flow back to the original.
static void test_reshape_preserves_autograd() {
  Tensor t = Tensor::from_data({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, {2, 3}, true);
  Tensor r = t.reshape({3, 2});
  Tensor loss = sum(r);
  loss.backward();
  assert(t.grad() != nullptr);
  const float* g = t.grad()->data_float();
  for (int i = 0; i < 6; ++i) {
    assert(std::fabs(g[i] - 1.f) < 1e-5f);
  }
}

// sum(x, dim) should be differentiable (2D) and broadcast grad along reduced axis.
static void test_sum_dim_backward() {
  // x shape (2,3); sum over dim=1 -> (2)
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, true);
  Tensor s = sum(x, 1, false);  // shape (2)
  Tensor loss = sum(s);
  loss.backward();
  assert(x.grad() != nullptr);
  const float* g = x.grad()->data_float();
  for (int i = 0; i < 6; ++i) {
    assert(std::fabs(g[i] - 1.f) < 1e-5f);
  }
}

// mean(x, dim) should be differentiable (2D) and scale grad by 1/denom.
static void test_mean_dim_backward() {
  // x shape (2,3); mean over dim=1 -> (2); loss=sum(mean) => each element grad = 1/3
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, true);
  Tensor m = mean(x, 1, false);
  Tensor loss = sum(m);
  loss.backward();
  assert(x.grad() != nullptr);
  const float* g = x.grad()->data_float();
  for (int i = 0; i < 6; ++i) {
    assert(std::fabs(g[i] - (1.f / 3.f)) < 1e-5f);
  }
}

// Finite-difference gradient check: compare autograd grad with numerical grad.
static void grad_check_add() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f}, {3}, true);
  Tensor b = Tensor::from_data({0.5f, 1.f, 1.5f}, {3}, true);
  Tensor c = add(a, b);
  Tensor loss = sum(c);
  loss.backward();

  assert(a.grad() != nullptr);
  assert(b.grad() != nullptr);
  const float eps = 1e-4f;
  for (int i = 0; i < 3; ++i) {
    assert(std::fabs(a.grad()->data_float()[i] - 1.f) < 1e-5f);
    assert(std::fabs(b.grad()->data_float()[i] - 1.f) < 1e-5f);
  }
}

static void grad_check_mul() {
  Tensor a = Tensor::from_data({1.f, 2.f}, {2}, true);
  Tensor b = Tensor::from_data({3.f, 4.f}, {2}, true);
  Tensor c = mul(a, b);
  Tensor loss = sum(c);
  loss.backward();

  assert(a.grad() != nullptr);
  assert(b.grad() != nullptr);
  assert(std::fabs(a.grad()->data_float()[0] - 3.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[1] - 4.f) < 1e-5f);
  assert(std::fabs(b.grad()->data_float()[0] - 1.f) < 1e-5f);
  assert(std::fabs(b.grad()->data_float()[1] - 2.f) < 1e-5f);
}

static void grad_check_sum() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f}, {3}, true);
  Tensor s = sum(a);
  s.backward();

  assert(a.grad() != nullptr);
  for (int i = 0; i < 3; ++i)
    assert(std::fabs(a.grad()->data_float()[i] - 1.f) < 1e-5f);
}

static void grad_check_matmul() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, true);
  Tensor b = Tensor::from_data({1.f, 0.f, 0.f, 1.f}, {2, 2}, true);
  Tensor c = matmul(a, b);
  Tensor loss = sum(c);
  loss.backward();

  assert(a.grad() != nullptr);
  assert(b.grad() != nullptr);
  // d(sum(A@B))/dA = ones, d(sum(A@B))/dB = ones (for B=I, A@I=A, sum(A)=sum of A elements, grad w.r.t. A is ones)
  assert(std::fabs(a.grad()->data_float()[0] - 1.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[1] - 1.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[2] - 1.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[3] - 1.f) < 1e-5f);
}

static void grad_check_transpose() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, true);
  Tensor t = transpose(a);
  Tensor loss = sum(t);
  loss.backward();

  assert(a.grad() != nullptr);
  for (int i = 0; i < 4; ++i)
    assert(std::fabs(a.grad()->data_float()[i] - 1.f) < 1e-5f);
}

// Broadcasting add for bias: (M,N) + (N).
static void test_add_bias_broadcast_grad() {
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, true);
  Tensor b = Tensor::from_data({0.5f, 1.f, 1.5f}, {3}, true);
  Tensor y = add(x, b);  // shape (2,3)
  Tensor loss = sum(y);
  loss.backward();

  // d(loss)/dx = ones, d(loss)/db = number of rows (2).
  assert(x.grad() != nullptr);
  assert(b.grad() != nullptr);
  for (int i = 0; i < 6; ++i) {
    assert(std::fabs(x.grad()->data_float()[i] - 1.f) < 1e-5f);
  }
  for (int j = 0; j < 3; ++j) {
    assert(std::fabs(b.grad()->data_float()[j] - 2.f) < 1e-5f);
  }
}

// Broadcasting mul for bias: (M,N) * (N).
static void test_mul_bias_broadcast_grad() {
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, true);
  Tensor b = Tensor::from_data({2.f, 3.f}, {2}, true);
  Tensor y = mul(x, b);  // shape (2,2)
  Tensor loss = sum(y);
  loss.backward();

  // y = x * b, loss = sum(y)
  // dL/dx[i,j] = b[j], dL/db[j] = sum_i x[i,j].
  assert(x.grad() != nullptr);
  assert(b.grad() != nullptr);
  assert(std::fabs(x.grad()->data_float()[0] - 2.f) < 1e-5f);
  assert(std::fabs(x.grad()->data_float()[1] - 3.f) < 1e-5f);
  assert(std::fabs(x.grad()->data_float()[2] - 2.f) < 1e-5f);
  assert(std::fabs(x.grad()->data_float()[3] - 3.f) < 1e-5f);

  float gb0 = b.grad()->data_float()[0];
  float gb1 = b.grad()->data_float()[1];
  // column 0: x[0,0] + x[1,0] = 1 + 3 = 4
  // column 1: x[0,1] + x[1,1] = 2 + 4 = 6
  assert(std::fabs(gb0 - 4.f) < 1e-5f);
  assert(std::fabs(gb1 - 6.f) < 1e-5f);
}

// Basic checks for sub/div/neg/mean/exp/max shapes and values (no full grad checks yet).
static void test_elementwise_and_reductions_shapes() {
  Tensor a = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, false);
  Tensor b = Tensor::from_data({4.f, 3.f, 2.f, 1.f}, {2, 2}, false);

  Tensor d = sub(a, b);
  assert(d.shape().size() == 2);
  assert(d.shape()[0] == 2 && d.shape()[1] == 2);

  Tensor q = div(a, b);
  assert(q.shape().size() == 2);
  assert(q.shape()[0] == 2 && q.shape()[1] == 2);

  Tensor n = neg(a);
  assert(n.shape().size() == 2);
  assert(std::fabs(n.data_float()[0] + 1.f) < 1e-5f);

  Tensor m0 = mean(a, 0, false);
  assert(m0.shape().size() == 1 && m0.shape()[0] == 2);

  Tensor m1 = mean(a, 1, false);
  assert(m1.shape().size() == 1 && m1.shape()[0] == 2);

  Tensor e = exp(a);
  assert(e.shape().size() == 2);

  Tensor mx0 = max(a, 0, false);
  assert(mx0.shape().size() == 1 && mx0.shape()[0] == 2);

  Tensor mx1 = max(a, 1, false);
  assert(mx1.shape().size() == 1 && mx1.shape()[0] == 2);
}

static void test_no_grad() {
  Tensor a = Tensor::from_data({1.f}, {1}, true);
  Tensor b = Tensor::from_data({2.f}, {1}, true);
  Tensor c;
  {
    llm::NoGradGuard guard;  // ops in this scope don't build the graph
    c = add(a, b);
  }
  assert(!c.requires_grad());
  assert(c.grad_fn() == nullptr);
  assert(c.numel() == 1);
  assert(std::fabs(c.data_float()[0] - 3.f) < 1e-5f);
}

static void test_detach() {
  Tensor a = Tensor::from_data({1.f, 2.f}, {2}, true);
  Tensor b = add(a, a);
  Tensor c = b.detach();  // same data, no grad_fn (stops backward here)
  assert(!c.requires_grad());
  assert(c.grad_fn() == nullptr);
  assert(c.numel() == 2);
  assert(std::fabs(c.data_float()[0] - 2.f) < 1e-5f);
}

// Simple test modules to verify parameter registration and train/eval propagation.
class LeafModule : public Module {
public:
  LeafModule() {
    register_parameter("w", Parameter::zeros({1}));
  }
};

class ParentModule : public Module {
public:
  ParentModule() {
    register_parameter("b", Parameter::zeros({1}));
    child = std::make_shared<LeafModule>();
    register_module("child", child);
  }

  std::shared_ptr<LeafModule> child;
};

static void test_module_parameters_and_modes() {
  ParentModule m;

  // parameters() should include both parent and child parameters.
  auto params = m.parameters();
  assert(params.size() == 2);
  for (Parameter* p : params) {
    assert(p != nullptr);
    assert(p->requires_grad());
  }

  // train()/eval() should propagate to submodules.
  assert(m.is_training());
  assert(m.child->is_training());

  m.eval();
  assert(!m.is_training());
  assert(!m.child->is_training());

  m.train();
  assert(m.is_training());
  assert(m.child->is_training());
}

// --- RNG & init tests ---

static void test_seed_uniform_determinism_and_range() {
  Tensor t1({100}, DType::Float32, Device::cpu(), false);
  Tensor t2({100}, DType::Float32, Device::cpu(), false);

  seed(42);
  uniform_(t1, 0.f, 1.f);

  seed(42);
  uniform_(t2, 0.f, 1.f);

  for (int64_t i = 0; i < 100; ++i) {
    assert(t1.data_float()[i] == t2.data_float()[i]);
    assert(t1.data_float()[i] >= 0.f && t1.data_float()[i] <= 1.f);
  }
}

static void test_uniform_range() {
  Tensor t({1000}, DType::Float32, Device::cpu(), false);
  seed(123);
  uniform_(t, -2.5f, 3.5f);

  for (int64_t i = 0; i < t.numel(); ++i) {
    float v = t.data_float()[i];
    assert(v >= -2.5f && v <= 3.5f);
  }
}

static void test_seed_normal_determinism() {
  Tensor t1({50}, DType::Float32, Device::cpu(), false);
  Tensor t2({50}, DType::Float32, Device::cpu(), false);

  seed(99);
  normal_(t1, 0.f, 1.f);

  seed(99);
  normal_(t2, 0.f, 1.f);

  for (int64_t i = 0; i < 50; ++i)
    assert(t1.data_float()[i] == t2.data_float()[i]);
}

static void test_normal_approximate_stats() {
  Tensor t({2000}, DType::Float32, Device::cpu(), false);
  seed(1);
  normal_(t, 5.f, 2.f);

  float sum = 0.f, sum_sq = 0.f;
  for (int64_t i = 0; i < t.numel(); ++i) {
    float v = t.data_float()[i];
    sum += v;
    sum_sq += v * v;
  }
  float mean = sum / static_cast<float>(t.numel());
  float var = sum_sq / static_cast<float>(t.numel()) - mean * mean;
  assert(std::fabs(mean - 5.f) < 0.2f);
  assert(std::fabs(std::sqrt(var) - 2.f) < 0.2f);
}

static void test_zeros_float32() {
  Tensor t({4, 5}, DType::Float32, Device::cpu(), false);
  uniform_(t, 1.f, 2.f);  // non-zero
  zeros_(t);
  for (int64_t i = 0; i < t.numel(); ++i)
    assert(t.data_float()[i] == 0.f);
}

static void test_zeros_int64() {
  Tensor t({3, 2}, DType::Int64, Device::cpu(), false);
  int64_t* p = t.data_int64();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = 999;
  zeros_(t);
  for (int64_t i = 0; i < t.numel(); ++i)
    assert(t.data_int64()[i] == 0);
}

static void test_xavier_uniform_range() {
  Tensor t({10, 20}, DType::Float32, Device::cpu(), false);
  seed(7);
  xavier_uniform_(t);

  float limit = std::sqrt(6.f / (10.f + 20.f));
  for (int64_t i = 0; i < t.numel(); ++i) {
    float v = t.data_float()[i];
    assert(v >= -limit && v <= limit);
  }
}

// --- Linear layer tests ---

static void test_linear_forward_shape() {
  Linear linear(3, 4, /*bias=*/true);
  Tensor x({2, 3}, DType::Float32, Device::cpu(), false);
  uniform_(x, 0.f, 1.f);

  Tensor y = linear(x);
  assert(y.dim() == 2);
  assert(y.shape()[0] == 2);
  assert(y.shape()[1] == 4);
}

static void test_linear_no_bias_forward_shape() {
  Linear linear(5, 2, /*bias=*/false);
  Tensor x({1, 5}, DType::Float32, Device::cpu(), false);
  zeros_(x);
  Tensor y = linear(x);
  assert(y.shape().size() == 2);
  assert(y.shape()[0] == 1);
  assert(y.shape()[1] == 2);
}

static void test_linear_backward_weight_and_bias() {
  seed(0);
  Linear linear(2, 3, /*bias=*/true);
  Tensor x = Tensor::from_data({1.f, 0.f, 0.f, 1.f}, {2, 2}, true);  // (2, 2)
  Tensor y = linear(x);
  Tensor loss = sum(y);
  loss.backward();

  auto params = linear.parameters();
  assert(params.size() == 2);  // weight, bias
  bool has_weight_grad = false, has_bias_grad = false;
  for (Parameter* p : params) {
    assert(p->grad() != nullptr);
    const auto& sh = p->grad()->shape();
    if (sh.size() == 2 && sh[0] == 3 && sh[1] == 2) has_weight_grad = true;
    if (sh.size() == 1 && sh[0] == 3) has_bias_grad = true;
  }
  assert(has_weight_grad && has_bias_grad);
}

// --- LayerNorm tests ---

static void test_layernorm_forward_shape() {
  LayerNorm ln(4, 1e-5f);
  Tensor x({2, 4}, DType::Float32, Device::cpu(), false);
  uniform_(x, 0.f, 1.f);
  Tensor y = ln(x);
  assert(y.dim() == 2);
  assert(y.shape()[0] == 2);
  assert(y.shape()[1] == 4);
}

static void test_layernorm_grad_check() {
  seed(0);
  LayerNorm ln(3, 1e-5f);
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, true);
  Tensor y = ln(x);
  Tensor loss = sum(y);
  loss.backward();

  assert(x.grad() != nullptr);
  assert((x.grad()->shape() == std::vector<int64_t>{2, 3}));

  auto params = ln.parameters();
  assert(params.size() == 2);
  for (Parameter* p : params) {
    assert(p->grad() != nullptr);
    assert((p->grad()->shape() == std::vector<int64_t>{3}));
  }

  // Numerical gradient check: perturb x[0,0] by eps and compare d(loss)/dx[0,0]
  const float eps = 1e-4f;
  float loss_plus = 0.f, loss_minus = 0.f;
  {
    llm::NoGradGuard guard;
    Tensor xp = Tensor::from_data({1.f + eps, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, false);
    Tensor yp = ln(xp);
    loss_plus = sum(yp).data_float()[0];
  }
  {
    llm::NoGradGuard guard;
    Tensor xm = Tensor::from_data({1.f - eps, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, false);
    Tensor ym = ln(xm);
    loss_minus = sum(ym).data_float()[0];
  }
  float numerical = (loss_plus - loss_minus) / (2.f * eps);
  float analytical = x.grad()->data_float()[0];
  assert(std::fabs(numerical - analytical) < 0.02f);
}

// --- gather & Embedding tests ---

static void test_gather_forward_shape_and_values() {
  Tensor weight = Tensor::from_data(
      {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f},
      {4, 3}, false);
  Tensor indices({3}, DType::Int64, Device::cpu(), false);
  indices.data_int64()[0] = 0;
  indices.data_int64()[1] = 2;
  indices.data_int64()[2] = 1;

  Tensor out = gather(weight, indices);
  assert(out.shape().size() == 2);
  assert(out.shape()[0] == 3);
  assert(out.shape()[1] == 3);
  assert(std::fabs(out.data_float()[0] - 0.f) < 1e-5f);
  assert(std::fabs(out.data_float()[1] - 1.f) < 1e-5f);
  assert(std::fabs(out.data_float()[2] - 2.f) < 1e-5f);
  assert(std::fabs(out.data_float()[3] - 6.f) < 1e-5f);
  assert(std::fabs(out.data_float()[4] - 7.f) < 1e-5f);
  assert(std::fabs(out.data_float()[5] - 8.f) < 1e-5f);
  assert(std::fabs(out.data_float()[6] - 3.f) < 1e-5f);
  assert(std::fabs(out.data_float()[7] - 4.f) < 1e-5f);
  assert(std::fabs(out.data_float()[8] - 5.f) < 1e-5f);
}

static void test_gather_backward() {
  Tensor weight = Tensor::from_data({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3}, true);
  Tensor indices({2}, DType::Int64, Device::cpu(), false);
  indices.data_int64()[0] = 0;
  indices.data_int64()[1] = 0;

  Tensor y = gather(weight, indices);
  Tensor loss = sum(y);
  loss.backward();

  assert(weight.grad() != nullptr);
  assert((weight.grad()->shape() == std::vector<int64_t>{2, 3}));
  const float* gw = weight.grad()->data_float();
  for (int64_t j = 0; j < 3; ++j) {
    assert(std::fabs(gw[0 * 3 + j] - 2.f) < 1e-5f);
    assert(std::fabs(gw[1 * 3 + j] - 0.f) < 1e-5f);
  }
}

static void test_embedding_forward_shape() {
  seed(0);
  Embedding emb(10, 4);
  Tensor indices({3}, DType::Int64, Device::cpu(), false);
  indices.data_int64()[0] = 1;
  indices.data_int64()[1] = 5;
  indices.data_int64()[2] = 0;

  Tensor y = emb(indices);
  assert(y.dim() == 2);
  assert(y.shape()[0] == 3);
  assert(y.shape()[1] == 4);
}

static void test_embedding_backward() {
  seed(0);
  Embedding emb(5, 3);
  Tensor indices({2}, DType::Int64, Device::cpu(), false);
  indices.data_int64()[0] = 0;
  indices.data_int64()[1] = 1;

  Tensor y = emb(indices);
  Tensor loss = sum(y);
  loss.backward();

  auto params = emb.parameters();
  assert(params.size() == 1);
  assert(params[0]->grad() != nullptr);
  assert((params[0]->grad()->shape() == std::vector<int64_t>{5, 3}));
}

// --- Dropout tests ---

static Tensor ones_2d(int64_t M, int64_t N) {
  Tensor t({M, N}, DType::Float32, Device::cpu(), false);
  float* p = t.data_float();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = 1.f;
  return t;
}

static void test_dropout_eval_identity() {
  Dropout d(0.5f);
  d.eval();
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, false);
  Tensor y = d(x);
  assert(y.shape() == x.shape());
  for (int i = 0; i < 4; ++i)
    assert(std::fabs(y.data_float()[i] - x.data_float()[i]) < 1e-6f);
}

static void test_dropout_train_deterministic_with_seed() {
  Dropout d(0.25f);
  d.train();
  Tensor x = ones_2d(3, 3);

  seed(123);
  Tensor y1 = d(x);
  seed(123);
  Tensor y2 = d(x);

  for (int64_t i = 0; i < y1.numel(); ++i)
    assert(y1.data_float()[i] == y2.data_float()[i]);
}

static void test_dropout_backward_matches_mask_scale() {
  Dropout d(0.5f);
  d.train();
  Tensor x = ones_2d(2, 4);
  x.set_requires_grad(true);

  seed(7);
  Tensor y = d(x);
  Tensor loss = sum(y);
  loss.backward();

  assert(x.grad() != nullptr);
  // For x = ones, d(sum(y))/dx == mask*scale, which equals y itself.
  for (int64_t i = 0; i < x.numel(); ++i)
    assert(std::fabs(x.grad()->data_float()[i] - y.data_float()[i]) < 1e-6f);
}

// --- GELU tests ---

static void test_gelu_forward_shape() {
  GELU g;
  Tensor x({2, 3}, DType::Float32, Device::cpu(), false);
  uniform_(x, -1.f, 1.f);
  Tensor y = g(x);
  assert(y.shape() == x.shape());
}

static void test_gelu_grad_check() {
  Tensor x = Tensor::from_data({-1.f, 0.f, 1.f}, {3}, true);
  Tensor y = gelu(x);
  Tensor loss = sum(y);
  loss.backward();
  assert(x.grad() != nullptr);

  const float eps = 1e-4f;
  float lp = 0.f, lm = 0.f;
  {
    llm::NoGradGuard guard;
    Tensor xp = Tensor::from_data({-1.f + eps, 0.f, 1.f}, {3}, false);
    lp = sum(gelu(xp)).data_float()[0];
  }
  {
    llm::NoGradGuard guard;
    Tensor xm = Tensor::from_data({-1.f - eps, 0.f, 1.f}, {3}, false);
    lm = sum(gelu(xm)).data_float()[0];
  }
  float num = (lp - lm) / (2.f * eps);
  float ana = x.grad()->data_float()[0];
  assert(std::fabs(num - ana) < 1e-2f);
}

// --- Softmax / LogSoftmax tests ---

static void test_softmax_row_sums_to_one() {
  Tensor x({3, 4}, DType::Float32, Device::cpu(), false);
  uniform_(x, -2.f, 2.f);
  Tensor y = softmax(x);
  for (int64_t i = 0; i < 3; ++i) {
    float s = 0.f;
    for (int64_t j = 0; j < 4; ++j) s += y.data_float()[i * 4 + j];
    assert(std::fabs(s - 1.f) < 1e-4f);
  }
}

static void test_log_softmax_exp_row_sums_to_one() {
  Tensor x({2, 5}, DType::Float32, Device::cpu(), false);
  uniform_(x, -2.f, 2.f);
  Tensor y = log_softmax(x);
  for (int64_t i = 0; i < 2; ++i) {
    float s = 0.f;
    for (int64_t j = 0; j < 5; ++j) s += std::exp(y.data_float()[i * 5 + j]);
    assert(std::fabs(s - 1.f) < 1e-4f);
  }
}

static void test_softmax_grad_check_one_element() {
  Tensor x = Tensor::from_data({0.2f, -0.1f, 0.3f}, {1, 3}, true);
  Tensor w = Tensor::from_data({1.0f, -2.0f, 0.5f}, {1, 3}, false);
  Tensor y = softmax(x);
  Tensor loss = sum(mul(y, w));
  loss.backward();
  assert(x.grad() != nullptr);

  const float eps = 1e-4f;
  float lp = 0.f, lm = 0.f;
  {
    llm::NoGradGuard guard;
    Tensor xp = Tensor::from_data({0.2f + eps, -0.1f, 0.3f}, {1, 3}, false);
    lp = sum(mul(softmax(xp), w)).data_float()[0];
  }
  {
    llm::NoGradGuard guard;
    Tensor xm = Tensor::from_data({0.2f - eps, -0.1f, 0.3f}, {1, 3}, false);
    lm = sum(mul(softmax(xm), w)).data_float()[0];
  }
  float num = (lp - lm) / (2.f * eps);
  float ana = x.grad()->data_float()[0];
  assert(std::fabs(num - ana) < 1e-2f);
}

static void test_log_softmax_grad_check_one_element() {
  Tensor x = Tensor::from_data({0.2f, -0.1f, 0.3f}, {1, 3}, true);
  Tensor w = Tensor::from_data({-1.0f, 1.5f, 0.25f}, {1, 3}, false);
  Tensor y = log_softmax(x);
  Tensor loss = sum(mul(y, w));
  loss.backward();
  assert(x.grad() != nullptr);

  const float eps = 1e-4f;
  float lp = 0.f, lm = 0.f;
  {
    llm::NoGradGuard guard;
    Tensor xp = Tensor::from_data({0.2f + eps, -0.1f, 0.3f}, {1, 3}, false);
    lp = sum(mul(log_softmax(xp), w)).data_float()[0];
  }
  {
    llm::NoGradGuard guard;
    Tensor xm = Tensor::from_data({0.2f - eps, -0.1f, 0.3f}, {1, 3}, false);
    lm = sum(mul(log_softmax(xm), w)).data_float()[0];
  }
  float num = (lp - lm) / (2.f * eps);
  float ana = x.grad()->data_float()[0];
  assert(std::fabs(num - ana) < 1e-2f);
}

// --- Scaled dot-product attention tests ---

static void test_attention_forward_shape() {
  // Q, K, V (T, D) -> out (T, D)
  Tensor Q = Tensor::from_data({1.f, 0.f, 0.f, 1.f, 0.f, 0.f}, {2, 3}, false);
  Tensor K = Tensor::from_data({1.f, 0.f, 0.f, 0.f, 1.f, 0.f}, {2, 3}, false);
  Tensor V = Tensor::from_data({1.f, 0.f, 0.f, 0.f, 1.f, 0.f}, {2, 3}, false);
  Tensor out = scaled_dot_product_attention(Q, K, V, /*causal=*/true);
  assert(out.shape().size() == 2);
  assert(out.shape()[0] == 2);
  assert(out.shape()[1] == 3);
}

static void test_attention_causal_mask() {
  // With causal mask, position 1 cannot attend to position 0, so output row 1 is determined only by row 1 of Q,K,V.
  // Use simple inputs: Q=K=V identity-like so attn is close to identity; with causal, row 0 gets row 0 of V, row 1 gets only row 1.
  seed(0);
  Tensor Q = Tensor::from_data({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, false);
  Tensor K = Tensor::from_data({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, false);
  Tensor V = Tensor::from_data({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, false);
  Tensor out = scaled_dot_product_attention(Q, K, V, true);
  // Each row of out should be a convex combination of rows of V (due to softmax). Causal: row i only sees rows 0..i of K.
  for (int64_t i = 0; i < 3; ++i) {
    float row_sum = 0.f;
    for (int64_t j = 0; j < 3; ++j) row_sum += out.data_float()[i * 3 + j];
    assert(std::isfinite(row_sum));
  }
}

static void test_attention_backward() {
  Tensor Q = Tensor::from_data({0.5f, -0.2f, 0.1f, 0.3f, 0.4f, -0.1f}, {2, 3}, true);
  Tensor K = Tensor::from_data({0.2f, 0.3f, 0.1f, -0.1f, 0.5f, 0.2f}, {2, 3}, true);
  Tensor V = Tensor::from_data({1.f, 0.f, 0.f, 0.f, 1.f, 0.f}, {2, 3}, true);
  Tensor out = scaled_dot_product_attention(Q, K, V, true);
  Tensor loss = sum(out);
  loss.backward();
  assert(Q.grad() != nullptr);
  assert(K.grad() != nullptr);
  assert(V.grad() != nullptr);
  assert((Q.grad()->shape() == std::vector<int64_t>{2, 3}));
  assert((K.grad()->shape() == std::vector<int64_t>{2, 3}));
  assert((V.grad()->shape() == std::vector<int64_t>{2, 3}));
}

static void test_attention_module_wrapper() {
  ScaledDotProductAttention attn;
  Tensor Q = Tensor::from_data({1.f, 0.f, 0.f, 1.f}, {2, 2}, false);
  Tensor K = Tensor::from_data({1.f, 0.f, 0.f, 1.f}, {2, 2}, false);
  Tensor V = Tensor::from_data({1.f, 0.f, 0.f, 1.f}, {2, 2}, false);
  Tensor out = attn(Q, K, V, false);
  assert(out.shape()[0] == 2 && out.shape()[1] == 2);
}

// --- Dataset / DataLoader tests ---

static void test_tensor_dataset_size_and_get() {
  const int64_t N = 4;
  const int64_t seq_len = 3;
  Tensor input({N, seq_len}, DType::Int64, Device::cpu(), false);
  Tensor target({N, seq_len}, DType::Int64, Device::cpu(), false);
  int64_t* pi = input.data_int64();
  int64_t* pt = target.data_int64();
  for (int64_t i = 0; i < N * seq_len; ++i) {
    pi[i] = static_cast<int64_t>(i);
    pt[i] = static_cast<int64_t>(i + 1);
  }
  TensorDataset ds(input, target);
  assert(ds.size() == N);
  assert(ds.seq_len() == seq_len);

  llm::Sample s = ds.get(1);
  Tensor x = s.first;
  Tensor y = s.second;
  assert(x.shape() == std::vector<int64_t>{seq_len});
  assert(y.shape() == std::vector<int64_t>{seq_len});
  assert(x.dtype() == DType::Int64);
  assert(x.data_int64()[0] == 3 && x.data_int64()[1] == 4 && x.data_int64()[2] == 5);
  assert(y.data_int64()[0] == 4 && y.data_int64()[1] == 5 && y.data_int64()[2] == 6);
}

static void test_dataloader_num_batches_and_batch_shape() {
  const int64_t N = 5;
  const int64_t seq_len = 2;
  Tensor input = Tensor::zeros({N, seq_len}, DType::Int64, Device::cpu(), false);
  Tensor target = Tensor::zeros({N, seq_len}, DType::Int64, Device::cpu(), false);
  TensorDataset ds(input, target);
  DataLoader loader(&ds, 2, false);

  assert(loader.num_batches() == 3);
  assert(loader.batch_size() == 2);

  Sample b0 = loader.get_batch(0);
  assert((b0.first.shape() == std::vector<int64_t>{2, seq_len}));
  assert((b0.second.shape() == std::vector<int64_t>{2, seq_len}));

  Sample b_last = loader.get_batch(2);
  assert(b_last.first.shape()[0] == 1);
  assert(b_last.first.shape()[1] == seq_len);
}

static void test_dataloader_shuffle_deterministic() {
  const int64_t N = 6;
  const int64_t seq_len = 2;
  Tensor input({N, seq_len}, DType::Int64, Device::cpu(), false);
  Tensor target({N, seq_len}, DType::Int64, Device::cpu(), false);
  int64_t* pi = input.data_int64();
  for (int64_t i = 0; i < N * seq_len; ++i) pi[i] = i;
  TensorDataset ds(input, target);

  DataLoader loader1(&ds, 2, true, 42u);
  DataLoader loader2(&ds, 2, true, 42u);
  Sample batch1 = loader1.get_batch(0);
  Sample batch2 = loader2.get_batch(0);
  for (int64_t i = 0; i < batch1.first.numel(); ++i)
    assert(batch1.first.data_int64()[i] == batch2.first.data_int64()[i]);
}

// --- Checkpointing tests ---

static void test_save_load_tensor() {
  Tensor t = Tensor::from_data({1.f, 2.f, 3.f, 4.f}, {2, 2}, false);
  const char* path = "test_ckpt_tensor.bin";
  save_tensor(path, t);
  Tensor loaded = load_tensor(path);
  std::remove(path);
  assert(loaded.shape() == t.shape());
  assert(loaded.dtype() == t.dtype());
  for (int64_t i = 0; i < t.numel(); ++i)
    assert(std::fabs(loaded.data_float()[i] - t.data_float()[i]) < 1e-6f);
}

static void test_save_load_state_dict_module_roundtrip() {
  seed(0);
  Linear linear(2, 3, true);
  auto state = linear.state_dict();
  const char* path = "test_ckpt_state.bin";
  save_state_dict(path, state);
  Module::StateDict loaded = load_state_dict(path);
  std::remove(path);
  assert(loaded.size() == state.size());
  for (const auto& kv : state) {
    auto it = loaded.find(kv.first);
    assert(it != loaded.end());
    assert(it->second.shape() == kv.second.shape());
    assert(it->second.dtype() == kv.second.dtype());
    if (kv.second.dtype() == DType::Float32) {
      for (int64_t i = 0; i < kv.second.numel(); ++i)
        assert(std::fabs(it->second.data_float()[i] - kv.second.data_float()[i]) < 1e-6f);
    }
  }
}

static void test_load_state_dict_restores_parameters() {
  seed(1);
  Linear linear1(2, 2, true);
  auto state = linear1.state_dict();
  seed(2);
  Linear linear2(2, 2, true);
  linear2.load_state_dict(state);
  auto params1 = linear1.parameters();
  auto params2 = linear2.parameters();
  assert(params1.size() == params2.size());
  for (size_t i = 0; i < params1.size(); ++i) {
    const float* a = params1[i]->data_float();
    const float* b = params2[i]->data_float();
    for (int64_t j = 0; j < params1[i]->numel(); ++j)
      assert(std::fabs(a[j] - b[j]) < 1e-6f);
  }
}

static void test_adamw_state_dict_roundtrip() {
  Parameter w = Parameter::zeros({2});
  w.set_requires_grad(true);
  std::vector<Parameter*> params = {&w};
  AdamW opt(params, 0.01f);
  w.set_grad(std::make_shared<Tensor>(Tensor::from_data({1.f, 0.f}, {2}, false)));
  opt.step();
  assert(opt.step_count() == 1);
  auto state = opt.state_dict();
  AdamW opt2(params, 0.01f);
  opt2.load_state_dict(state);
  assert(opt2.step_count() == 1);
}

// --- CrossEntropyLoss tests ---

static void test_cross_entropy_known_value() {
  // logits = [0,0] => probs = [0.5,0.5], target=0 => loss = -log(0.5)
  Tensor logits = Tensor::from_data({0.f, 0.f}, {1, 2}, false);
  Tensor targets({1}, DType::Int64, Device::cpu(), false);
  targets.data_int64()[0] = 0;
  Tensor loss = cross_entropy(logits, targets);
  assert(loss.numel() == 1);
  assert(std::fabs(loss.data_float()[0] - 0.69314718f) < 1e-4f);
}

static void test_cross_entropy_grad_check_one_element() {
  Tensor logits = Tensor::from_data({0.2f, -0.1f, 0.3f}, {1, 3}, true);
  Tensor targets({1}, DType::Int64, Device::cpu(), false);
  targets.data_int64()[0] = 2;
  Tensor loss = cross_entropy(logits, targets);
  loss.backward();
  assert(logits.grad() != nullptr);

  const float eps = 1e-4f;
  float lp = 0.f, lm = 0.f;
  {
    llm::NoGradGuard guard;
    Tensor lp_logits = Tensor::from_data({0.2f + eps, -0.1f, 0.3f}, {1, 3}, false);
    lp = cross_entropy(lp_logits, targets).data_float()[0];
  }
  {
    llm::NoGradGuard guard;
    Tensor lm_logits = Tensor::from_data({0.2f - eps, -0.1f, 0.3f}, {1, 3}, false);
    lm = cross_entropy(lm_logits, targets).data_float()[0];
  }
  float num = (lp - lm) / (2.f * eps);
  float ana = logits.grad()->data_float()[0];
  assert(std::fabs(num - ana) < 1e-2f);

  // Gradient across classes sums to ~0 for each sample.
  float sumg = logits.grad()->data_float()[0] + logits.grad()->data_float()[1] + logits.grad()->data_float()[2];
  assert(std::fabs(sumg) < 1e-5f);
}

static void test_cross_entropy_module_wrapper() {
  CrossEntropyLoss cel;
  Tensor logits = Tensor::from_data({0.f, 0.f, 0.f, 0.f}, {2, 2}, false);
  Tensor targets({2}, DType::Int64, Device::cpu(), false);
  targets.data_int64()[0] = 0;
  targets.data_int64()[1] = 1;
  Tensor loss = cel(logits, targets);
  assert(loss.numel() == 1);
}

// --- SGD optimizer tests ---

static void test_sgd_quadratic_descent() {
  // Optimize f(w) = (w - 3)^2 starting from w=0.
  Parameter w = Parameter::zeros({1});
  w.set_requires_grad(true);

  std::vector<Parameter*> params = {&w};
  SGD opt(params, /*lr=*/0.1f, /*weight_decay=*/0.0f);

  Tensor offset = Tensor::from_data({-3.f}, {1}, false);  // w - 3 = w + (-3)

  for (int step = 0; step < 100; ++step) {
    opt.zero_grad();
    Tensor diff = add(w, offset);
    Tensor sq = mul(diff, diff);
    Tensor loss = sum(sq);
    loss.backward();
    opt.step();
  }

  float w_val = w.data_float()[0];
  // w should be close to 3.
  assert(std::fabs(w_val - 3.f) < 1e-2f);
}

static void test_adamw_quadratic_descent() {
  // Same setup: minimize (w - 3)^2 with AdamW.
  Parameter w = Parameter::zeros({1});
  w.set_requires_grad(true);

  std::vector<Parameter*> params = {&w};
  AdamW opt(params, /*lr=*/0.1f, /*beta1=*/0.9f, /*beta2=*/0.999f,
            /*eps=*/1e-8f, /*weight_decay=*/0.0f);

  Tensor offset = Tensor::from_data({-3.f}, {1}, false);

  for (int step = 0; step < 150; ++step) {
    opt.zero_grad();
    Tensor diff = add(w, offset);
    Tensor sq = mul(diff, diff);
    Tensor loss = sum(sq);
    loss.backward();
    opt.step();
  }

  float w_val = w.data_float()[0];
  assert(std::fabs(w_val - 3.f) < 1e-2f);
}

static void test_clip_grad_norm() {
  // Single param with grad [3, 4] -> L2 norm = 5. Clip to 2 -> scale by 2/5.
  Parameter p = Parameter::zeros({2});
  p.set_requires_grad(true);
  std::shared_ptr<Tensor> g = std::make_shared<Tensor>(Tensor::from_data({3.f, 4.f}, {2}, false));
  p.set_grad(g);

  std::vector<Parameter*> params = {&p};
  float norm_before = clip_grad_norm_(params, 2.f);
  assert(std::fabs(norm_before - 5.f) < 1e-5f);

  const float* gw = p.grad()->data_float();
  float norm_after = std::sqrt(gw[0] * gw[0] + gw[1] * gw[1]);
  assert(std::fabs(norm_after - 2.f) < 1e-5f);
  assert(std::fabs(gw[0] - 1.2f) < 1e-5f);
  assert(std::fabs(gw[1] - 1.6f) < 1e-5f);
}

// --- Optimizer edge cases (empty params, null grad, clip no-op) ---

// SGD with no parameters: step() and zero_grad() must not crash.
static void test_sgd_empty_params() {
  std::vector<Parameter*> params;
  SGD opt(params, 0.1f, 0.f);
  opt.zero_grad();
  opt.step();
}

// SGD step when a parameter has no grad (e.g. zero_grad then step without backward): param is skipped, no crash.
static void test_sgd_skips_param_without_grad() {
  Parameter w = Parameter::zeros({1});
  w.set_requires_grad(true);
  std::vector<Parameter*> p = {&w};
  SGD opt(p, 0.1f, 0.f);
  opt.zero_grad();
  opt.step();
  assert(std::fabs(w.data_float()[0]) < 1e-6f);
}

// AdamW step_count() increments each step (helps readers see optimizer state).
static void test_adamw_step_count_increments() {
  Parameter w = Parameter::zeros({1});
  w.set_requires_grad(true);
  std::vector<Parameter*> p = {&w};
  AdamW opt(p, 0.01f);
  assert(opt.step_count() == 0);
  w.set_grad(std::make_shared<Tensor>(Tensor::from_data({1.f}, {1}, false)));
  opt.step();
  assert(opt.step_count() == 1);
  opt.step();
  assert(opt.step_count() == 2);
}

// clip_grad_norm_ with no parameters returns 0 and does nothing.
static void test_clip_grad_norm_empty_params() {
  std::vector<Parameter*> params;
  float n = clip_grad_norm_(params, 1.f);
  assert(n == 0.f);
}

// When total norm is below max_norm, grads are unchanged.
static void test_clip_grad_norm_below_max_no_change() {
  Parameter p = Parameter::zeros({2});
  p.set_grad(std::make_shared<Tensor>(Tensor::from_data({0.3f, 0.4f}, {2}, false)));
  std::vector<Parameter*> params = {&p};
  float norm = clip_grad_norm_(params, 10.f);
  assert(std::fabs(norm - 0.5f) < 1e-5f);
  assert(std::fabs(p.grad()->data_float()[0] - 0.3f) < 1e-5f);
  assert(std::fabs(p.grad()->data_float()[1] - 0.4f) < 1e-5f);
}

// Parameters with null grad are excluded from norm; only params with grad are clipped.
static void test_clip_grad_norm_excludes_params_without_grad() {
  Parameter a = Parameter::zeros({1});
  Parameter b = Parameter::zeros({1});
  a.set_grad(std::make_shared<Tensor>(Tensor::from_data({3.f}, {1}, false)));
  b.set_grad(nullptr);
  std::vector<Parameter*> params = {&a, &b};
  float n = clip_grad_norm_(params, 2.f);
  assert(std::fabs(n - 3.f) < 1e-5f);
  assert(std::fabs(a.grad()->data_float()[0] - 2.f) < 1e-5f);
}

// --- Module state_dict (documents serialization API for readers) ---

// state_dict() returns a map of dotted names -> tensors (e.g. "weight", "bias" for Linear).
static void test_module_state_dict_collects_named_params() {
  Linear linear(2, 3, true);
  auto state = linear.state_dict();
  assert(state.size() >= 2);
  bool has_weight = false, has_bias = false;
  for (const auto& kv : state) {
    if (kv.first.find("weight") != std::string::npos) {
      has_weight = true;
      assert((kv.second.shape() == std::vector<int64_t>{3, 2}));
    }
    if (kv.first.find("bias") != std::string::npos) {
      has_bias = true;
      assert((kv.second.shape() == std::vector<int64_t>{3}));
    }
  }
  assert(has_weight && has_bias);
}

// --- NN layer edge cases (constant input, single row, single class) ---

// LayerNorm with constant input: output is beta (normalized part is 0, then gamma*0+beta).
static void test_layernorm_constant_input() {
  LayerNorm ln(3, 1e-5f);
  Tensor x = Tensor::from_data({5.f, 5.f, 5.f, 5.f, 5.f, 5.f}, {2, 3}, false);
  Tensor y = ln(x);
  for (int64_t i = 0; i < y.numel(); ++i) {
    assert(std::fabs(y.data_float()[i]) < 1e-4f);
  }
}

// Softmax with a single row (1, D): output shape (1, D), row sum 1.
static void test_softmax_single_row() {
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {1, 3}, false);
  Tensor y = softmax(x);
  assert(y.shape()[0] == 1 && y.shape()[1] == 3);
  float row_sum = 0.f;
  for (int64_t j = 0; j < 3; ++j) row_sum += y.data_float()[j];
  assert(std::fabs(row_sum - 1.f) < 1e-5f);
}

// CrossEntropy with single sample and single class (C=1): edge case.
static void test_cross_entropy_single_sample_single_class() {
  Tensor logits = Tensor::from_data({0.f}, {1, 1}, false);
  Tensor targets({1}, DType::Int64, Device::cpu(), false);
  targets.data_int64()[0] = 0;
  Tensor loss = cross_entropy(logits, targets);
  assert(loss.numel() == 1);
  assert(std::isfinite(loss.data_float()[0]));
}

// Embedding with num_embeddings=1: only one row in weight, all indices must be 0.
static void test_embedding_single_vocab() {
  seed(0);
  Embedding emb(1, 2);
  Tensor indices({2}, DType::Int64, Device::cpu(), false);
  indices.data_int64()[0] = 0;
  indices.data_int64()[1] = 0;
  Tensor y = emb(indices);
  assert(y.shape()[0] == 2 && y.shape()[1] == 2);
  assert(std::fabs(y.data_float()[0] - y.data_float()[2]) < 1e-5f);
}

// Dropout p=0: no masking, scale 1, output equals input in train mode.
static void test_dropout_p_zero_no_drop() {
  Dropout d(0.f);
  d.train();
  Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {1, 3}, false);
  seed(99);
  Tensor y = d(x);
  for (int64_t i = 0; i < 3; ++i)
    assert(std::fabs(y.data_float()[i] - x.data_float()[i]) < 1e-5f);
}

void run_all_tests() {
  test_version();
  test_tensor_basic_shape();
  test_tensor_zeros();
  test_tensor_from_data();
  test_tensor_reshape();
  test_tensor_single_element_scalar_like();
  test_tensor_reshape_to_one_dim();
  test_matmul_1x1_edge();
  test_backward_grad_accumulation();
  test_detach_stops_gradient_flow();
  test_mean_single_row();

  grad_check_add();
  grad_check_mul();
  grad_check_sum();
  grad_check_matmul();
  grad_check_transpose();
  test_add_bias_broadcast_grad();
  test_mul_bias_broadcast_grad();
  test_elementwise_and_reductions_shapes();
  test_no_grad();
  test_detach();
  test_detach_stops_gradient_flow();
  test_module_parameters_and_modes();
  test_module_state_dict_collects_named_params();

  test_seed_uniform_determinism_and_range();
  test_uniform_range();
  test_seed_normal_determinism();
  test_normal_approximate_stats();
  test_zeros_float32();
  test_zeros_int64();
  test_xavier_uniform_range();

  test_linear_forward_shape();
  test_linear_no_bias_forward_shape();
  test_linear_backward_weight_and_bias();

  test_layernorm_forward_shape();
  test_layernorm_grad_check();

  test_gather_forward_shape_and_values();
  test_gather_backward();
  test_embedding_forward_shape();
  test_embedding_backward();

  test_dropout_eval_identity();
  test_dropout_train_deterministic_with_seed();
  test_dropout_backward_matches_mask_scale();

  test_gelu_forward_shape();
  test_gelu_grad_check();

  test_softmax_row_sums_to_one();
  test_log_softmax_exp_row_sums_to_one();
  test_softmax_grad_check_one_element();
  test_log_softmax_grad_check_one_element();

  test_attention_forward_shape();
  test_attention_causal_mask();
  test_attention_backward();
  test_attention_module_wrapper();

  test_tensor_dataset_size_and_get();
  test_dataloader_num_batches_and_batch_shape();
  test_dataloader_shuffle_deterministic();

  test_save_load_tensor();
  test_save_load_state_dict_module_roundtrip();
  test_load_state_dict_restores_parameters();
  test_adamw_state_dict_roundtrip();

  test_cross_entropy_known_value();
  test_cross_entropy_grad_check_one_element();
  test_cross_entropy_module_wrapper();

  test_sgd_quadratic_descent();
  test_adamw_quadratic_descent();
  test_clip_grad_norm();
  test_sgd_empty_params();
  test_sgd_skips_param_without_grad();
  test_adamw_step_count_increments();
  test_clip_grad_norm_empty_params();
  test_clip_grad_norm_below_max_no_change();
  test_clip_grad_norm_excludes_params_without_grad();

  test_layernorm_constant_input();
  test_softmax_single_row();
  test_cross_entropy_single_sample_single_class();
  test_embedding_single_vocab();
  test_dropout_p_zero_no_drop();
}
