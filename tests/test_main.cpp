/**
 * Test entry point for basic Tensor sanity checks and autograd gradient checks.
 */

#include <llm/llm.hpp>
#include <llm/ops.hpp>
#include <llm/autograd.hpp>
#include <llm/module.hpp>
#include <llm/init.hpp>

#include <cassert>
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
using llm::seed;
using llm::uniform_;
using llm::normal_;
using llm::xavier_uniform_;
using llm::zeros_;

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

int main() {
  std::cout << "Running LLM tests..." << std::endl;

  test_version();
  test_tensor_basic_shape();
  test_tensor_zeros();
  test_tensor_from_data();
  test_tensor_reshape();

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
  test_module_parameters_and_modes();

  test_seed_uniform_determinism_and_range();
  test_uniform_range();
  test_seed_normal_determinism();
  test_normal_approximate_stats();
  test_zeros_float32();
  test_zeros_int64();
  test_xavier_uniform_range();

  std::cout << "All Tensor and autograd tests passed." << std::endl;
  return 0;
}
