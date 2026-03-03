#include <llm/ops.hpp>
#include <llm/autograd.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llm {

namespace {

void expect_float32(const Tensor& t, const char* name) {
  if (t.dtype() != DType::Float32)
    throw std::invalid_argument(std::string(name) + ": expected float32");
}

void expect_same_shape(const Tensor& a, const Tensor& b, const char* name) {
  if (a.shape() != b.shape())
    throw std::invalid_argument(std::string(name) + ": shape mismatch");
}

// Check if (A,B) form a simple bias-style broadcast: (N,D)+(D) or (D)+(N,D).
// Returns true and sets out_M,out_N and flags if supported; otherwise false.
bool is_bias_broadcast(const Tensor& a, const Tensor& b,
                       int64_t& M, int64_t& N,
                       bool& a_is_matrix, bool& b_is_matrix) {
  const auto& ash = a.shape();
  const auto& bsh = b.shape();

  if (a.dim() == 2 && b.dim() == 1 &&
      ash[1] == bsh[0]) {
    M = ash[0];
    N = ash[1];
    a_is_matrix = true;
    b_is_matrix = false;
    return true;
  }
  if (a.dim() == 1 && b.dim() == 2 &&
      ash[0] == bsh[1]) {
    M = bsh[0];
    N = bsh[1];
    a_is_matrix = false;
    b_is_matrix = true;
    return true;
  }
  return false;
}

// Add (same-shape): gradient passes through unchanged to both inputs.
class AddBackward : public AutogradNode {
public:
  AddBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b)
      : a_(std::move(a)), b_(std::move(b)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_, b_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (a_ && a_->requires_grad()) {
      a_->accumulate_grad(*grad_output);
    }
    if (b_ && b_->requires_grad()) {
      b_->accumulate_grad(*grad_output);
    }
  }
private:
  std::shared_ptr<Tensor> a_, b_;
};

// Add with bias-style broadcast: out = mat + bias, where mat is (M,N) and bias is (N).
class AddBiasBackward : public AutogradNode {
public:
  AddBiasBackward(std::shared_ptr<Tensor> mat,
                  std::shared_ptr<Tensor> bias,
                  bool mat_is_a,
                  int64_t M,
                  int64_t N)
      : mat_(std::move(mat)),
        bias_(std::move(bias)),
        mat_is_a_(mat_is_a),
        M_(M),
        N_(N) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    if (mat_is_a_) {
      return {mat_, bias_};
    } else {
      return {bias_, mat_};
    }
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    const float* g = grad_output->data_float();

    // Gradient w.r.t. matrix is grad_output itself (same shape).
    if (mat_ && mat_->requires_grad()) {
      std::shared_ptr<Tensor> g_mat = std::make_shared<Tensor>(
          std::vector<int64_t>{M_, N_}, DType::Float32, mat_->device(), false);
      std::memcpy(g_mat->data_float(), g,
                  static_cast<size_t>(M_ * N_) * sizeof(float));
      mat_->accumulate_grad(*g_mat);
    }

    // Gradient w.r.t. bias is sum over rows.
    if (bias_ && bias_->requires_grad()) {
      std::shared_ptr<Tensor> g_bias = std::make_shared<Tensor>(
          std::vector<int64_t>{N_}, DType::Float32, bias_->device(), false);
      float* gb = g_bias->data_float();
      for (int64_t j = 0; j < N_; ++j) {
        float acc = 0.f;
        for (int64_t i = 0; i < M_; ++i) {
          acc += g[i * N_ + j];
        }
        gb[j] = acc;
      }
      bias_->accumulate_grad(*g_bias);
    }
  }

private:
  std::shared_ptr<Tensor> mat_;
  std::shared_ptr<Tensor> bias_;
  bool mat_is_a_;
  int64_t M_;
  int64_t N_;
};

// Mul: d(a*b)/da = b, d(a*b)/db = a. Need a_val, b_val for the backward (same-shape).
class MulBackward : public AutogradNode {
public:
  MulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
              std::shared_ptr<Tensor> a_val, std::shared_ptr<Tensor> b_val)
      : a_(std::move(a)), b_(std::move(b)),
        a_val_(std::move(a_val)), b_val_(std::move(b_val)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_, b_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    NoGradGuard guard;
    if (a_ && a_->requires_grad()) {
      Tensor ga = mul(*grad_output, *b_val_);
      a_->accumulate_grad(ga);
    }
    if (b_ && b_->requires_grad()) {
      Tensor gb = mul(*grad_output, *a_val_);
      b_->accumulate_grad(gb);
    }
  }
private:
  std::shared_ptr<Tensor> a_, b_, a_val_, b_val_;
};

// Mul with bias-style broadcast: out = mat * bias, shapes (M,N)*(N).
class MulBiasBackward : public AutogradNode {
public:
  MulBiasBackward(std::shared_ptr<Tensor> mat,
                  std::shared_ptr<Tensor> bias,
                  std::shared_ptr<Tensor> mat_val,
                  std::shared_ptr<Tensor> bias_val,
                  bool mat_is_a,
                  int64_t M,
                  int64_t N)
      : mat_(std::move(mat)),
        bias_(std::move(bias)),
        mat_val_(std::move(mat_val)),
        bias_val_(std::move(bias_val)),
        mat_is_a_(mat_is_a),
        M_(M),
        N_(N) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    if (mat_is_a_) {
      return {mat_, bias_};
    } else {
      return {bias_, mat_};
    }
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    NoGradGuard guard;
    const float* g = grad_output->data_float();

    // Grad w.r.t. matrix: g_mat[i,j] = g[i,j] * bias[j]
    if (mat_ && mat_->requires_grad()) {
      std::shared_ptr<Tensor> g_mat = std::make_shared<Tensor>(
          std::vector<int64_t>{M_, N_}, DType::Float32, mat_->device(), false);
      float* gm = g_mat->data_float();
      const float* bval = bias_val_->data_float();
      for (int64_t i = 0; i < M_; ++i) {
        for (int64_t j = 0; j < N_; ++j) {
          gm[i * N_ + j] = g[i * N_ + j] * bval[j];
        }
      }
      mat_->accumulate_grad(*g_mat);
    }

    // Grad w.r.t. bias: g_bias[j] = sum_i g[i,j] * mat_val[i,j]
    if (bias_ && bias_->requires_grad()) {
      std::shared_ptr<Tensor> g_bias = std::make_shared<Tensor>(
          std::vector<int64_t>{N_}, DType::Float32, bias_->device(), false);
      float* gb = g_bias->data_float();
      const float* mval = mat_val_->data_float();
      for (int64_t j = 0; j < N_; ++j) {
        float acc = 0.f;
        for (int64_t i = 0; i < M_; ++i) {
          acc += g[i * N_ + j] * mval[i * N_ + j];
        }
        gb[j] = acc;
      }
      bias_->accumulate_grad(*g_bias);
    }
  }

private:
  std::shared_ptr<Tensor> mat_;
  std::shared_ptr<Tensor> bias_;
  std::shared_ptr<Tensor> mat_val_;
  std::shared_ptr<Tensor> bias_val_;
  bool mat_is_a_;
  int64_t M_;
  int64_t N_;
};

// Sum: gradient is scalar broadcast to input shape.
class SumBackward : public AutogradNode {
public:
  explicit SumBackward(std::shared_ptr<Tensor> a) : a_(std::move(a)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!a_ || !a_->requires_grad()) return;
    float g = grad_output->data_float()[0];
    std::shared_ptr<Tensor> grad_a = std::make_shared<Tensor>(
        a_->shape(), DType::Float32, a_->device(), false);
    float* p = grad_a->data_float();
    for (int64_t i = 0; i < grad_a->numel(); ++i) p[i] = g;
    a_->accumulate_grad(*grad_a);
  }
private:
  std::shared_ptr<Tensor> a_;
};

// Transpose: gradient of transpose is transpose of gradient.
class TransposeBackward : public AutogradNode {
public:
  explicit TransposeBackward(std::shared_ptr<Tensor> a) : a_(std::move(a)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!a_ || !a_->requires_grad()) return;
    NoGradGuard guard;
    Tensor g = transpose(*grad_output);
    a_->accumulate_grad(g);
  }
private:
  std::shared_ptr<Tensor> a_;
};

// --- Matmul ---
class MatmulBackward : public AutogradNode {
public:
  MatmulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                 std::shared_ptr<Tensor> a_val, std::shared_ptr<Tensor> b_val)
      : a_(std::move(a)), b_(std::move(b)),
        a_val_(std::move(a_val)), b_val_(std::move(b_val)) {}
  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_, b_};
  }
  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    NoGradGuard guard;
    // d(out)/dA = grad_output @ B^T, d(out)/dB = A^T @ grad_output
    if (a_ && a_->requires_grad()) {
      Tensor bt = transpose(*b_val_);
      Tensor ga = matmul(*grad_output, bt);
      a_->accumulate_grad(ga);
    }
    if (b_ && b_->requires_grad()) {
      Tensor at = transpose(*a_val_);
      Tensor gb = matmul(at, *grad_output);
      b_->accumulate_grad(gb);
    }
  }
private:
  std::shared_ptr<Tensor> a_, b_, a_val_, b_val_;
};

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  expect_float32(a, "add");
  expect_float32(b, "add");
  int64_t M = 0, N = 0;
  bool a_is_matrix = false, b_is_matrix = false;

  if (a.shape() == b.shape()) {
    Tensor out(a.shape(), DType::Float32, a.device(), false);
    const float* pa = a.data_float();
    const float* pb = b.data_float();
    float* po = out.data_float();
    for (int64_t i = 0; i < out.numel(); ++i)
      po[i] = pa[i] + pb[i];

    if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
      out.set_requires_grad(true);
      auto node = std::make_shared<AddBackward>(
          std::make_shared<Tensor>(a),
          std::make_shared<Tensor>(b));
      out.set_grad_fn(node);
    }
    return out;
  } else if (is_bias_broadcast(a, b, M, N, a_is_matrix, b_is_matrix)) {
    // Bias-style broadcast: (M,N) + (N)
    Tensor out({M, N}, DType::Float32, a.device(), false);
    const float* pa = a.data_float();
    const float* pb = b.data_float();
    float* po = out.data_float();

    if (a_is_matrix) {
      // a: (M,N), b: (N)
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          po[i * N + j] = pa[i * N + j] + pb[j];
    } else {
      // a: (N), b: (M,N)
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          po[i * N + j] = pa[j] + pb[i * N + j];
    }

    if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
      out.set_requires_grad(true);
      std::shared_ptr<Tensor> mat =
          a_is_matrix ? std::make_shared<Tensor>(a) : std::make_shared<Tensor>(b);
      std::shared_ptr<Tensor> bias =
          a_is_matrix ? std::make_shared<Tensor>(b) : std::make_shared<Tensor>(a);
      auto node = std::make_shared<AddBiasBackward>(mat, bias, a_is_matrix, M, N);
      out.set_grad_fn(node);
    }
    return out;
  } else {
    expect_same_shape(a, b, "add");
    // unreachable, expect_same_shape will throw.
    return a;
  }
}

Tensor mul(const Tensor& a, const Tensor& b) {
  expect_float32(a, "mul");
  expect_float32(b, "mul");
  int64_t M = 0, N = 0;
  bool a_is_matrix = false, b_is_matrix = false;

  if (a.shape() == b.shape()) {
    Tensor out(a.shape(), DType::Float32, a.device(), false);
    const float* pa = a.data_float();
    const float* pb = b.data_float();
    float* po = out.data_float();
    for (int64_t i = 0; i < out.numel(); ++i)
      po[i] = pa[i] * pb[i];

    if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
      out.set_requires_grad(true);
      auto a_copy = std::make_shared<Tensor>(a);
      auto b_copy = std::make_shared<Tensor>(b);
      auto node = std::make_shared<MulBackward>(a_copy, b_copy, a_copy, b_copy);
      out.set_grad_fn(node);
    }
    return out;
  } else if (is_bias_broadcast(a, b, M, N, a_is_matrix, b_is_matrix)) {
    Tensor out({M, N}, DType::Float32, a.device(), false);
    const float* pa = a.data_float();
    const float* pb = b.data_float();
    float* po = out.data_float();

    if (a_is_matrix) {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          po[i * N + j] = pa[i * N + j] * pb[j];
    } else {
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          po[i * N + j] = pa[j] * pb[i * N + j];
    }

    if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
      out.set_requires_grad(true);
      std::shared_ptr<Tensor> mat =
          a_is_matrix ? std::make_shared<Tensor>(a) : std::make_shared<Tensor>(b);
      std::shared_ptr<Tensor> bias =
          a_is_matrix ? std::make_shared<Tensor>(b) : std::make_shared<Tensor>(a);
      auto mat_val = mat;
      auto bias_val = bias;
      auto node = std::make_shared<MulBiasBackward>(mat, bias, mat_val, bias_val,
                                                    a_is_matrix, M, N);
      out.set_grad_fn(node);
    }
    return out;
  } else {
    expect_same_shape(a, b, "mul");
    // unreachable, expect_same_shape will throw.
    return a;
  }
}

Tensor sub(const Tensor& a, const Tensor& b) {
  expect_float32(a, "sub");
  expect_float32(b, "sub");
  expect_same_shape(a, b, "sub");

  Tensor out(a.shape(), DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  const float* pb = b.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i)
    po[i] = pa[i] - pb[i];

  // For now, treat sub as a simple op with no dedicated backward; it can be
  // expressed as add(a, -b) when needed for autograd-aware code.
  return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
  expect_float32(a, "div");
  expect_float32(b, "div");
  expect_same_shape(a, b, "div");

  Tensor out(a.shape(), DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  const float* pb = b.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i) {
    if (pb[i] == 0.f)
      throw std::runtime_error("div: division by zero");
    po[i] = pa[i] / pb[i];
  }
  // For now, div is forward-only for use in higher-level ops.
  return out;
}

Tensor neg(const Tensor& a) {
  expect_float32(a, "neg");
  Tensor out(a.shape(), DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < a.numel(); ++i) {
    po[i] = -pa[i];
  }
  // Forward-only; gradient can be expressed via higher-level ops later.
  return out;
}

Tensor sum(const Tensor& a) {
  expect_float32(a, "sum");
  std::vector<int64_t> scalar_shape = {1};
  Tensor out(scalar_shape, DType::Float32, a.device(), false);
  float s = 0;
  const float* p = a.data_float();
  for (int64_t i = 0; i < a.numel(); ++i) s += p[i];
  out.data_float()[0] = s;

  if (is_grad_enabled() && a.requires_grad()) {
    out.set_requires_grad(true);
    auto node = std::make_shared<SumBackward>(std::make_shared<Tensor>(a));
    out.set_grad_fn(node);
  }
  return out;
}

Tensor sum(const Tensor& a, int64_t dim, bool keepdim) {
  expect_float32(a, "sum");
  if (a.dim() != 2)
    throw std::invalid_argument("sum(x, dim): only 2D tensors supported for now");
  if (dim < 0 || dim > 1)
    throw std::invalid_argument("sum(x, dim): dim must be 0 or 1");

  int64_t M = a.shape()[0];
  int64_t N = a.shape()[1];
  const float* pa = a.data_float();

  std::vector<int64_t> out_shape;
  if (dim == 0) {
    // Reduce rows → length-N vector.
    out_shape = keepdim ? std::vector<int64_t>{1, N} : std::vector<int64_t>{N};
    Tensor out(out_shape, DType::Float32, a.device(), false);
    float* po = out.data_float();
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int64_t i = 0; i < M; ++i) {
        acc += pa[i * N + j];
      }
      po[keepdim ? j : j] = acc;
    }
    return out;
  } else {
    // dim == 1: reduce columns → length-M vector.
    out_shape = keepdim ? std::vector<int64_t>{M, 1} : std::vector<int64_t>{M};
    Tensor out(out_shape, DType::Float32, a.device(), false);
    float* po = out.data_float();
    for (int64_t i = 0; i < M; ++i) {
      float acc = 0.f;
      for (int64_t j = 0; j < N; ++j) {
        acc += pa[i * N + j];
      }
      po[keepdim ? i : i] = acc;
    }
    return out;
  }
}

Tensor mean(const Tensor& a, int64_t dim, bool keepdim) {
  expect_float32(a, "mean");
  if (a.dim() != 2)
    throw std::invalid_argument("mean(x, dim): only 2D tensors supported for now");
  if (dim < 0 || dim > 1)
    throw std::invalid_argument("mean(x, dim): dim must be 0 or 1");

  int64_t M = a.shape()[0];
  int64_t N = a.shape()[1];
  Tensor s = sum(a, dim, keepdim);
  float* ps = s.data_float();
  int64_t len = (dim == 0) ? N : M;
  float denom = static_cast<float>((dim == 0) ? M : N);
  for (int64_t i = 0; i < len; ++i) {
    ps[i] /= denom;
  }
  return s;
}

Tensor exp(const Tensor& a) {
  expect_float32(a, "exp");
  Tensor out(a.shape(), DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  float* po = out.data_float();
  for (int64_t i = 0; i < a.numel(); ++i) {
    po[i] = std::exp(pa[i]);
  }
  // For now, we treat exp as non-differentiable; softmax will provide its own backward.
  return out;
}

Tensor max(const Tensor& a, int64_t dim, bool keepdim) {
  expect_float32(a, "max");
  if (a.dim() != 2)
    throw std::invalid_argument("max(x, dim): only 2D tensors supported for now");
  if (dim < 0 || dim > 1)
    throw std::invalid_argument("max(x, dim): dim must be 0 or 1");

  int64_t M = a.shape()[0];
  int64_t N = a.shape()[1];
  const float* pa = a.data_float();

  if (dim == 0) {
    std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>{1, N}
                                             : std::vector<int64_t>{N};
    Tensor out(out_shape, DType::Float32, a.device(), false);
    float* po = out.data_float();
    for (int64_t j = 0; j < N; ++j) {
      float m = pa[j];
      for (int64_t i = 1; i < M; ++i) {
        float v = pa[i * N + j];
        if (v > m) m = v;
      }
      po[keepdim ? j : j] = m;
    }
    return out;
  } else {
    std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>{M, 1}
                                             : std::vector<int64_t>{M};
    Tensor out(out_shape, DType::Float32, a.device(), false);
    float* po = out.data_float();
    for (int64_t i = 0; i < M; ++i) {
      float m = pa[i * N];
      for (int64_t j = 1; j < N; ++j) {
        float v = pa[i * N + j];
        if (v > m) m = v;
      }
      po[keepdim ? i : i] = m;
    }
    return out;
  }
}

Tensor transpose(const Tensor& a) {
  expect_float32(a, "transpose");
  if (a.dim() < 2)
    throw std::invalid_argument("transpose: need at least 2 dimensions");
  const auto& sh = a.shape();
  std::vector<int64_t> new_shape(sh.size());
  for (size_t i = 0; i < sh.size(); ++i) new_shape[i] = sh[i];
  std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);

  Tensor out(new_shape, DType::Float32, a.device(), false);
  int64_t M = sh[sh.size() - 2];
  int64_t N = sh[sh.size() - 1];
  const float* src = a.data_float();
  float* dst = out.data_float();
  if (a.dim() == 2) {
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j)
        dst[j * M + i] = src[i * N + j];
  } else {
    int64_t batch = 1;
    for (size_t i = 0; i < sh.size() - 2; ++i) batch *= sh[i];
    for (int64_t b = 0; b < batch; ++b) {
      const float* s = src + b * M * N;
      float* d = dst + b * M * N;
      for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
          d[j * M + i] = s[i * N + j];
    }
  }

  if (is_grad_enabled() && a.requires_grad()) {
    out.set_requires_grad(true);
    auto node = std::make_shared<TransposeBackward>(std::make_shared<Tensor>(a));
    out.set_grad_fn(node);
  }
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  expect_float32(a, "matmul");
  expect_float32(b, "matmul");
  if (a.dim() != 2 || b.dim() != 2)
    throw std::invalid_argument("matmul: 2D tensors only");
  int64_t M = a.shape()[0];
  int64_t K = a.shape()[1];
  if (b.shape()[0] != K)
    throw std::invalid_argument("matmul: incompatible shapes");
  int64_t N = b.shape()[1];

  Tensor out({M, N}, DType::Float32, a.device(), false);
  const float* pa = a.data_float();
  const float* pb = b.data_float();
  float* po = out.data_float();
  std::memset(po, 0, static_cast<size_t>(M * N) * sizeof(float));
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) {
      float aik = pa[i * K + k];
      for (int64_t j = 0; j < N; ++j)
        po[i * N + j] += aik * pb[k * N + j];
    }

  if (is_grad_enabled() && (a.requires_grad() || b.requires_grad())) {
    out.set_requires_grad(true);
    auto a_copy = std::make_shared<Tensor>(a);
    auto b_copy = std::make_shared<Tensor>(b);
    auto node = std::make_shared<MatmulBackward>(a_copy, b_copy, a_copy, b_copy);
    out.set_grad_fn(node);
  }
  return out;
}

Tensor ones_like(const Tensor& t) {
  Tensor out(t.shape(), DType::Float32, t.device(), false);
  float* p = out.data_float();
  for (int64_t i = 0; i < out.numel(); ++i) p[i] = 1.0f;
  return out;
}

}  // namespace llm
