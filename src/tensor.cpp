#include <llm/tensor.hpp>
#include <llm/autograd.hpp>
#include <llm/dtype.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace llm {

namespace {

// Total element count; throws if any dim <= 0.
int64_t compute_numel(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 0;
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0) throw std::invalid_argument("Tensor shape dimensions must be positive");
    n *= d;
  }
  return n;
}

// Row-major strides: strides[i] = product of shape[i+1..end]
std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

class ReshapeBackward : public AutogradNode {
public:
  ReshapeBackward(std::shared_ptr<Tensor> a, std::vector<int64_t> original_shape)
      : a_(std::move(a)), original_shape_(std::move(original_shape)) {}

  std::vector<std::shared_ptr<Tensor>> inputs() const override {
    return {a_};
  }

  void backward(const std::shared_ptr<Tensor>& grad_output) override {
    if (!a_ || !a_->requires_grad()) return;
    NoGradGuard guard;
    Tensor g_view = grad_output->reshape(original_shape_);
    a_->accumulate_grad(g_view);
  }

private:
  std::shared_ptr<Tensor> a_;
  std::vector<int64_t> original_shape_;
};

}  // namespace

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
  impl_ = std::make_shared<TensorImpl>();
  impl_->shape = shape;
  impl_->strides = compute_strides(shape);
  impl_->numel = compute_numel(shape);
  impl_->dtype = dtype;
  impl_->device = device;
  impl_->requires_grad = requires_grad;
  const std::size_t bytes = static_cast<std::size_t>(impl_->numel) * element_size(dtype);
  void* ptr = operator new(bytes);
  impl_->storage.reset(ptr, [](void* p) { operator delete(p); });  // custom deleter for void*
  std::memset(impl_->storage.get(), 0, bytes);
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device, bool requires_grad) {
  return Tensor(shape, dtype, device, requires_grad);
}

Tensor Tensor::from_data(std::vector<float> data, const std::vector<int64_t>& shape, bool requires_grad) {
  Tensor t(shape, DType::Float32, Device::cpu(), requires_grad);
  if (t.numel() != static_cast<int64_t>(data.size()))
    throw std::invalid_argument("from_data: data size does not match shape numel");
  std::memcpy(t.data(), data.data(), data.size() * sizeof(float));
  return t;
}

const std::vector<int64_t>& Tensor::shape() const {
  if (!impl_) throw std::runtime_error("Tensor: null impl");
  return impl_->shape;
}
const std::vector<int64_t>& Tensor::strides() const {
  if (!impl_) throw std::runtime_error("Tensor: null impl");
  return impl_->strides;
}
int64_t Tensor::dim() const { return static_cast<int64_t>(shape().size()); }
int64_t Tensor::numel() const { return impl_ ? impl_->numel : 0; }
DType Tensor::dtype() const { return impl_ ? impl_->dtype : DType::Float32; }
Device Tensor::device() const { return impl_ ? impl_->device : Device::cpu(); }
bool Tensor::requires_grad() const { return impl_ && impl_->requires_grad; }
void Tensor::set_requires_grad(bool v) { if (impl_) impl_->requires_grad = v; }

std::shared_ptr<Tensor> Tensor::grad() const { return impl_ ? impl_->grad : nullptr; }
void Tensor::set_grad(const std::shared_ptr<Tensor>& g) { if (impl_) impl_->grad = g; }

void* Tensor::data() { return impl_ ? impl_->storage.get() : nullptr; }
const void* Tensor::data() const { return impl_ ? impl_->storage.get() : nullptr; }
float* Tensor::data_float() {
  return (impl_ && impl_->dtype == DType::Float32) ? static_cast<float*>(impl_->storage.get()) : nullptr;
}
const float* Tensor::data_float() const {
  return (impl_ && impl_->dtype == DType::Float32) ? static_cast<const float*>(impl_->storage.get()) : nullptr;
}
int64_t* Tensor::data_int64() {
  return (impl_ && impl_->dtype == DType::Int64) ? static_cast<int64_t*>(impl_->storage.get()) : nullptr;
}
const int64_t* Tensor::data_int64() const {
  return (impl_ && impl_->dtype == DType::Int64) ? static_cast<const int64_t*>(impl_->storage.get()) : nullptr;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
  if (!impl_) throw std::runtime_error("Tensor: null impl");
  int64_t new_numel = compute_numel(new_shape);
  if (new_numel != impl_->numel)
    throw std::invalid_argument("reshape: total number of elements must not change");
  auto impl = std::make_shared<TensorImpl>();
  impl->storage = impl_->storage;  // share storage, no copy
  impl->shape = new_shape;
  impl->strides = compute_strides(new_shape);
  impl->numel = impl_->numel;
  impl->dtype = impl_->dtype;
  impl->device = impl_->device;
  impl->requires_grad = impl_->requires_grad;
  Tensor out(impl);

  if (is_grad_enabled() && impl_->requires_grad) {
    out.set_requires_grad(true);
    auto node = std::make_shared<ReshapeBackward>(
        std::make_shared<Tensor>(*this),
        this->shape());
    out.set_grad_fn(node);
  }
  return out;
}

void Tensor::copy_(const Tensor& other) {
  if (!impl_ || !other.impl_)
    throw std::runtime_error("copy_: null tensor");
  if (impl_->shape != other.impl_->shape)
    throw std::invalid_argument("copy_: shape mismatch");
  if (impl_->dtype != other.impl_->dtype)
    throw std::invalid_argument("copy_: dtype mismatch");
  const std::size_t bytes = static_cast<std::size_t>(impl_->numel) * element_size(impl_->dtype);
  std::memcpy(impl_->storage.get(), other.impl_->storage.get(), bytes);
}

// Copy with same data but no grad tracking (breaks the graph).
Tensor Tensor::detach() const {
  if (!impl_) return Tensor();
  auto impl = std::make_shared<TensorImpl>();
  impl->storage = impl_->storage;
  impl->shape = impl_->shape;
  impl->strides = impl_->strides;
  impl->numel = impl_->numel;
  impl->dtype = impl_->dtype;
  impl->device = impl_->device;
  impl->requires_grad = false;
  return Tensor(impl);
}

void Tensor::backward() {
  if (impl_) run_backward(*this);
}

// Add g into this tensor's grad. Used when tensor feeds into multiple ops.
void Tensor::accumulate_grad(const Tensor& g) {
  if (!impl_) return;
  if (g.numel() != impl_->numel || g.dtype() != impl_->dtype)
    throw std::invalid_argument("accumulate_grad: shape/dtype mismatch");
  if (!impl_->grad)
    impl_->grad = std::make_shared<Tensor>(g);
  else {
    const float* src = g.data_float();
    float* dst = impl_->grad->data_float();
    for (int64_t i = 0; i < impl_->numel; ++i) dst[i] += src[i];
  }
}

void Tensor::set_grad_fn(std::shared_ptr<AutogradNode> fn) {
  if (impl_) impl_->grad_fn = std::move(fn);
}

std::shared_ptr<AutogradNode> Tensor::grad_fn() const {
  return impl_ ? impl_->grad_fn : nullptr;
}

std::string Tensor::debug_string() const {
  if (!impl_) return "Tensor(null)";
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < impl_->shape.size(); ++i) {
    oss << impl_->shape[i];
    if (i + 1 < impl_->shape.size()) oss << ", ";
  }
  oss << "], dtype=";
  oss << (impl_->dtype == DType::Float32 ? "float32" : "int64");
  oss << ", requires_grad=" << (impl_->requires_grad ? "true" : "false") << ")";
  return oss.str();
}

}  // namespace llm
