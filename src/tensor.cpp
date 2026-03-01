#include <llm/tensor.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace llm {

int64_t Tensor::compute_numel(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0) {
      throw std::invalid_argument("Tensor shape dimensions must be positive");
    }
    n *= d;
  }
  return n;
}

std::vector<int64_t> Tensor::compute_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

std::size_t Tensor::element_size(DType dt) {
  switch (dt) {
    case DType::Float32:
      return sizeof(float);
    case DType::Int64:
      return sizeof(int64_t);
    default:
      throw std::runtime_error("Unsupported DType in element_size");
  }
}

Tensor::Tensor(const std::vector<int64_t>& shape,
               DType dtype,
               Device device,
               bool requires_grad)
    : shape_(shape),
      strides_(compute_strides(shape)),
      numel_(compute_numel(shape)),
      dtype_(dtype),
      device_(device),
      requires_grad_(requires_grad) {
  const std::size_t bytes = static_cast<std::size_t>(numel_) * element_size(dtype_);
  // Allocate raw storage and wrap it in shared_ptr with custom deleter.
  void* ptr = operator new(bytes);
  storage_.reset(ptr, [](void* p) { operator delete(p); });
  // Zero initialize for determinism.
  std::memset(storage_.get(), 0, bytes);
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape,
                     DType dtype,
                     Device device,
                     bool requires_grad) {
  return Tensor(shape, dtype, device, requires_grad);
}

Tensor Tensor::from_data(std::vector<float> data,
                         const std::vector<int64_t>& shape,
                         bool requires_grad) {
  Tensor t(shape, DType::Float32, Device::cpu(), requires_grad);
  if (t.numel() != static_cast<int64_t>(data.size())) {
    throw std::invalid_argument("from_data: data size does not match shape numel");
  }
  std::memcpy(t.data(), data.data(),
              static_cast<std::size_t>(data.size()) * sizeof(float));
  return t;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
  int64_t new_numel = compute_numel(new_shape);
  if (new_numel != numel_) {
    throw std::invalid_argument("reshape: total number of elements must not change");
  }

  Tensor out;
  out.shape_ = new_shape;
  out.strides_ = compute_strides(new_shape);
  out.numel_ = numel_;
  out.dtype_ = dtype_;
  out.device_ = device_;
  out.requires_grad_ = requires_grad_;
  out.storage_ = storage_;  // share underlying data
  out.grad_fn_ = grad_fn_;
  out.grad_ = grad_;
  return out;
}

std::string Tensor::debug_string() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (std::size_t i = 0; i < shape_.size(); ++i) {
    oss << shape_[i];
    if (i + 1 < shape_.size()) {
      oss << ", ";
    }
  }
  oss << "], dtype=";
  switch (dtype_) {
    case DType::Float32:
      oss << "float32";
      break;
    case DType::Int64:
      oss << "int64";
      break;
  }
  oss << ", device=CPU";
  oss << ", requires_grad=" << (requires_grad_ ? "true" : "false");
  oss << ")";
  return oss.str();
}

}  // namespace llm

