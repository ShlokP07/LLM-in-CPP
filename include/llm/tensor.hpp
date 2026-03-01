#pragma once

// Tensor core public API.
// This header defines a minimal, CPU-only tensor abstraction used by the rest
// of the "LLM From Scratch" project. The focus is on predictable layout
// (contiguous row-major), explicit shape/stride metadata, and hooks for a
// future autograd system, rather than on performance optimizations.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llm
{

  // Device representation is intentionally tiny: we only support CPU for now,
  // but keep the type so we can grow into CUDA or other backends later.
  enum class DeviceType
  {
    CPU,
  };

  struct Device
  {
    DeviceType type;

    constexpr Device(DeviceType t = DeviceType::CPU) : type(t) {}

    static constexpr Device cpu() { return Device(DeviceType::CPU); }
  };

  // Supported dtypes (extend later if needed).
  enum class DType
  {
    Float32,
    Int64,
  };

  // Forward-declare autograd node type (defined in autograd core later).
  class AutogradNode;

  // Lightweight Tensor handle. Owns shared storage and metadata; all of the
  // heavier logic (allocation, shape checks, etc.) lives in the .cpp file so
  // users of the library only have to include this header.
  class Tensor
  {
  public:
    Tensor() = default;

    // Construct an uninitialized tensor with given shape and dtype.
    Tensor(const std::vector<int64_t> &shape,
           DType dtype = DType::Float32,
           Device device = Device::cpu(),
           bool requires_grad = false);

    // Factory helpers.
    static Tensor zeros(const std::vector<int64_t> &shape,
                        DType dtype = DType::Float32,
                        Device device = Device::cpu(),
                        bool requires_grad = false);

    static Tensor from_data(std::vector<float> data,
                            const std::vector<int64_t> &shape,
                            bool requires_grad = false);

    // Basic metadata.
    const std::vector<int64_t> &shape() const { return shape_; }
    const std::vector<int64_t> &strides() const { return strides_; }
    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }

    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool v) { requires_grad_ = v; }

    // Gradient access (will be populated by autograd engine later).
    std::shared_ptr<Tensor> grad() const { return grad_; }
    void set_grad(const std::shared_ptr<Tensor> &g) { grad_ = g; }

    // Raw data accessors. Caller must respect dtype().
    void *data() { return storage_.get(); }
    const void *data() const { return storage_.get(); }

    // Convenience typed accessors for float32 and int64.
    float *data_float()
    {
      return dtype_ == DType::Float32 ? static_cast<float *>(storage_.get())
                                      : nullptr;
    }
    const float *data_float() const
    {
      return dtype_ == DType::Float32 ? static_cast<const float *>(storage_.get())
                                      : nullptr;
    }

    int64_t *data_int64()
    {
      return dtype_ == DType::Int64 ? static_cast<int64_t *>(storage_.get())
                                    : nullptr;
    }
    const int64_t *data_int64() const
    {
      return dtype_ == DType::Int64 ? static_cast<const int64_t *>(storage_.get())
                                    : nullptr;
    }

    // Simple reshape that preserves total number of elements.
    // For now, we enforce contiguous layout and update strides accordingly.
    Tensor reshape(const std::vector<int64_t> &new_shape) const;

    // Debug helper.
    std::string debug_string() const;

  private:
    std::shared_ptr<void> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t numel_ = 0;
    DType dtype_ = DType::Float32;
    Device device_ = Device::cpu();
    bool requires_grad_ = false;

    // Autograd metadata (filled in by autograd engine).
    std::weak_ptr<AutogradNode> grad_fn_;
    std::shared_ptr<Tensor> grad_;

    static int64_t compute_numel(const std::vector<int64_t> &shape);
    static std::vector<int64_t> compute_strides(const std::vector<int64_t> &shape);
    static std::size_t element_size(DType dt);
  };

} // namespace llm
