#pragma once

// Tensor core public API.
// Tensor is a handle to shared state (TensorImpl) so that autograd nodes
// and the user refer to the same tensor and gradients accumulate correctly.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llm {

  enum class DeviceType { CPU };

  struct Device {
    DeviceType type;
    constexpr Device(DeviceType t = DeviceType::CPU) : type(t) {}
    static constexpr Device cpu() { return Device(DeviceType::CPU); }
  };

  enum class DType { Float32, Int64 };

  class AutogradNode;

  struct TensorImpl;

  class Tensor {
  public:
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(const std::vector<int64_t>& shape,
           DType dtype = DType::Float32,
           Device device = Device::cpu(),
           bool requires_grad = false);

    static Tensor zeros(const std::vector<int64_t>& shape,
                        DType dtype = DType::Float32,
                        Device device = Device::cpu(),
                        bool requires_grad = false);

    static Tensor from_data(std::vector<float> data,
                            const std::vector<int64_t>& shape,
                            bool requires_grad = false);

    const std::vector<int64_t>& shape() const;
    const std::vector<int64_t>& strides() const;
    int64_t dim() const;
    int64_t numel() const;

    DType dtype() const;
    Device device() const;

    bool requires_grad() const;
    void set_requires_grad(bool v);

    std::shared_ptr<Tensor> grad() const;
    void set_grad(const std::shared_ptr<Tensor>& g);

    void* data();
    const void* data() const;
    float* data_float();
    const float* data_float() const;
    int64_t* data_int64();
    const int64_t* data_int64() const;

    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor detach() const;

    /** Copy data from other into this. Shapes and dtypes must match. */
    void copy_(const Tensor& other);

    void backward();
    void accumulate_grad(const Tensor& g);
    void set_grad_fn(std::shared_ptr<AutogradNode> fn);
    std::shared_ptr<AutogradNode> grad_fn() const;

    std::string debug_string() const;

    std::shared_ptr<TensorImpl> impl() const { return impl_; }

  private:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
  };

  // Internal representation. Tensor is a thin handle; impl_ is shared across copies.
  struct TensorImpl {
    std::shared_ptr<void> storage;   // raw bytes, freed via custom deleter
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;    // row-major: strides[i] = product of shape[i+1..]
    int64_t numel = 0;
    DType dtype = DType::Float32;
    Device device = Device::cpu();
    bool requires_grad = false;
    std::shared_ptr<AutogradNode> grad_fn;  // op that produced this tensor
    std::shared_ptr<Tensor> grad;           // accumulated gradient (filled by backward)
  };

}  // namespace llm
