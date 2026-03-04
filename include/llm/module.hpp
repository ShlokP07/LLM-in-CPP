#pragma once

#include <llm/tensor.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

// Parameter is a thin wrapper around Tensor that defaults to requires_grad = true.
class Parameter : public Tensor {
public:
  Parameter() = default;

  explicit Parameter(const Tensor& t);

  static Parameter zeros(const std::vector<int64_t>& shape,
                         DType dtype = DType::Float32,
                         Device device = Device::cpu());
};

// Base class for all neural network modules.
class Module : public std::enable_shared_from_this<Module> {
public:
  using StateDict = std::unordered_map<std::string, Tensor>;

  Module();
  virtual ~Module();

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  void register_parameter(const std::string& name, const Parameter& p);

  void register_module(const std::string& name,
                       const std::shared_ptr<Module>& module);

  // All parameters in this module and its submodules.
  virtual std::vector<Parameter*> parameters();

  virtual void train();
  virtual void eval();

  bool is_training() const;

  // Flat mapping from dotted names to tensors.
  virtual StateDict state_dict() const;
  virtual void load_state_dict(const StateDict& state);

protected:
  std::unordered_map<std::string, Parameter> parameters_;
  std::unordered_map<std::string, std::shared_ptr<Module>> modules_;
  bool training_ = true;

  void collect_parameters(std::vector<Parameter*>& out,
                          const std::string& prefix);

  void collect_state(StateDict& out, const std::string& prefix) const;

  /** Recursively load state by dotted name; used by load_state_dict. */
  void load_state_dict_impl(const StateDict& state, const std::string& prefix);
};

}  // namespace llm

