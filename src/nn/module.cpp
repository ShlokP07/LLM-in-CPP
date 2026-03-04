#include <llm/module.hpp>

namespace llm {

Parameter::Parameter(const Tensor& t) : Tensor(t) {
  set_requires_grad(true);
}

Parameter Parameter::zeros(const std::vector<int64_t>& shape,
                           DType dtype,
                           Device device) {
  Tensor t = Tensor::zeros(shape, dtype, device, /*requires_grad=*/true);
  return Parameter(t);
}

Module::Module() = default;

Module::~Module() = default;

void Module::register_parameter(const std::string& name, const Parameter& p) {
  parameters_[name] = p;
}

void Module::register_module(const std::string& name,
                             const std::shared_ptr<Module>& module) {
  modules_[name] = module;
}

std::vector<Parameter*> Module::parameters() {
  std::vector<Parameter*> result;
  collect_parameters(result, /*prefix=*/"");
  return result;
}

void Module::train() {
  training_ = true;
  for (auto& kv : modules_) {
    if (kv.second) {
      kv.second->train();
    }
  }
}

void Module::eval() {
  training_ = false;
  for (auto& kv : modules_) {
    if (kv.second) {
      kv.second->eval();
    }
  }
}

bool Module::is_training() const {
  return training_;
}

Module::StateDict Module::state_dict() const {
  StateDict state;
  collect_state(state, /*prefix=*/"");
  return state;
}

void Module::load_state_dict(const StateDict& state) {
  load_state_dict_impl(state, "");
}

void Module::load_state_dict_impl(const StateDict& state, const std::string& prefix) {
  for (auto& kv : parameters_) {
    const std::string& name = kv.first;
    std::string full_name = prefix.empty() ? name : (prefix + "." + name);
    auto it = state.find(full_name);
    if (it != state.end()) {
      kv.second.copy_(it->second);
    }
  }
  for (auto& kv : modules_) {
    const std::string& name = kv.first;
    if (!kv.second) continue;
    std::string child_prefix = prefix.empty() ? name : (prefix + "." + name);
    kv.second->load_state_dict_impl(state, child_prefix);
  }
}

void Module::collect_parameters(std::vector<Parameter*>& out,
                                const std::string& /*prefix*/) {
  for (auto& kv : parameters_) {
    out.push_back(&kv.second);
  }
  for (auto& kv : modules_) {
    if (kv.second) {
      kv.second->collect_parameters(out, /*prefix=*/"");
    }
  }
}

void Module::collect_state(StateDict& out, const std::string& prefix) const {
  for (const auto& kv : parameters_) {
    const std::string& name = kv.first;
    std::string full_name = prefix.empty() ? name : (prefix + "." + name);
    out.emplace(full_name, static_cast<const Tensor&>(kv.second));
  }
  for (const auto& kv : modules_) {
    const std::string& name = kv.first;
    const auto& module = kv.second;
    if (module) {
      std::string child_prefix =
          prefix.empty() ? name : (prefix + "." + name);
      module->collect_state(out, child_prefix);
    }
  }
}

}  // namespace llm

