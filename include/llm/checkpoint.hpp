#pragma once

#include <llm/module.hpp>

#include <string>

namespace llm {

/** Save a single tensor to a binary file (shape, dtype, data). */
void save_tensor(const std::string& path, const Tensor& tensor);

/** Load a tensor from a file produced by save_tensor. */
Tensor load_tensor(const std::string& path);

/** Save a state_dict (name -> tensor) to a binary file. */
void save_state_dict(const std::string& path, const Module::StateDict& state);

/** Load a state_dict from a file produced by save_state_dict. */
Module::StateDict load_state_dict(const std::string& path);

}  // namespace llm
