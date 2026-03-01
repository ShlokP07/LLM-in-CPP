#pragma once

/**
 * LLM From Scratch - top-level public header.
 *
 * This file is the small "face" of the library: including it pulls in the
 * fundamental tensor type and exposes versioning information. As the project
 * grows, additional core headers (autograd, ops, nn, optim, data) can be
 * re-exported from here to provide a single, ergonomic include for users.
 */

#include <llm/tensor.hpp>

namespace llm {

// Library version helper. Bump this when you make incompatible API changes.
inline const char* version() { return "0.1.0"; }

}  // namespace llm
