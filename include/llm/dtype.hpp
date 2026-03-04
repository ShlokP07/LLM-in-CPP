#pragma once

#include <llm/tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llm {

inline std::size_t element_size(DType dt) {
  switch (dt) {
    case DType::Float32: return sizeof(float);
    case DType::Int64: return sizeof(int64_t);
    default: throw std::runtime_error("element_size: unsupported dtype");
  }
}

}  // namespace llm

