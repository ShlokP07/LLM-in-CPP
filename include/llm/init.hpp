#pragma once

#include <llm/tensor.hpp>

#include <cstdint>

namespace llm {

/** Set the global RNG seed (used by uniform_, normal_, xavier_uniform_, etc.). */
void seed(uint64_t s);

/** Fill tensor with values from Uniform(low, high). In-place; expects float32. */
void uniform_(Tensor& t, float low, float high);

/** Fill tensor with values from Normal(mean, stddev). In-place; expects float32. */
void normal_(Tensor& t, float mean, float stddev);

/**
 * Xavier uniform: fill 2D tensor with Uniform(-limit, limit) where
 * limit = sqrt(6 / (fan_in + fan_out)). For shape (out, in), fan_in=in, fan_out=out.
 */
void xavier_uniform_(Tensor& t);

/** Fill tensor with zeros. In-place; works for float32 (and int64 if needed). */
void zeros_(Tensor& t);

}  // namespace llm
