#pragma once

#include <llm/tensor.hpp>

namespace llm {

/** Element-wise add: out = a + b. Supports same-shape and (N,D)+(D) bias-style broadcast. */
Tensor add(const Tensor& a, const Tensor& b);

/** Element-wise multiply: out = a * b. Supports same-shape and (N,D)*(D) bias-style broadcast. */
Tensor mul(const Tensor& a, const Tensor& b);

/** Element-wise subtract: out = a - b. Same-shape only. */
Tensor sub(const Tensor& a, const Tensor& b);

/** Element-wise divide: out = a / b. Same-shape only. */
Tensor div(const Tensor& a, const Tensor& b);

/** Element-wise negation: out = -a. */
Tensor neg(const Tensor& a);

/** Sum all elements to scalar. */
Tensor sum(const Tensor& a);

/** Sum along a dimension (currently supports 2D, dim=0 or 1). */
Tensor sum(const Tensor& a, int64_t dim, bool keepdim);

/** Mean along a dimension (currently supports 2D, dim=0 or 1). */
Tensor mean(const Tensor& a, int64_t dim, bool keepdim);

/** Element-wise exponential: out = exp(a). */
Tensor exp(const Tensor& a);

/** Max along a dimension (returns values only; supports 2D, dim=0 or 1). */
Tensor max(const Tensor& a, int64_t dim, bool keepdim);

/** Matrix multiply (2D): out = a @ b. a: (M,K), b: (K,N) → (M,N). */
Tensor matmul(const Tensor& a, const Tensor& b);

/** Transpose last two dimensions. For 2D: out = a^T. */
Tensor transpose(const Tensor& a);

/** Ones with same shape as t (float32, no grad). */
Tensor ones_like(const Tensor& t);

}  // namespace llm
