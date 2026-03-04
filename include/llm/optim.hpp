#pragma once

#include <llm/module.hpp>

#include <vector>

namespace llm {

/**
 * Stochastic Gradient Descent optimizer.
 *
 * Updates parameters with:
 *   p = p - lr * (grad + weight_decay * p)
 * for each float32 Parameter in the parameter list.
 */
class SGD {
public:
  SGD(const std::vector<Parameter*>& params,
      float lr,
      float weight_decay = 0.0f);

  void step();
  void zero_grad();

  float lr() const { return lr_; }
  float weight_decay() const { return weight_decay_; }

private:
  std::vector<Parameter*> params_;
  float lr_;
  float weight_decay_;
};

}  // namespace llm

