#include <llm/autograd.hpp>
#include <llm/ops.hpp>

#include <algorithm>
#include <functional>
#include <stack>
#include <unordered_set>

namespace llm {

namespace {

// Thread-local: true when we are allowed to record ops (not inside no_grad).
bool& grad_enabled_ref() {
  static thread_local bool enabled = true;
  return enabled;
}

using NodeOutputPair = std::pair<std::shared_ptr<AutogradNode>, Tensor*>;

// DFS post-order over the graph = reverse topo order for backward pass.
std::vector<NodeOutputPair> reverse_topo(Tensor& root) {
  std::vector<NodeOutputPair> order;
  std::unordered_set<AutogradNode*> visited;

  std::function<void(Tensor*)> dfs;
  dfs = [&](Tensor* t) {
    auto fn = t->grad_fn();
    if (!fn || visited.count(fn.get()))
      return;
    visited.insert(fn.get());
    for (const auto& inp : fn->inputs()) {
      if (inp)
        dfs(inp.get());
    }
    order.emplace_back(fn, t);
  };

  dfs(&root);
  std::reverse(order.begin(), order.end());
  return order;
}

}  // namespace

bool is_grad_enabled() {
  return grad_enabled_ref();
}

NoGradGuard::NoGradGuard() {
  grad_enabled_ref() = false;
}

NoGradGuard::~NoGradGuard() {
  grad_enabled_ref() = true;
}

void run_backward(Tensor& root) {
  if (root.dtype() != DType::Float32) {
    throw std::runtime_error("backward: only float32 tensors supported");
  }
  std::shared_ptr<Tensor> grad = root.grad();
  if (!grad && root.requires_grad()) {  // default: d(loss)/d(loss) = 1
    grad = std::make_shared<Tensor>(ones_like(root));
    root.set_grad(grad);
  }
  if (!grad) return;

  auto order = reverse_topo(root);
  for (auto& p : order) {
    std::shared_ptr<AutogradNode> node = p.first;
    Tensor* out_tensor = p.second;
    std::shared_ptr<Tensor> out_grad = out_tensor->grad();
    if (out_grad)
      node->backward(out_grad);
  }
}

}  // namespace llm
