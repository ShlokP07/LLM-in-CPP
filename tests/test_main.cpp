/**
 * Test entry point for grad checks, shape validation, etc.
 * Add tests as implementation progresses.
 */
#include <llm/llm.hpp>
#include <cassert>
#include <iostream>

int main() {
  std::cout << "Running LLM tests..." << std::endl;
  assert(llm::version() != nullptr);
  std::cout << "Basic sanity check passed." << std::endl;
  return 0;
}
