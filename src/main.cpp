/**
 * LLM From Scratch - Entry point.
 *
 * For now this binary is just a smoke test that the library builds, links,
 * and can be run. As the project grows we can turn this into an interactive
 * demo or training script that exercises the higher-level APIs.
 */

#include <llm/llm.hpp>
#include <iostream>

int main() {
  std::cout << "LLM From Scratch v" << llm::version() << std::endl;
  std::cout << "Environment ready. Core implementation pending." << std::endl;
  return 0;
}
