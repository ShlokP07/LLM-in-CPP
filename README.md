# LLM From Scratch (C++)

Minimal Tensor + Autograd + Transformer framework for training miniGPT. CPU-first, debuggable, PyTorch-like API.

## Environment Setup

### Prerequisites

- **C++17** compiler (MSVC 2019+, GCC 8+, Clang 7+)
- **CMake** 3.16+

### Configure & Build

```powershell
cd "d:\Downloads\VS Code Projects\LLM From Scartch in C++"
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

BLAS (for faster matmul) is optional and can be added later by installing a BLAS library manually and letting CMake's `find_package(BLAS)` locate it.

### 3. Run

```powershell
.\Release\llm_main.exe
```

### 4. Run Tests

```powershell
ctest -C Release
# or
.\Release\llm_tests.exe
```

## Project Structure

```
├── CMakeLists.txt
├── include/llm/         # Public headers
├── src/                 # Implementation
│   ├── main.cpp
│   ├── core_stub.cpp
│   ├── tensor.cpp       # (to add)
│   ├── autograd.cpp     # (to add)
│   ├── ops.cpp          # (to add)
│   ├── nn/              # Linear, Embedding, LayerNorm, etc.
│   ├── optim/           # SGD, AdamW
│   ├── data/            # Dataset, DataLoader
│   └── utils/           # init, checkpoint
└── tests/
```

## Next Steps

1. Implement `Tensor` (storage, shape, dtype, requires_grad)
2. Implement Autograd engine (Node, backward, topo sort)
3. Add ops (add, mul, matmul, softmax, etc.)
4. Add nn layers (Linear, Embedding, LayerNorm, GELU, Attention)
5. Add optimizers (SGD, AdamW)
6. Add DataLoader for token sequences
7. Build miniGPT and train
