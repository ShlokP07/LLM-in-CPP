#include <llm/checkpoint.hpp>
#include <llm/tensor.hpp>
#include <llm/dtype.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace llm {

namespace {

constexpr const char* TENSOR_MAGIC = "LLT1";
constexpr const char* STATE_MAGIC = "LLS1";
constexpr int32_t VERSION = 1;

int32_t dtype_to_int(DType dt) {
  switch (dt) {
    case DType::Float32: return 0;
    case DType::Int64: return 1;
    default: return -1;
  }
}

DType int_to_dtype(int32_t v) {
  if (v == 0) return DType::Float32;
  if (v == 1) return DType::Int64;
  throw std::runtime_error("checkpoint: invalid dtype in file");
}

}  // namespace

void save_tensor(const std::string& path, const Tensor& tensor) {
  if (!tensor.impl())
    throw std::invalid_argument("save_tensor: null tensor");
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("save_tensor: cannot open " + path);

  const auto& shape = tensor.shape();
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int32_t dtype_val = dtype_to_int(tensor.dtype());
  const std::size_t data_bytes = static_cast<std::size_t>(tensor.numel()) * element_size(tensor.dtype());

  out.write(TENSOR_MAGIC, 4);
  out.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
  out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
  out.write(reinterpret_cast<const char*>(shape.data()), static_cast<std::streamsize>(ndim * sizeof(int64_t)));
  out.write(reinterpret_cast<const char*>(&dtype_val), sizeof(dtype_val));
  if (tensor.dtype() == DType::Float32)
    out.write(reinterpret_cast<const char*>(tensor.data_float()), static_cast<std::streamsize>(data_bytes));
  else
    out.write(reinterpret_cast<const char*>(tensor.data_int64()), static_cast<std::streamsize>(data_bytes));
  if (!out)
    throw std::runtime_error("save_tensor: write failed");
}

Tensor load_tensor(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("load_tensor: cannot open " + path);

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || std::memcmp(magic, TENSOR_MAGIC, 4) != 0)
    throw std::runtime_error("load_tensor: invalid magic");

  int32_t version = 0;
  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != VERSION)
    throw std::runtime_error("load_tensor: unsupported version");

  int64_t ndim = 0;
  in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
  if (ndim < 0 || ndim > 64)
    throw std::runtime_error("load_tensor: invalid ndim");

  std::vector<int64_t> shape(static_cast<size_t>(ndim));
  if (ndim > 0)
    in.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(ndim * sizeof(int64_t)));

  int32_t dtype_val = 0;
  in.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
  DType dtype = int_to_dtype(dtype_val);

  Tensor t = Tensor::zeros(shape, dtype, Device::cpu(), false);
  const std::size_t data_bytes = static_cast<std::size_t>(t.numel()) * element_size(dtype);
  if (dtype == DType::Float32)
    in.read(reinterpret_cast<char*>(t.data_float()), static_cast<std::streamsize>(data_bytes));
  else
    in.read(reinterpret_cast<char*>(t.data_int64()), static_cast<std::streamsize>(data_bytes));
  if (!in)
    throw std::runtime_error("load_tensor: read failed");
  return t;
}

void save_state_dict(const std::string& path, const Module::StateDict& state) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("save_state_dict: cannot open " + path);

  out.write(STATE_MAGIC, 4);
  out.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
  const int64_t num_keys = static_cast<int64_t>(state.size());
  out.write(reinterpret_cast<const char*>(&num_keys), sizeof(num_keys));

  for (const auto& kv : state) {
    const std::string& key = kv.first;
    const Tensor& tensor = kv.second;
    const int64_t key_len = static_cast<int64_t>(key.size());
    out.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
    out.write(key.data(), static_cast<std::streamsize>(key.size()));

    const auto& shape = tensor.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int32_t dtype_val = dtype_to_int(tensor.dtype());
    out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    out.write(reinterpret_cast<const char*>(shape.data()), static_cast<std::streamsize>(ndim * sizeof(int64_t)));
    out.write(reinterpret_cast<const char*>(&dtype_val), sizeof(dtype_val));
    const std::size_t data_bytes = static_cast<std::size_t>(tensor.numel()) * element_size(tensor.dtype());
    if (tensor.dtype() == DType::Float32)
      out.write(reinterpret_cast<const char*>(tensor.data_float()), static_cast<std::streamsize>(data_bytes));
    else
      out.write(reinterpret_cast<const char*>(tensor.data_int64()), static_cast<std::streamsize>(data_bytes));
  }
  if (!out)
    throw std::runtime_error("save_state_dict: write failed");
}

Module::StateDict load_state_dict(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("load_state_dict: cannot open " + path);

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || std::memcmp(magic, STATE_MAGIC, 4) != 0)
    throw std::runtime_error("load_state_dict: invalid magic");

  int32_t version = 0;
  in.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != VERSION)
    throw std::runtime_error("load_state_dict: unsupported version");

  int64_t num_keys = 0;
  in.read(reinterpret_cast<char*>(&num_keys), sizeof(num_keys));
  if (num_keys < 0)
    throw std::runtime_error("load_state_dict: invalid num_keys");

  Module::StateDict state;
  for (int64_t k = 0; k < num_keys; ++k) {
    int64_t key_len = 0;
    in.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
    if (key_len < 0 || key_len > 1024 * 1024)
      throw std::runtime_error("load_state_dict: invalid key_len");
    std::string key(static_cast<size_t>(key_len), '\0');
    in.read(&key[0], static_cast<std::streamsize>(key_len));

    int64_t ndim = 0;
    in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (ndim < 0 || ndim > 64)
      throw std::runtime_error("load_state_dict: invalid ndim");
    std::vector<int64_t> shape(static_cast<size_t>(ndim));
    if (ndim > 0)
      in.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(ndim * sizeof(int64_t)));

    int32_t dtype_val = 0;
    in.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
    DType dtype = int_to_dtype(dtype_val);

    Tensor t = Tensor::zeros(shape, dtype, Device::cpu(), false);
    const std::size_t data_bytes = static_cast<std::size_t>(t.numel()) * element_size(dtype);
    if (dtype == DType::Float32)
      in.read(reinterpret_cast<char*>(t.data_float()), static_cast<std::streamsize>(data_bytes));
    else
      in.read(reinterpret_cast<char*>(t.data_int64()), static_cast<std::streamsize>(data_bytes));

    state.emplace(std::move(key), std::move(t));
  }
  if (!in)
    throw std::runtime_error("load_state_dict: read failed");
  return state;
}

}  // namespace llm
