// CUDA-12 / PyTorch-ABI compatibility extension for PR #33824 experiments.
//
// SGLang's piecewise CUDA-graph runtime needs a non-owning Tensor view for
// intermediate buffers held alive by a graph memory pool.  The upstream
// sglang-kernel wheel in this environment was built against a different
// PyTorch C++ ABI, so this tiny extension is compiled locally with the active
// hbf-cxl PyTorch.  It intentionally exposes only this graph-lifetime helper;
// callers must still fail explicitly for any other unavailable sgl_kernel op.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <vector>

namespace {

at::Tensor weak_ref_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "weak_ref_tensor expects a CUDA tensor");

  // from_blob creates a Tensor with the same pointer/shape/strides but no
  // deleter for the original allocation.  During graph replay that allocation
  // remains owned by the CUDAGraph memory pool; retaining the original Tensor
  // here would otherwise pin every captured intermediate indefinitely.
  return at::from_blob(
      tensor.data_ptr(), tensor.sizes().vec(), tensor.strides().vec(), tensor.options());
}

}  // namespace

TORCH_LIBRARY(sglang_weak_ref_compat, library) {
  library.def("weak_ref_tensor(Tensor tensor) -> Tensor");
  library.impl("weak_ref_tensor", c10::kCUDA, TORCH_FN(weak_ref_tensor));
}
