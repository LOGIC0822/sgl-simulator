#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/python/sglang/kernels/jit/csrc/elementwise/rope.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_rope, (FusedRopeKernel<true, 128, false, bf16_t, bf16_t>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_rope_store, (FusedRopeKernel<true, 128, false, bf16_t, bf16_t>::run_fused));
}  // namespace sglang
