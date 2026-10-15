#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/python/sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gptq_marlin_gemm, (gptq_marlin_gemm<fp16_t>));
}  // namespace sglang
