#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_align_block_size, (MoeAlignBlockSizeKernel<int32_t>::run));
}  // namespace sglang
