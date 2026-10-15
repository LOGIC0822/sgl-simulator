#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/python/sglang/kernels/jit/csrc/elementwise/activation.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation, (ActivationKernel<fp16_t, false>::run_activation));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation_with_rounding, (ActivationKernel<fp16_t, false>::run_activation_with_rounding));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation_with_rounding_input_inplace, (ActivationKernel<fp16_t, false>::run_activation_with_rounding_input_inplace));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_activation_filtered, (ActivationKernel<fp16_t, false>::run_activation_filtered));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_unary_activation, (ActivationKernel<fp16_t, false>::run_unary_activation));
}  // namespace sglang
