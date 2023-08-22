#include "jblas_task_dispatcher.hpp"

#define INTERFACE_TEMPLATE                                            \
  template <class _Launcher_T, template <class _T> class _Parallel_T> \
  class Interface
#define LAUNCHER_TEMPLATE                                                                              \
  template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T, template <class _T, JBLAS_ISA> class _PrologueA_T, \
            template <class _T, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T> \
  class Launcher

inline bool check_amx() { return jblas::utils::parallel::CpuDevice::getInstance()->AMX_BF16(); }
inline bool check_vnni() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX_VNNI(); }
inline bool check_avx512f() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX512F(); }

inline void set_nk(qbits_runtime_ctx* ctx) {
  ctx->n = ctx->transpose ? ctx->weight->sizes()[0] : ctx->weight->sizes()[1];
  ctx->k = ctx->transpose ? ctx->weight->sizes()[1] : ctx->weight->sizes()[0];
}

template <class KERNEL>
void qbits_quantize(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  PrologueB compress_kernel;
  set_nk(ctx);
  auto ptr = (typename PrologueB::StorageWeight*)compress_kernel.createStorage(ctx->n, ctx->k, ctx->blocksize);
  if (ctx->transpose)
    compress_kernel.unpackTransposeWeight(ctx->n, ctx->k, ptr, ctx->output->data_ptr<float>(), ctx->k);
  else
    compress_kernel.unpackWeight(ctx->n, ctx->k, ptr, ctx->output->data_ptr<float>(), ctx->k);
  auto size = ptr->getSerializedSize();
  ctx->output->resize_(size);
  ctx->output->to(torch::kInt8);
  ptr->serializeToBuffer(ctx->output->data_ptr<int8_t>());
}

template <class KERNEL>
void qbits_dequantize(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using PrologueB = typename KERNEL::WeightType;
  PrologueB decompress_kernel;
  set_nk(ctx);
  auto deserial_wei = jblas::prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(
      ctx->weight->data_ptr<float>(), false);
  auto parse_wei = dynamic_cast<typename PrologueB::StorageWeight*>(deserial_wei);
  TORCH_CHECK(parse_wei != nullptr, "unresolved compressed weight.");
  if (ctx->transpose)
    decompress_kernel.packTransposeWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->k, parse_wei);
  else
    decompress_kernel.packWeight(ctx->n, ctx->k, ctx->weight->data_ptr<float>(), ctx->k, parse_wei);
}

template <QBITS_TASK TASK, class KERNEL>
void execute_task(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  switch (TASK) {
    case QBITS_QUANTIZE:
      return qbits_quantize<KERNEL>(p, ctx);
    case QBITS_DEQUANTIZE:
      return qbits_dequantize<KERNEL>(p, ctx);
      // case QBITS_LINEAR:
      //   return jblas_gemm<KERNEL>(p, ctx);
  }
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA, template <class _T, JBLAS_ISA> class PrologueB, template <class _T, JBLAS_ISA> class PrologueA>
void parse_store(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->dst_dt == QBITS_FP32) {
    using namespace jblas::epilogue::gemm;
    return execute_task<TASK,
                        Interface<Launcher<ISA, Gemmcore, PrologueA, PrologueB, AccumulatorWriteBackFp32>, Parallel>>(
        p, ctx);
  }
  TORCH_CHECK(false, "unsupported dst data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA, template <class _T, JBLAS_ISA> class PrologueB>
void parse_activation(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::gemm;
  if (p->compute_type == "int8" && p->src_dt == QBITS_FP32 && check_amx()) {
    return parse_store<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, PrologueB, ActivationF32S8KBlockQuantize>(
        p, ctx);
  }
  TORCH_CHECK(false, "unsupported src data type.");
}

template <QBITS_TASK TASK, INTERFACE_TEMPLATE, LAUNCHER_TEMPLATE, class Gemmcore, template <class _T> class Parallel,
          JBLAS_ISA ISA>
void parse_weight(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  using namespace jblas::prologue::weight_comp::gemm_kblcok;
  if (p->weight_type == "s8_scalef32") {
    return parse_activation<TASK, Interface, Launcher, Gemmcore, Parallel, ISA, WeightS8ScaleFp32>(p, ctx);
  }
  if (p->weight_type == "s4clip_scalef32") {
  }
  if (p->weight_type == "s4fullrange_scalef32") {
  }
  TORCH_CHECK(false, "unsupported weight_type, weight_type==" + p->weight_type);
}

template <QBITS_TASK TASK>
void parse_gemm_core(qbits_config_param* p, qbits_runtime_ctx* ctx) {
  if (p->compute_type == "int8") {
    if (check_amx()) {
      return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                          jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                          jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
                          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAMX_INT8>(p, ctx);
    }
    if (check_vnni()) {
      return parse_weight<TASK, jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight,
                          jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight,
                          jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
                          jblas::utils::parallel::Parallel2DGemmKBlockFixed, JblasAVX512_VNNI>(p, ctx);
    }
    TORCH_CHECK(false, "device ISA muster lagger than VNNI when compute_type==int8");
  }
  if (p->compute_type == "fp32") {
  }
  if (p->compute_type == "bf16") {
  }
  TORCH_CHECK(false, "unsupported compute_type, compute_type==" + p->compute_type);
}

void task_dispatcher(qbits_config_param* p, qbits_runtime_ctx* ctx, QBITS_TASK task) {
  if (task == QBITS_QUANTIZE) return parse_gemm_core<QBITS_QUANTIZE>(p, ctx);
  if (task == QBITS_DEQUANTIZE) return parse_gemm_core<QBITS_DEQUANTIZE>(p, ctx);
  if (task == QBITS_LINEAR) return parse_gemm_core<QBITS_LINEAR>(p, ctx);
}