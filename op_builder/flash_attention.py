import os
from .builder import Builder
from .utils import append_nvcc_threads, get_cuda_cc_flag


class FlashAttentionBuilder(Builder):
    NAME = "flash_attention"
    PREBUILT_IMPORT_PATH = "colossalai._C.flash_attention"

    def __init__(self):
        super().__init__(name=FlashAttentionBuilder.NAME, prebuilt_import_path=FlashAttentionBuilder.PREBUILT_IMPORT_PATH)
        
    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname) for fname in [
                'fmha_api.cpp', 
                'src/fmha_fwd_hdim32.cu',
                'src/fmha_fwd_hdim64.cu',
                'src/fmha_fwd_hdim128.cu',
                'src/fmha_bwd_hdim32.cu',
                'src/fmha_bwd_hdim64.cu',
                'src/fmha_bwd_hdim128.cu',
                'src/fmha_block_fprop_fp16_kernel.sm80.cu',
                'src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu',
            ]
        ]
        return ret

    def include_dirs(self):
        ret = [self.csrc_abs_path('src/fmha'), self.csrc_abs_path('src'),self.csrc_abs_path('cutlass/include'), self.get_cuda_home_include()]
        return ret

    def cxx_flags(self):
        return ['-O3'] + self.version_dependent_macros
        #return ['-O3']


    def nvcc_flags(self):
        extra_cuda_flags = [
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '--ptxas-options=-v']
        ret = ['-O3'] + self.version_dependent_macros + extra_cuda_flags
        #ret = ['-O3', '--use_fast_math'] + extra_cuda_flags
        return append_nvcc_threads(ret)
