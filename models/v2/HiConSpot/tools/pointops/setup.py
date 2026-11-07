   
   
   
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from setuptools._distutils.sysconfig import get_config_vars
import torch

   
   
   
   
   
   
   
   
   
   
   
   
   
   
os.environ["CUDA_HOME"] = "/home/linh/.local/cuda-12.9"
os.environ["CUDA_PATH"] = "/home/linh/.local/cuda-12.9"
os.environ["CUDA_INCLUDE_DIRS"] = "/home/linh/.local/cuda-12.9/include"
os.environ["PATH"] = "/home/linh/.local/cuda-12.9/bin:" + os.environ["PATH"]
   
   
   
   
   
   
   
   
   
   

   
   
cuda_12_9_path = "/home/linh/.local/cuda-12.9"
   
   
   
   
   
   
   
   
   
   
   

(opt,) = get_config_vars('OPT')
if opt:
    os.environ['OPT'] = " ".join(
        flag for flag in opt.split() if flag != '-Wstrict-prototypes'
    )

print(os.environ.get('CUDA_HOME'))

setup(
    name='pointops_cuda',
    author='Hengshuang Zhao',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/subtraction/subtraction_cuda.cpp',
            'src/subtraction/subtraction_cuda_kernel.cu',
            'src/aggregation/aggregation_cuda.cpp',
            'src/aggregation/aggregation_cuda_kernel.cu',
            ],
               
            include_dirs=[
                os.path.join(cuda_12_9_path, "include"),     
                'src',     
            ],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2', '--expt-relaxed-constexpr', '--allow-unsupported-compiler']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
