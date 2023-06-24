from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

# setup(
#     name='numpy_test',
#     version='2.8',
#     author='lee',
#     description='numpy_test',
#     long_description='numpy_test',
#     ext_modules=[
#         CUDAExtension(
#             name='numpy_test',
#             include_dirs=['./include', "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v11.3\\common\\inc"],
#             sources=['cal_every.cpp', 'cal_every_cuda.cu']
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

setup(
    name='gascore',
    version='3.1',
    author='lee',
    description='gascore',
    long_description='gascore',
    ext_modules=[
        CppExtension(
            name='gascore',
            include_dirs=['E:/virtualenv_py37/torch1_11_cuda_11_3/Lib/site-packages/pybind11/include',
                          'E:/virtualenv_py37/torch1_11_cuda_11_3/Lib/site-packages/torch/include/**',
                          'E:/Program Files (x86)/Python37/include'],
            sources=['cal_score.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


