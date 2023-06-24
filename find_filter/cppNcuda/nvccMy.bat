@echo off

nvcc -I "H:\virtualenv\torch111_cuda113\Scripts","C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include","C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64","C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.3\common\inc" -o cal_every_cuda.exe cal_every_cuda.cu