#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
    // std::cout << "error" << std::endl;
      std::cout << stderr << "GPUassert: " << cudaGetErrorString(code)<<file<< line << std::endl;
    //   "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void add(float *x, float *y, float *z, int border){
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    // int j = blockIdx.y * blockDim.y + threadIdx.y;

    // if (i < border*border){
        // int ptr = border * i + j;
        // z[i] = x[i] + y[i];
    // }
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < border*border; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}


void test_cuda_cu(float *arr1_buf, 
                float *arr2_buf,
                float *arr3_buf,
                int bd){

    int nBytes = bd*bd*sizeof(float);
    
    // alloc
    float *x, *y, *z;
    gpuErrchk(cudaMalloc((void**)&x, nBytes));
    gpuErrchk(cudaMalloc((void**)&y, nBytes));
    gpuErrchk(cudaMalloc((void**)&z, nBytes));

    gpuErrchk(cudaMemcpy(x, arr1_buf, nBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(y, arr2_buf, nBytes, cudaMemcpyHostToDevice));

    // conf kernel
    dim3 block(256);
    dim3 grid((int)ceil(bd*bd / block.x));
    // exec
    add<<<grid, block>>>(x, y, z, bd);
    // sync
    // cudaDeviceSynchronize();
    // check
    // std::cout << "item " << z[2] << std::endl;
    // py::print(z[2]);
    // copy
    gpuErrchk(cudaMemcpy(arr3_buf, z, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
}

void initarr(float *arr, int bd, float val){
    for (int i = 0; i < bd * bd; i++)
    {
        arr[i] = val;
    }
    
}
int main(){
    int bd = 10;
    float a[100] = {2.0}; initarr(a, bd, 2.0);
    float b[100] = {1.0}; initarr(b, bd, 1.0);
    float c[100] = {0.0}; initarr(c, bd, 0.0);

    std::cout << b[3] << std::endl;

    test_cuda_cu(a, b, c, bd);

    for(float i : c){
        std::cout << i;
    }


    return 0;
}





