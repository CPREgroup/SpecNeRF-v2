#include <iostream>
#include "utils.h"

namespace py=pybind11;


py::array_t<float> test(py::array_t<float> arr1, py::array_t<float> arr2){
    py::buffer_info arr1buf = arr1.request();
    py::buffer_info arr2buf = arr2.request();
    // auto dir_arr1 = arr1.unchecked<2>();
    // auto dir_arr2 = arr2.unchecked<2>();

    auto result = py::array_t<float>(arr1buf.size);
    result.resize(arr1buf.shape);
    // auto dir_res = result.mutable_unchecked<2>();
    py::buffer_info res1 = result.request();

    // for (py::size_t i = 0; i < result.shape(0); i++)
    // {
    //     for (py::size_t j = 0; j < result.shape(1); j++)
    //     {
    //         dir_res(i, j) = dir_arr1(i, j) + dir_arr2(i, j);
    //     }
        
    // }

    int ti = 1;
    for(int i : arr1buf.shape){
        ti *= i;
    }

    float *ptr1 = static_cast<float *>(arr1buf.ptr);   
    float *ptr2 = static_cast<float *>(arr2buf.ptr);  
    float *ptr3 = static_cast<float *>(res1.ptr);  
    
    for (size_t i = 0; i < ti; i++)
    {
        ptr3[i] = ptr1[i] + ptr2[i];
    }
    
    
    return result;
    
}


py::array_t<float> test_cuda(py::array_t<float> arr1, py::array_t<float> arr2){
    py::buffer_info arr1_buf = arr1.request(), arr2_buf = arr2.request();

    auto result = py::array_t<float>(arr1.size());
    result.resize(arr1_buf.shape);
    py::buffer_info res_buf = result.request();

    // std::cout << "in cpp" << std::endl;

    test_cuda_cu((float*)arr1_buf.ptr,(float*)arr2_buf.ptr,(float*)res_buf.ptr, arr1_buf.shape[0]);

    return result;
}

// void initarr(py::buffer_info *arr, int bd){
//     float *ptr = static_cast<float *>(arr->ptr);
//     for (py::size_t i = 0; i < bd * bd; i++)
//     {
//         ptr[i] = 2.0;
//     }
    
// }

// int main(){
//     std::cout << "begin!" << std::endl;

//     int bd = 100;
//     py::array_t<float> arr1 = py::array_t<float>(bd *bd);
//     py::array_t<float> arr2 = py::array_t<float>(bd *bd);
//     std::cout << "begin1!" << std::endl;
//     py::buffer_info arr1_buf = arr1.request(), arr2_buf = arr2.request();
//     arr1.resize({bd, bd});
//     arr2.resize({bd, bd});
//     std::cout << "begin2!" << std::endl;
//     initarr(&arr1_buf, bd);
//     initarr(&arr2_buf, bd);
//     std::cout << "checkarr1!" << std::endl;
//     std::cout << arr1.index_at(2, 2) << std::endl;

//     // auto result = py::array_t<float>(arr1.size());
//     // result.resize(arr1_buf.shape);
//     // py::buffer_info res_buf = result.request();

//     // test_cuda_cu(&arr1_buf,&arr2_buf,&res_buf);
//     // auto res = test(arr1, arr2);
//     // std::cout << res.index_at(2, 2) << std::endl;

//     return 0;
// }




PYBIND11_MODULE(numpy_test, m) {
	m.def("test", &test, "test", py::return_value_policy::reference);
    m.def("test_cuda", &test_cuda, "test_cuda", py::return_value_policy::reference);
}


