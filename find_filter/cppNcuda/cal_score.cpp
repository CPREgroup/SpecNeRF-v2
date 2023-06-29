#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py=pybind11;
typedef py::array_t<double> arrf64;
typedef py::array_t<int> arrint;
typedef py::detail::unchecked_reference<double, 2i64> unc;
typedef py::detail::unchecked_reference<double, 1i64> unc1;

double cal_idv(arrint &idv, const unc &dis_mtx_r,
            const unc &ssf_coor_r, const unc &cosSimi_r, 
            const unc1 &ssf_trans_r, const double weight){
    auto idv_r = idv.unchecked<2>();
    double score = 0;
    double transsum = 0;
    int number = idv.shape(0);


    for (py::ssize_t i = 1; i < number; i++)
    {
        int r1 = idv_r(i, 0);
        int c1 = idv_r(i, 1);
        for (int j = 0; j < i; j++)
        {
            int r2 = idv_r(j, 0);
            int c2 = idv_r(j, 1);
            // std::cout << r1 << "," << c1<<","<<r2<<","<<c2<<std::endl;

            double dis = dis_mtx_r(r1, r2);
            double coorela = ssf_coor_r(c1, c2);
            double viewdir_coorela = cosSimi_r(r1, r2);
            score += dis * coorela * viewdir_coorela;
        }
    }

    // for (py::ssize_t i = 0; i < number; i++)
    // {
    //     transsum += ssf_trans_r(idv_r(i, 1));
    // }
    
    
    return 1 / score; // + weight * transsum;
}


arrf64 cal_score(const py::list &all, const arrf64 &dis_mtx,
            const arrf64 &ssf_coor, const arrf64 &cosSimi, 
            const arrf64 &ssf_trans, const double weight){
    int idx = 0;
    auto dis_mtx_r = dis_mtx.unchecked<2>();
    auto ssf_coor_r = ssf_coor.unchecked<2>();
    auto cosSimi_r = cosSimi.unchecked<2>();
    auto ssf_trans_r = ssf_trans.unchecked<1>();
    
    try
    {
        auto res = arrf64(py::len(all));
        auto res_r = res.mutable_unchecked<1>();
        for (auto idv_t : all){
            arrint idv = py::reinterpret_borrow<arrint>(idv_t);
            // py::print(idv);
            res_r(idx++) = cal_idv(idv, dis_mtx_r, ssf_coor_r, cosSimi_r, ssf_trans_r, weight);
        }
        return res;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}


PYBIND11_MODULE(gascore, m) {
	m.def("cal_score", &cal_score, "cal_score");
}

