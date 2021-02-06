#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <plasma/client.h>
#include <vovp/ndarray.h>
#include <dlpack/dlpack.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(_vovp, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    py::class_<vovp::runtime::NDArray>(m, "NDArray")
    .def("from_dlpack", [](py::capsule capsule){
        DLManagedTensor* dltensor_ptr = reinterpret_cast<DLManagedTensor*>(capsule.get_pointer());
        auto ndarray = vovp::runtime::NDArray();
        ndarray.FromDLPack(dltensor_ptr);
        return ndarray;
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
