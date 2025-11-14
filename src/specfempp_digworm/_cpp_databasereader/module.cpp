#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "io/interface.hpp"
#include "quadrature/interface.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;

PYBIND11_MODULE(_databasereader, m, py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

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

  m.def(
      "gll",
      [](type_real alpha, type_real beta, int n) {
        constexpr size_t real_size = sizeof(type_real);
        const auto quad = specfem::quadrature::gll::gll(0,0);

        // auto arr = py::array_t<double>(n);
        // auto view = arr.mutable_unchecked<1>();
        
        // for (int i = 0; i < n; i++) {
        //   view(i) = quad.get_xi()(i);
        // }
        // return arr;
        // return quad.get_xi()(0);
        return 1;
      },
      R"pbdoc(
        Computes gll quadrature points
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}