#pragma once

#include <pybind11/numpy.h>

pybind11::array_t<double> kmeans_sensitivity(const pybind11::array_t<double>& points,
                                   const pybind11::array_t<double>& weights_,
                                   const pybind11::array_t<double>& centers,
                                   const double alpha);