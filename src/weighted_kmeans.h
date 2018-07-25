#pragma once

#include <pybind11/numpy.h>

pybind11::tuple assignment_inertia(const pybind11::array_t<double>& points,
                                   const pybind11::array_t<double>& centers);

pybind11::array_t<double> update_centers(const pybind11::array_t<double>& points_,
                               const pybind11::array_t<double>& weights_,
                               const pybind11::array_t<double>& centers_,
                               const pybind11::array_t<int>& assignments_);