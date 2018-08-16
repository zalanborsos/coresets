#include "weighted_kmeans.h"

#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;
using namespace std;
namespace {

const char* assignment_inertia_doc = R"(assignment_inertia_doc)";
const char* update_centers_doc = R"(update_centers_doc)";

} // namespace

pybind11::tuple assignment_inertia(const pybind11::array_t<double>& points_,
    const pybind11::array_t<double>& centers_)
{
    if (points_.shape(1) != centers_.shape(1)) {
        throw std::runtime_error(
            "Data points and centers should have the save dimension.");
    }
    auto points = points_.unchecked<2>();
    auto centers = centers_.unchecked<2>();

    py::array_t<double> dists_ = py::array_t<double>(points.shape(0));
    auto dists = dists_.mutable_unchecked<1>();

    py::array_t<int> assign_ = py::array_t<int>(points.shape(0));
    auto assign = assign_.mutable_unchecked<1>();

    int n = points.shape(0);

    // determine the cluster assignments for points and calculate
    // the squared Euclidean distance the closest centers
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < centers.shape(0); j++) {
            double dist = 0;
            for (int k = 0; k < points.shape(1); k++) {
                dist += pow(points(i, k) - centers(j, k), 2);
            }
            if (j == 0 || dist < dists[i]) {
                dists(i) = dist;
                assign(i) = j;
            }
        }
    }

    return py::make_tuple(assign_, dists_);
}

pybind11::array_t<double> update_centers(const pybind11::array_t<double>& points_,
    const pybind11::array_t<double>& weights_,
    const pybind11::array_t<double>& centers_,
    const pybind11::array_t<int>& assignments_)
{
    auto points = points_.unchecked<2>();
    auto weights = weights_.unchecked<1>();
    auto assignments = assignments_.unchecked<1>();

    py::array_t<double> new_centers_({ centers_.shape(0), centers_.shape(1) });
    auto new_centers = new_centers_.mutable_unchecked<2>();

    int n = points.shape(0);
    int d = new_centers.shape(1);
    int n_clusters = centers_.shape(0);

    std::vector<double> cnt(n_clusters);

    for (int i = 0; i < n_clusters; i++) {
        for (int j = 0; j < d; j++) {
            new_centers(i, j) = 0;
        }
        cnt[i] = 0;
    }

    //calculate the updated centers based on the new assignments
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            new_centers(assignments(i), j) += points(i, j) * weights(i);
        }
        cnt[assignments(i)] += weights(i);
    }

    for (int i = 0; i < n_clusters; i++) {
        for (int j = 0; j < d; j++) {
            new_centers(i, j) /= cnt[i];
        }
    }

    return new_centers_;
}

PYBIND11_MODULE(weighted_kmeans_, m) {
    py::options options;
    options.disable_function_signatures();
    m.def("assignment_inertia", assignment_inertia, assignment_inertia_doc);
    m.def("update_centers", update_centers, update_centers_doc);
}
