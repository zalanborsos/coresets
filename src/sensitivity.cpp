#include "sensitivity.h"

#include <algorithm>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

namespace {

const char* k_means_doc = R"(alma)";

}  // namespace

py::array_t<double> kmeans_sensitivity(const py::array_t<double>& points_,
                                      const py::array_t<double>& centers_,
                                      const double alpha) {
  if (points_.shape(1) != centers_.shape(1)) {
    throw std::runtime_error(
        "Data points and centers should have the save dimension.");
  }
  auto points = points_.unchecked<2>();
  auto centers = centers_.unchecked<2>();

  std::vector<double> dists(points.shape(0));
  std::vector<double> assign(points.shape(0));
  std::vector<double> centers_cnt(centers.shape(0));
  std::vector<double> centers_total_dist(centers.shape(0));
  double total = 0;
  int n = points.shape(0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < centers.shape(0); j++) {
      double dist = 0;
      for (int k = 0; k < points.shape(1); k++) {
        dist += pow(points(i, k) - centers(j, k), 2);
      }
      if (j == 0 || dist < dists[i]) {
        dists[i] = dist;
        assign[i] = j;
      }
    }
  }

  for (int i = 0; i < centers.shape(0); i++) {
    centers_cnt[i] = 0;
    centers_total_dist[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    total += dists[i];
    centers_cnt[assign[i]]++;
    centers_total_dist[assign[i]] += dists[i];
  }
  total /= points.shape(0);

  py::array_t<double> sens_ = py::array_t<double>(points.shape(0));
  auto sens = sens_.mutable_unchecked<1>();

  for (int i = 0; i < n; i++) {
    sens(i) = 2 * alpha * dists[i] / total;
    sens(i) += 4 * alpha * centers_total_dist[assign[i]] /
               (centers_cnt[assign[i]] * total);
    sens(i) += 4 * n / centers_cnt[assign[i]];
  }
  return sens_;
}

PYBIND11_MODULE(sensitivity, m) {
  py::options options;
  options.disable_function_signatures();
  m.def("kmeans_sensitivity", kmeans_sensitivity, k_means_doc);
}
