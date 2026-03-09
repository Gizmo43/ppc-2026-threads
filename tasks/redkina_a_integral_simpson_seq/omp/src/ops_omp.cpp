#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace redkina_a_integral_simpson_seq {

namespace {

int GetSimpsonWeight(int idx, int max_idx) {
  if (idx == 0 || idx == max_idx) {
    return 1;
  }
  if (idx % 2 == 1) {
    return 4;
  }
  return 2;
}

bool AdvanceIndices(std::vector<int> &indices, const std::vector<int> &n) {
  int dim = static_cast<int>(indices.size());
  int pos = dim - 1;
  while (pos >= 0 && indices[pos] == n[pos]) {
    indices[pos] = 0;
    --pos;
  }
  if (pos < 0) {
    return false;
  }
  ++indices[pos];
  return true;
}

double TraverseRemainingDimensions(const std::vector<double>& sub_a,
                                   const std::vector<double>& sub_h,
                                   const std::vector<int>& sub_n,
                                   double w_first,
                                   const std::function<double(const std::vector<double>&)>& func,
                                   std::vector<double>& point,
                                   size_t sub_dim) {
  if (sub_dim == 0) {
    return 0.0;
  }
  double sum = 0.0;
  std::vector<int> indices(sub_dim, 0);
  bool has_next = true;
  while (has_next) {
    double w_prod = w_first;
    for (size_t i = 0; i < sub_dim; ++i) {
      int idx = indices[i];
      point[i + 1] = sub_a[i] + (static_cast<double>(idx) * sub_h[i]);
      int w = GetSimpsonWeight(idx, sub_n[i]);
      w_prod *= static_cast<double>(w);
    }
    sum += w_prod * func(point);
    has_next = AdvanceIndices(indices, sub_n);
  }
  return sum;
}

double ProcessSubspace(const std::vector<double>& a,
                       const std::vector<double>& h,
                       const std::vector<int>& n,
                       int first_index,
                       const std::function<double(const std::vector<double>&)>& func,
                       size_t dim) {
  if (dim == 0) {
    return 0.0;
  }
  std::vector<double> point(dim);
  point[0] = a[0] + (static_cast<double>(first_index) * h[0]);

  int w_first = GetSimpsonWeight(first_index, n[0]);

  if (dim == 1) {
    return static_cast<double>(w_first) * func(point);
  }

  size_t sub_dim = dim - 1;
  std::vector<int> sub_n(sub_dim);
  std::vector<double> sub_h(sub_dim);
  std::vector<double> sub_a(sub_dim);

  for (size_t dim_idx = 1; dim_idx < dim; ++dim_idx) {
    sub_n[dim_idx - 1] = n[dim_idx];
    sub_h[dim_idx - 1] = h[dim_idx];
    sub_a[dim_idx - 1] = a[dim_idx];
  }

  double sub_sum = TraverseRemainingDimensions(sub_a, sub_h, sub_n,
                                               static_cast<double>(w_first),
                                               func, point, sub_dim);
  return sub_sum;
}

}  // namespace

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();
  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;
  result_ = 0.0;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  size_t dim = a_.size();

  std::vector<double> h(dim);
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
  }

  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h_prod *= h[i];
  }

  double total_sum = 0.0;
  int n0 = n_[0];

  const auto& a_ref = a_;
  const auto& h_ref = h;
  const auto& n_ref = n_;
  const auto& func_ref = func_;
  size_t dim_val = dim;
  int n0_val = n0;

#pragma omp parallel for reduction(+ : total_sum) schedule(static) default(none) \
    shared(a_ref, h_ref, n_ref, func_ref, dim_val, n0_val) num_threads(ppc::util::GetNumThreads())
  for (int i0 = 0; i0 <= n0_val; ++i0) {
    total_sum += ProcessSubspace(a_ref, h_ref, n_ref, i0, func_ref, dim_val);
  }

  double denominator = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    denominator *= 3.0;
  }

  result_ = (h_prod / denominator) * total_sum;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq