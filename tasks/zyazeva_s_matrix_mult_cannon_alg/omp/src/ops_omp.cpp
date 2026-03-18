#include "zyazeva_s_matrix_mult_cannon_alg/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "zyazeva_s_matrix_mult_cannon_alg/common/include/common.hpp"

namespace zyazeva_s_matrix_mult_cannon_alg {

bool ZyazevaSMatrixMultCannonAlgOMP::IsPerfectSquare(int x) {
  int root = static_cast<int>(std::sqrt(x));
  return root * root == x;
}

void ZyazevaSMatrixMultCannonAlgOMP::MultiplyBlocks(const std::vector<double> &a, const std::vector<double> &b,
                                                    std::vector<double> &c, int block_size) {
  for (int i = 0; i < block_size; ++i) {
    for (int k = 0; k < block_size; ++k) {
      double a_ik = a[static_cast<size_t>(i) * block_size + k];
      for (int j = 0; j < block_size; ++j) {
        c[static_cast<size_t>(i) * block_size + j] += a_ik * b[static_cast<size_t>(k) * block_size + j];
      }
    }
  }
}

ZyazevaSMatrixMultCannonAlgOMP::ZyazevaSMatrixMultCannonAlgOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ZyazevaSMatrixMultCannonAlgOMP::ValidationImpl() {
  const size_t sz = std::get<0>(GetInput());
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  return sz > 0 && m1.size() == sz * sz && m2.size() == sz * sz;
}

bool ZyazevaSMatrixMultCannonAlgOMP::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool ZyazevaSMatrixMultCannonAlgOMP::RunImpl() {
  const auto sz = static_cast<int>(std::get<0>(GetInput()));
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  std::vector<double> res_m(static_cast<size_t>(sz) * sz, 0.0);

  int num_threads = 1;
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }

  if (!IsPerfectSquare(num_threads) || sz < num_threads) {
#pragma omp parallel for default(none) shared(m1, m2, res_m, sz)
    for (int i = 0; i < sz; ++i) {
      for (int j = 0; j < sz; ++j) {
        double sum = 0.0;
        for (int k = 0; k < sz; ++k) {
          sum += m1[static_cast<size_t>(i) * sz + k] * m2[static_cast<size_t>(k) * sz + j];
        }
        res_m[static_cast<size_t>(i) * sz + j] = sum;
      }
    }
    GetOutput() = res_m;
    return true;
  }

  int grid_size = static_cast<int>(std::sqrt(num_threads));

  if (sz % grid_size != 0) {
#pragma omp parallel for default(none) shared(m1, m2, res_m, sz)
    for (int i = 0; i < sz; ++i) {
      for (int j = 0; j < sz; ++j) {
        double sum = 0.0;
        for (int k = 0; k < sz; ++k) {
          sum += m1[static_cast<size_t>(i) * sz + k] * m2[static_cast<size_t>(k) * sz + j];
        }
        res_m[static_cast<size_t>(i) * sz + j] = sum;
      }
    }
    GetOutput() = res_m;
    return true;
  }

  int block_size = sz / grid_size;

  std::vector<std::vector<double>> blocks_a(static_cast<size_t>(grid_size) * grid_size);
  std::vector<std::vector<double>> blocks_b(static_cast<size_t>(grid_size) * grid_size);
  std::vector<std::vector<double>> blocks_c(static_cast<size_t>(grid_size) * grid_size,
                                            std::vector<double>(static_cast<size_t>(block_size) * block_size, 0.0));

  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      size_t block_idx = static_cast<size_t>(i) * grid_size + j;
      blocks_a[block_idx].resize(static_cast<size_t>(block_size) * block_size);
      blocks_b[block_idx].resize(static_cast<size_t>(block_size) * block_size);
      for (int bi = 0; bi < block_size; ++bi) {
        for (int bj = 0; bj < block_size; ++bj) {
          size_t global_i = static_cast<size_t>(i) * block_size + bi;
          size_t global_j = static_cast<size_t>(j) * block_size + bj;
          size_t local_idx = static_cast<size_t>(bi) * block_size + bj;

          blocks_a[block_idx][local_idx] = m1[global_i * sz + global_j];
          blocks_b[block_idx][local_idx] = m2[global_i * sz + global_j];
        }
      }
    }
  }

  std::vector<std::vector<double>> aligned_a(static_cast<size_t>(grid_size) * grid_size);
  std::vector<std::vector<double>> aligned_b(static_cast<size_t>(grid_size) * grid_size);

#pragma omp parallel for default(none) shared(blocks_a, blocks_b, aligned_a, aligned_b, grid_size) collapse(2)
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      size_t block_idx = static_cast<size_t>(i) * grid_size + j;

      size_t a_src_idx = static_cast<size_t>(i) * grid_size + (j + i) % grid_size;
      aligned_a[block_idx] = blocks_a[a_src_idx];

      size_t b_src_idx = static_cast<size_t>((i + j) % grid_size) * grid_size + j;
      aligned_b[block_idx] = blocks_b[b_src_idx];
    }
  }

  for (int step = 0; step < grid_size; ++step) {
#pragma omp parallel for default(none) shared(aligned_a, aligned_b, blocks_c, grid_size, block_size) collapse(2)
    for (int i = 0; i < grid_size; ++i) {
      for (int j = 0; j < grid_size; ++j) {
        size_t block_idx = static_cast<size_t>(i) * grid_size + j;
        MultiplyBlocks(aligned_a[block_idx], aligned_b[block_idx], blocks_c[block_idx], block_size);
      }
    }

    if (step < grid_size - 1) {
      std::vector<std::vector<double>> new_aligned_a(static_cast<size_t>(grid_size) * grid_size);
      std::vector<std::vector<double>> new_aligned_b(static_cast<size_t>(grid_size) * grid_size);

#pragma omp parallel for default(none) shared(aligned_a, aligned_b, new_aligned_a, new_aligned_b, grid_size) collapse(2)
      for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
          size_t block_idx = static_cast<size_t>(i) * grid_size + j;

          size_t a_src_idx = static_cast<size_t>(i) * grid_size + (j + 1) % grid_size;
          new_aligned_a[block_idx] = aligned_a[a_src_idx];

          size_t b_src_idx = static_cast<size_t>((i + 1) % grid_size) * grid_size + j;
          new_aligned_b[block_idx] = aligned_b[b_src_idx];
        }
      }

      aligned_a = new_aligned_a;
      aligned_b = new_aligned_b;
    }
  }

#pragma omp parallel for default(none) shared(blocks_c, res_m, grid_size, block_size, sz) collapse(2)
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      size_t block_idx = static_cast<size_t>(i) * grid_size + j;
      const auto &block = blocks_c[block_idx];

      for (int bi = 0; bi < block_size; ++bi) {
        for (int bj = 0; bj < block_size; ++bj) {
          size_t global_i = static_cast<size_t>(i) * block_size + bi;
          size_t global_j = static_cast<size_t>(j) * block_size + bj;
          size_t local_idx = static_cast<size_t>(bi) * block_size + bj;

          res_m[global_i * sz + global_j] = block[local_idx];
        }
      }
    }
  }

  GetOutput() = res_m;
  return true;
}

bool ZyazevaSMatrixMultCannonAlgOMP::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_matrix_mult_cannon_alg
