#include "nikitina_v_hoar_sort_batcher/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace nikitina_v_hoar_sort_batcher {

namespace {

void CompareSplit(std::vector<int> &arr, int start1, int len1, int start2, int len2) {
  std::vector<int> temp(len1 + len2);
  int idx_i = start1;
  int idx_j = start2;
  int idx_k = 0;

  while (idx_i < start1 + len1 && idx_j < start2 + len2) {
    if (arr[idx_i] <= arr[idx_j]) {
      temp[idx_k++] = arr[idx_i++];
    } else {
      temp[idx_k++] = arr[idx_j++];
    }
  }
  while (idx_i < start1 + len1) {
    temp[idx_k++] = arr[idx_i++];
  }
  while (idx_j < start2 + len2) {
    temp[idx_k++] = arr[idx_j++];
  }

  for (int idx = 0; idx < len1; ++idx) {
    arr[start1 + idx] = temp[idx];
  }
  for (int idx = 0; idx < len2; ++idx) {
    arr[start2 + idx] = temp[len1 + idx];
  }
}

void BuildPairs(std::vector<std::pair<int, int>> &pairs, int num_threads, int step_p, int step_k) {
  for (int idx_j = step_k % step_p; idx_j + step_k < num_threads; idx_j += (step_k * 2)) {
    for (int idx_i = 0; idx_i < std::min(step_k, num_threads - idx_j - step_k); idx_i++) {
      if ((idx_j + idx_i) / (step_p * 2) == (idx_j + idx_i + step_k) / (step_p * 2)) {
        pairs.emplace_back(idx_j + idx_i, idx_j + idx_i + step_k);
      }
    }
  }
}

void BatcherMergePhase(std::vector<int> &output, const std::vector<int> &offsets, int num_threads) {
  for (int step_p = 1; step_p < num_threads; step_p *= 2) {
    for (int step_k = step_p; step_k > 0; step_k /= 2) {
      std::vector<std::pair<int, int>> pairs;
      BuildPairs(pairs, num_threads, step_p, step_k);

      int num_pairs = static_cast<int>(pairs.size());

#pragma omp parallel for num_threads(num_threads) default(none) shared(output, offsets, pairs, num_pairs)
      for (int idx = 0; idx < num_pairs; ++idx) {
        int block_a = pairs[idx].first;
        int block_b = pairs[idx].second;
        CompareSplit(output, offsets[block_a], offsets[block_a + 1] - offsets[block_a], offsets[block_b],
                     offsets[block_b + 1] - offsets[block_b]);
      }
    }
  }
}

}  // namespace

HoareSortBatcherOMP::HoareSortBatcherOMP(InType in) : input_(std::move(in)) {}

bool HoareSortBatcherOMP::ValidationImpl() {
  return true;
}

bool HoareSortBatcherOMP::PreProcessingImpl() {
  output_ = input_;
  return true;
}

bool HoareSortBatcherOMP::RunImpl() {
  int n = static_cast<int>(output_.size());
  if (n <= 1) {
    return true;
  }

  int max_threads = omp_get_max_threads();
  int t = 1;
  while (t * 2 <= max_threads && t * 2 <= n) {
    t *= 2;
  }

  if (t == 1) {
    std::ranges::sort(output_);
    return true;
  }

  std::vector<int> offsets(t + 1, 0);
  int base_chunk = n / t;
  int rem = n % t;
  for (int i = 0; i < t; ++i) {
    offsets[i + 1] = offsets[i] + base_chunk + (i < rem ? 1 : 0);
  }

#pragma omp parallel num_threads(t) default(none) shared(output_, offsets)
  {
    int tid = omp_get_thread_num();
    std::sort(output_.begin() + offsets[tid], output_.begin() + offsets[tid + 1]);
  }

  BatcherMergePhase(output_, offsets, t);

  return true;
}

bool HoareSortBatcherOMP::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_hoar_sort_batcher
