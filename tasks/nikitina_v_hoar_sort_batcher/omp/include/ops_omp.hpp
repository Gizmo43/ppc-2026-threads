#pragma once

#include "nikitina_v_hoar_sort_batcher/common/include/common.hpp"

namespace nikitina_v_hoar_sort_batcher {

class HoareSortBatcherOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit HoareSortBatcherOMP(InType in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_;
  OutType output_;
};

}  // namespace nikitina_v_hoar_sort_batcher
