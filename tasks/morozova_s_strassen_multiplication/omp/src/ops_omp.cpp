#include "morozova_s_strassen_multiplication/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "morozova_s_strassen_multiplication/common/include/common.hpp"

namespace morozova_s_strassen_multiplication {

namespace {

Matrix AddMatrixImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = a(i, j) + b(i, j);
    }
  }

  return result;
}

Matrix SubtractMatrixImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = a(i, j) - b(i, j);
    }
  }

  return result;
}

Matrix MultiplyStandardImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(dynamic, 1) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

void SplitMatrixImpl(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21, Matrix &m22) {
  int n = m.size;
  int half = n / 2;

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      m11(i, j) = m(i, j);
      m12(i, j) = m(i, j + half);
      m21(i, j) = m(i + half, j);
      m22(i, j) = m(i + half, j + half);
    }
  }
}

Matrix MergeMatricesImpl(const Matrix &m11, const Matrix &m12, const Matrix &m21, const Matrix &m22) {
  int half = m11.size;
  int n = 2 * half;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(m11, m12, m21, m22, result, half)
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      result(i, j) = m11(i, j);
      result(i, j + half) = m12(i, j);
      result(i + half, j) = m21(i, j);
      result(i + half, j + half) = m22(i, j);
    }
  }

  return result;
}

Matrix MultiplyStandardParallelImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel default(none) shared(a, b, result, n)
  {
#pragma omp for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += a(i, k) * b(k, j);
        }
        result(i, j) = sum;
      }
    }
  }

  return result;
}

void ComputeStrassenLevel(const Matrix &a11, const Matrix &a12, const Matrix &a21, const Matrix &a22, const Matrix &b11,
                          const Matrix &b12, const Matrix &b21, const Matrix &b22, Matrix &c11, Matrix &c12,
                          Matrix &c21, Matrix &c22, int leaf_size, int max_depth, int current_depth) {
  Matrix p1;
  Matrix p2;
  Matrix p3;
  Matrix p4;
  Matrix p5;
  Matrix p6;
  Matrix p7;

  if (current_depth < max_depth) {
#pragma omp parallel sections default(none) \
    shared(a11, a12, a21, a22, b11, b12, b21, b22, leaf_size, max_depth, current_depth, p1, p2, p3, p4, p5, p6, p7)
    {
#pragma omp section
      {
        Matrix temp = SubtractMatrixImpl(b12, b22);
        p1 = MultiplyStandardParallelImpl(a11, temp);
      }

#pragma omp section
      {
        Matrix temp = AddMatrixImpl(a11, a12);
        p2 = MultiplyStandardParallelImpl(temp, b22);
      }

#pragma omp section
      {
        Matrix temp = AddMatrixImpl(a21, a22);
        p3 = MultiplyStandardParallelImpl(temp, b11);
      }

#pragma omp section
      {
        Matrix temp = SubtractMatrixImpl(b21, b11);
        p4 = MultiplyStandardParallelImpl(a22, temp);
      }

#pragma omp section
      {
        Matrix temp1 = AddMatrixImpl(a11, a22);
        Matrix temp2 = AddMatrixImpl(b11, b22);
        p5 = MultiplyStandardParallelImpl(temp1, temp2);
      }

#pragma omp section
      {
        Matrix temp1 = SubtractMatrixImpl(a12, a22);
        Matrix temp2 = AddMatrixImpl(b21, b22);
        p6 = MultiplyStandardParallelImpl(temp1, temp2);
      }

#pragma omp section
      {
        Matrix temp1 = SubtractMatrixImpl(a11, a21);
        Matrix temp2 = AddMatrixImpl(b11, b12);
        p7 = MultiplyStandardParallelImpl(temp1, temp2);
      }
    }
  } else {
    Matrix temp1;
    Matrix temp2;

    temp1 = SubtractMatrixImpl(b12, b22);
    p1 = MultiplyStandardParallelImpl(a11, temp1);

    temp1 = AddMatrixImpl(a11, a12);
    p2 = MultiplyStandardParallelImpl(temp1, b22);

    temp1 = AddMatrixImpl(a21, a22);
    p3 = MultiplyStandardParallelImpl(temp1, b11);

    temp1 = SubtractMatrixImpl(b21, b11);
    p4 = MultiplyStandardParallelImpl(a22, temp1);

    temp1 = AddMatrixImpl(a11, a22);
    temp2 = AddMatrixImpl(b11, b22);
    p5 = MultiplyStandardParallelImpl(temp1, temp2);

    temp1 = SubtractMatrixImpl(a12, a22);
    temp2 = AddMatrixImpl(b21, b22);
    p6 = MultiplyStandardParallelImpl(temp1, temp2);

    temp1 = SubtractMatrixImpl(a11, a21);
    temp2 = AddMatrixImpl(b11, b12);
    p7 = MultiplyStandardParallelImpl(temp1, temp2);
  }

  Matrix temp;

  temp = AddMatrixImpl(p5, p4);
  temp = SubtractMatrixImpl(temp, p2);
  c11 = AddMatrixImpl(temp, p6);

  c12 = AddMatrixImpl(p1, p2);
  c21 = AddMatrixImpl(p3, p4);

  temp = AddMatrixImpl(p5, p1);
  temp = SubtractMatrixImpl(temp, p3);
  c22 = SubtractMatrixImpl(temp, p7);
}

Matrix StrassenMultiplyIterative(const Matrix &a, const Matrix &b, int leaf_size, int max_depth) {
  int n = a.size;

  if (n <= leaf_size || n % 2 != 0) {
    return MultiplyStandardParallelImpl(a, b);
  }

  int half = n / 2;

  Matrix a11(half);
  Matrix a12(half);
  Matrix a21(half);
  Matrix a22(half);
  Matrix b11(half);
  Matrix b12(half);
  Matrix b21(half);
  Matrix b22(half);

  SplitMatrixImpl(a, a11, a12, a21, a22);
  SplitMatrixImpl(b, b11, b12, b21, b22);

  Matrix c11(half);
  Matrix c12(half);
  Matrix c21(half);
  Matrix c22(half);

  ComputeStrassenLevel(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22, leaf_size, max_depth, 0);

  return MergeMatricesImpl(c11, c12, c21, c22);
}

}  // namespace

MorozovaSStrassenMultiplicationOMP::MorozovaSStrassenMultiplicationOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool MorozovaSStrassenMultiplicationOMP::ValidationImpl() {
  return true;
}

bool MorozovaSStrassenMultiplicationOMP::PreProcessingImpl() {
  if (GetInput().empty()) {
    valid_data_ = false;
    return true;
  }

  double size_val = GetInput()[0];
  if (size_val <= 0.0) {
    valid_data_ = false;
    return true;
  }

  int n = static_cast<int>(size_val);

  if (GetInput().size() != 1 + (2 * static_cast<size_t>(n) * static_cast<size_t>(n))) {
    valid_data_ = false;
    return true;
  }

  valid_data_ = true;
  n_ = n;

  a_ = Matrix(n_);
  b_ = Matrix(n_);

  int idx = 1;
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      a_(i, j) = GetInput()[idx++];
    }
  }

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      b_(i, j) = GetInput()[idx++];
    }
  }

  return true;
}

bool MorozovaSStrassenMultiplicationOMP::RunImpl() {
  if (!valid_data_) {
    return true;
  }

  const int leaf_size = 64;

  if (n_ <= leaf_size) {
    c_ = MultiplyStandardParallel(a_, b_);
  } else {
    c_ = StrassenMultiplyIterative(a_, b_, leaf_size, kMaxParallelDepth);
  }

  return true;
}

bool MorozovaSStrassenMultiplicationOMP::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();

  if (!valid_data_) {
    return true;
  }

  output.push_back(static_cast<double>(n_));

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      output.push_back(c_(i, j));
    }
  }

  return true;
}

Matrix MorozovaSStrassenMultiplicationOMP::AddMatrix(const Matrix &a, const Matrix &b) {
  return AddMatrixImpl(a, b);
}

Matrix MorozovaSStrassenMultiplicationOMP::SubtractMatrix(const Matrix &a, const Matrix &b) {
  return SubtractMatrixImpl(a, b);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStandard(const Matrix &a, const Matrix &b) {
  return MultiplyStandardImpl(a, b);
}

void MorozovaSStrassenMultiplicationOMP::SplitMatrix(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21,
                                                     Matrix &m22) {
  SplitMatrixImpl(m, m11, m12, m21, m22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MergeMatrices(const Matrix &m11, const Matrix &m12, const Matrix &m21,
                                                         const Matrix &m22) {
  return MergeMatricesImpl(m11, m12, m21, m22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStrassen(const Matrix &a, const Matrix &b, int leaf_size) {
  return StrassenMultiplyIterative(a, b, leaf_size, kMaxParallelDepth);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStandardParallel(const Matrix &a, const Matrix &b) {
  return MultiplyStandardParallelImpl(a, b);
}

}  // namespace morozova_s_strassen_multiplication
