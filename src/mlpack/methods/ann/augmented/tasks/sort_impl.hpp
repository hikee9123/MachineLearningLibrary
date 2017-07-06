/**
 * @file sort_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of SortTask class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AUGMENTED_TASKS_SORT_IMPL_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_SORT_IMPL_HPP

#include "sort.hpp"

#include <cassert>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

SortTask::SortTask(const size_t maxLength, const size_t bitLen,
                   bool addSeparator = false)
  : maxLength(maxLength), bitLen(bitLen), addSeparator(addSeparator) {
  assert(maxLength > 1);
  assert(bitLen > 0);
}

void SortTask::Generate(arma::field<arma::mat>& input,
                        arma::field<arma::mat>& labels,
                        const size_t batchSize,
                        bool fixedLength = false)
{
  input = arma::field<arma::mat>(batchSize);
  labels = arma::field<arma::mat>(batchSize);
  size_t size = maxLength;
  for (size_t i = 0; i < batchSize; ++i) {
    if (!fixedLength) {
      // Random uniform length from [2..maxLength]
      size = RandInt(2, maxLength+1);
    }
    input(i) = arma::randi<arma::mat>(bitLen, size, arma::distr_param(0, 1));
    arma::mat itemAns = arma::mat(bitLen, size);
    arma::colvec vals(size);
    for (size_t j = 0; j < size; ++j) {
      int val = 0;
      for (size_t k = 0; k < bitLen; ++k) {
        val <<= 1;
        val += input(i).at(k, j);
      }
      vals[j] = val;
    }
    arma::uvec indices = arma::sort_index(vals);
    for (size_t j = 0; j < size; ++j) {
      itemAns.col(j) = input(i).col(indices.at(j));
    }
    labels(i) = itemAns;
    input(i).reshape(input(i).n_elem, 1);
    if (addSeparator) {
      arma::mat sepInput = arma::zeros(input(i).n_elem + size, 1);
      size_t ptr = 0, origPtr = 0;
      for (size_t j = 0; j < size; ++j) {
        sepInput.rows(ptr, ptr + bitLen - 1) =
          input(i).rows(origPtr, origPtr + bitLen - 1);
        ptr += bitLen;
        origPtr += bitLen;
        sepInput.at(ptr, 0) = 0.5;
        ++ptr;
      }
      input(i) = sepInput;
    }
    labels(i).reshape(labels(i).n_elem, 1);
  }
}

void SortTask::Generate(arma::mat& input, arma::mat& labels,
                        const size_t batchSize) {
  arma::field<arma::mat> fieldInput, fieldLabels;
  Generate(fieldInput, fieldLabels, batchSize, true);
  size_t input_rows = fieldInput(0).n_rows;
  size_t label_rows = fieldLabels(0).n_rows;
  size_t cols = batchSize;
  input = arma::zeros(input_rows, cols);
  labels = arma::zeros(label_rows, cols);
  for (size_t i = 0; i < cols; ++i) {
    input.col(i) = fieldInput.at(i);
    labels.col(i) = fieldLabels.at(i);
}

} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack
#endif
