/**
 * @file methods/ann/layer/group_norm_impl.hpp
 * @author Shikhar Jaiswal
 *
 * Implementation of the Group Normalization class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_GROUPNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GROUPNORM_IMPL_HPP

// In case it is not included.
#include "group_norm.hpp"

namespace mlpack {
namespace ann { /** Artificial Neural Network. */


template<typename InputDataType, typename OutputDataType>
GroupNorm<InputDataType, OutputDataType>::GroupNorm() :
    size(0),
    groupCount(1),
    eps(1e-8),
    loading(false)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
GroupNorm<InputDataType, OutputDataType>::GroupNorm(
    const size_t size, const size_t groupCount, const double eps) :
    size(size),
    groupCount(groupCount),
    eps(eps),
    loading(false)
{
  if (size % groupCount != 0)
  {
    Log::Fatal << "Total input Size must be divisible by groupCount!" << std::endl;
  }

  weights.set_size(size * groupCount * 2 + 1, 1);
}

template<typename InputDataType, typename OutputDataType>
void GroupNorm<InputDataType, OutputDataType>::Reset()
{
  gamma = arma::mat(weights.memptr(), size * groupCount, 1, false, false);
  beta = arma::mat(weights.memptr() + gamma.n_elem, size * groupCount, 1, false, false);

  if (!loading)
  {
    gamma.fill(1.0);
    beta.fill(0.0);
  }

  loading = false;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  arma::mat reshapedInput(const_cast<arma::Mat<eT>&>(input).memptr(),
    size / groupCount, groupCount * input.n_cols, false, false);

  if (output.is_empty())
    output.zeros(size, input.n_cols);

  arma::mat reshapedOutput((output).memptr(),
    size / groupCount, groupCount * output.n_cols, false, false);

  mean = arma::mean(reshapedInput, 0);
  variance = arma::var(reshapedInput, 1, 0);

  // Normalize the input.
  reshapedOutput = reshapedInput.each_row() - mean;
  inputMean = reshapedOutput;
  reshapedOutput.each_row() /= arma::sqrt(variance + eps);

  // Reused in the backward and gradient step.
  normalized = output;

  // Scale and shift the output.
  reshapedOutput.each_col() %= gamma;
  reshapedOutput.each_col() += beta;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  arma::mat inputReshaped(const_cast<arma::Mat<eT>&>(input).memptr(),
    size / groupCount, groupCount * input.n_cols, false, false);
  arma::mat gyReshaped(const_cast<arma::Mat<eT>&>(gy).memptr(),
    size / groupCount, groupCount * gy.n_cols, false, false);
  arma::mat gReshaped((g).memptr(),
    size / groupCount, groupCount * g.n_cols, false, false);

  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);

  // dl / dxhat.
  const arma::mat norm = gyReshaped.each_col() % gamma;

  // sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  const arma::mat var = arma::sum(norm % inputMean, 0) %
      arma::pow(stdInv, 3.0) * -0.5;

  // dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  gReshaped = (norm.each_row() % stdInv) + (inputMean.each_row() %
      var * 2 / inputReshaped.n_rows);

  // sum (dl / dxhat * -1 / stdInv) + variance *
  // (sum -2 * (x - mu)) / m.
  gReshaped.each_row() += arma::sum(norm.each_row() % -stdInv, 0) / inputReshaped.n_rows;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void GroupNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  gradient.set_size(size + size, 1);

  // Step 5: dl / dy * xhat.
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized % error, 1);

  // Step 6: dl / dy.
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(error, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void GroupNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(groupCount));

  if (cereal::is_loading<Archive>())
  {
    weights.set_size(size * groupCount * 2 + 1, 1);
    loading = true;
  }

  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(gamma));
  ar(CEREAL_NVP(beta));
}

} // namespace ann
} // namespace mlpack

#endif
