/**
 * @file core/dists/diagonal_gaussian_distribution.cpp
 * @author Kim SangYeon
 *
 * Implementation of Gaussian distribution class with diagonal covariance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_DIAGONAL_GAUSSIAN_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTRIBUTIONS_DIAGONAL_GAUSSIAN_DISTRIBUTION_IMPL_HPP

#include "diagonal_gaussian_distribution.hpp"
#include <mlpack/methods/gmm/diagonal_constraint.hpp>

namespace mlpack {
namespace distribution {

template<typename MatType, typename Vectype>
DiagonalGaussianDistribution<MatType, Vectype>::DiagonalGaussianDistribution(
    const Vectype& mean,
    const Vectype& covariance) :
    mean(mean)
{
  Covariance(covariance);
}

template<typename MatType, typename Vectype>
void DiagonalGaussianDistribution<MatType, Vectype>::Covariance(const Vectype& covariance)
{
  this->invCov = 1 / covariance;
  this->logDetCov = arma::accu(log(covariance));
  this->covariance = covariance;
}

template<typename MatType, typename Vectype>
void DiagonalGaussianDistribution<MatType, Vectype>::Covariance(Vectype&& covariance)
{
  this->invCov = 1 / covariance;
  this->logDetCov = arma::accu(log(covariance));
  this->covariance = std::move(covariance);
}

template<typename MatType, typename Vectype>
double DiagonalGaussianDistribution<MatType, Vectype>::LogProbability(
    const Vectype& observation) const
{
  const size_t k = observation.n_elem;
  const Vectype diff = observation - mean;
  const Vectype logExponent = diff.t() * arma::diagmat(invCov) * diff;
  return -0.5 * k * log2pi - 0.5 * logDetCov - 0.5 * logExponent(0);
}

template<typename MatType, typename Vectype>
void DiagonalGaussianDistribution<MatType, Vectype>::LogProbability(
    const MatType& observations,
    Vectype& logProbabilities) const
{
  const size_t k = observations.n_rows;

  // Column i of 'diffs' is the difference between observations.col(i) and
  // the mean.
  MatType diffs = observations.each_col() - mean;

  // Calculates log of exponent equation in multivariate Gaussian
  // distribution. We use only diagonal part for faster computation.
  Vectype logExponents = -0.5 * arma::trans(diffs % diffs) * invCov;

  logProbabilities = -0.5 * k * log2pi - 0.5 * logDetCov + logExponents;
}

template<typename MatType, typename Vectype>
Vectype DiagonalGaussianDistribution<MatType, Vectype>::Random() const
{
  return (arma::sqrt(covariance) % arma::randn<Vectype>(mean.n_elem)) + mean;
}

template<typename MatType, typename Vectype>
void DiagonalGaussianDistribution<MatType, Vectype>::Train(const MatType& observations)
{
  if (observations.n_cols > 1)
  {
    covariance.zeros(observations.n_rows);
  }
  else
  {
    mean.zeros(0);
    covariance.zeros(0);
    return;
  }

  // Calculate and normalize the mean.
  mean = arma::sum(observations, 1) / observations.n_cols;

  // Now calculate the covariance.
  const MatType diffs = observations.each_col() - mean;
  covariance += arma::sum(diffs % diffs, 1);

  // Finish estimating the covariance by normalizing, with the (1 / (n - 1))
  // to make the estimator unbiased.
  covariance /= (observations.n_cols - 1);
  invCov = 1 / covariance;
  logDetCov = arma::accu(log(covariance));
}

template<typename MatType, typename Vectype>
void DiagonalGaussianDistribution<MatType, Vectype>::Train(const MatType& observations,
                                                           const Vectype& probabilities)
{
  if (observations.n_cols > 0)
  {
    covariance.zeros(observations.n_rows);
  }
  else
  {
    mean.zeros(0);
    covariance.zeros(0);
    return;
  }

  // We'll normalize the covariance with (v1 - (v2 / v1))
  // for unbiased estimator in the weighted arithmetic mean. The v1 is the sum
  // of the weights, and the v2 is the sum of the each weight squared.
  // If you want to know more detailed description,
  // please refer to https://en.wikipedia.org/wiki/Weighted_arithmetic_mean.
  double v1 = arma::accu(probabilities);

  // If their sum is 0, there is nothing in this Gaussian.
  // At least, set the covariance so that it's invertible.
  if (v1 == 0)
  {
    invCov = 1 / (covariance += 1e-50);
    logDetCov = arma::accu(log(covariance));
    return;
  }

  // Normalize the probabilities.
  Vectype normalizedProbs = probabilities / v1;

  // Calculate the mean.
  mean = observations * normalizedProbs;

  // Now calculate the covariance.
  const MatType diffs = observations.each_col() - mean;
  covariance += (diffs % diffs) * normalizedProbs;

  // Calculate the sum of each weight squared.
  const double v2 = arma::accu(normalizedProbs % normalizedProbs);

  // Finish estimating the covariance by normalizing, with
  // the (1 / (v1 - (v2 / v1))) to make the estimator unbiased.
  if (v2 != 1)
    covariance /= (1 - v2);

  invCov = 1 / covariance;
  logDetCov = arma::accu(log(covariance));
}

} // namespace distribution
} // namespace mlpack

#endif
