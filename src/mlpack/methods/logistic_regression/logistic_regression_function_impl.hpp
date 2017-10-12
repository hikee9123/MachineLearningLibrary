/**
 * @file logistic_regression_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegressionFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression_function.hpp"

namespace mlpack {
namespace regression {

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const double lambda) :
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  initialPoint = arma::rowvec(predictors.n_rows + 1, arma::fill::zeros);

  // Sanity check.
  if (responses.n_elem != predictors.n_cols)
    Log::Fatal << "LogisticRegressionFunction::LogisticRegressionFunction(): "
        << "predictors matrix has " << predictors.n_cols << " points, but "
        << "responses vector has " << responses.n_elem << " elements (should be"
        << " " << predictors.n_cols << ")!" << std::endl;
}

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const arma::vec& initialPoint,
    const double lambda) :
    initialPoint(initialPoint),
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  // To check if initialPoint is compatible with predictors.
  if (initialPoint.n_rows != (predictors.n_rows + 1) ||
      initialPoint.n_cols != 1)
    this->initialPoint = arma::rowvec(predictors.n_rows + 1, arma::fill::zeros);
}
/**
* Shuffle the order of points. This may be called by the optimizer.
* @param parameters Vector of logistic regression parameters.
*/
/*arma::mat LogisticRegressionFunction<MatType>::Shuffle(const arma::mat& parameters)
{
  return arma::shuffle(parameters);
}*/

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters.
 */
template<typename MatType>
double LogisticRegressionFunction<MatType>::Evaluate(
    const arma::mat& parameters) const
{
  // The objective function is the log-likelihood function (w is the parameters
  // vector for the model; y is the responses; x is the predictors; sig() is the
  // sigmoid function):
  //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
  // We want to minimize this function.  L2-regularization is just lambda
  // multiplied by the squared l2-norm of the parameters then divided by two.

  // For the regularization, we ignore the first term, which is the intercept
  // term and take every term except the last one in the decision variable.
  const double regularization = 0.5 * lambda *
      arma::dot(parameters.tail_cols(parameters.n_elem - 1),
      parameters.tail_cols(parameters.n_elem - 1));

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const arma::rowvec exponents = parameters(0, 0) +
    parameters.tail_cols(parameters.n_elem - 1) * predictors;
  const arma::rowvec sigmoid = 1.0 / (1.0 + arma::exp(-exponents));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.
  double result = 0.0;
  for (size_t i = 0; i < responses.n_elem; ++i)
  {
    if (responses[i] == 1)
      result += log(sigmoid[i]);
    else
      result += log(1.0 - sigmoid[i]);
  }

  // Invert the result, because it's a minimization.
  return -result + regularization;
}

/**
 * Evaluate the logistic regression objective function, but with only one point.
 * This is useful for optimizers that use a separable objective function, such
 * as SGD.
 */
/*
template<typename MatType>
double LogisticRegressionFunction<MatType>::Evaluate(
    const arma::mat& parameters,
    const size_t i) const
{
  // Calculate the regularization term.  We must divide by the number of points,
  // so that sum(Evaluate(parameters, [1:points])) == Evaluate(parameters).
  double norm = arma::norm(parameters.tail_cols(parameters.n_elem - 1));

  const double regularization = lambda * (1.0 / (2.0 * predictors.n_cols)) *
      norm * norm;

  // Calculate sigmoid.
  const double exponent = parameters(0, 0) + arma::dot(predictors.col(i),
      parameters.tail_cols(parameters.n_elem - 1).t());
  const double sigmoid = 1.0 / (1.0 + std::exp(-exponent));

  if (responses[i] == 1)
    return -log(sigmoid) + regularization;
  else
    return -log(1.0 - sigmoid) + regularization;
}
*/
/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters for a given batch from a given point.
 */
template<typename MatType>
double LogisticRegressionFunction<MatType>::Evaluate(
                  const arma::mat& parameters,
                  const size_t begin,
                  const size_t batchSize) const
{
  //parameters = Shuffle(parameters);
  // Calculating the regularization term.
  const double regularization = 0.5 * lambda *
      arma::dot(parameters.tail_cols(parameters.n_elem - 1),
      parameters.tail_cols(parameters.n_elem - 1));

  // Calculating the hypothesis that has to be passed to the sigmoid function.
  /*const arma::mat exponents = parameters(0, 0) + arma::dot(
      predictors.cols(begin, begin + batchSize)*
      parameters.tail_cols(parameters.n_elem - 1).t());*/
  const arma::rowvec exponents = parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1);
  // Calculating the sigmoid function values.
  const arma::mat sigmoid =1.0 / (1.0 + arma::exp(-exponents));

  // Iterating for the given batch size from a given point
  double result = 0.0;
  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    if (responses[i] == 1)
      result += log(sigmoid[i]);
    else
      result += log(1.0 - sigmoid[i]);
  }

  // Invert the result, because it's a minimization.
  return -result + regularization;
}

//! Evaluate the gradient of the logistic regression objective function.
template<typename MatType>
void LogisticRegressionFunction<MatType>::Gradient(
    const arma::mat& parameters,
    arma::mat& gradient) const
{
  // Regularization term.
  arma::mat regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  const arma::rowvec sigmoids = (1 / (1 + arma::exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(arma::size(parameters));
  gradient[0] = -arma::accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
    predictors.t() + regularization;
}

/**
 * Evaluate the individual gradients of the logistic regression objective
 * function with respect to individual points.  This is useful for optimizers
 * that use a separable objective function, such as SGD.
 */
/*template <typename MatType>
template <typename GradType>
void LogisticRegressionFunction<MatType>::Gradient(
    const arma::mat& parameters,
    const size_t i,
    GradType& gradient) const
{
  // Calculate the regularization term.
  GradType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1)
      / predictors.n_cols;

  const double sigmoid = 1.0 / (1.0 + std::exp(-parameters(0, 0)
      - arma::dot(
        predictors.col(i), parameters.tail_cols(parameters.n_elem - 1).t())));

  gradient.set_size(arma::size(parameters));
  gradient[0] = -(responses[i] - sigmoid);
  gradient.tail_cols(parameters.n_elem - 1) = -predictors.col(i).t()
      * (responses[i] - sigmoid) + regularization;
}
*/
//! Evaluate the gradient of the logistic regression
//  objective function for a given batch size.
template<typename MatType>
void LogisticRegressionFunction<MatType>::Gradient(
                const arma::mat& parameters,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize) const
{
  // Regularization term.
  arma::mat regularization;
  regularization = lambda * parameters.col(0).subvec(begin, begin + batchSize);

  const arma::rowvec sigmoids = (1 / (1 + arma::exp(-parameters(0, 0)
      - parameters.col(0).subvec(begin, begin + batchSize).t() * predictors)));

  gradient.set_size(parameters.n_elem);
  gradient[0] = -arma::accu(responses - sigmoids);
  gradient.col(0).subvec(begin, batchSize - 1) = -predictors.cols(begin,
      begin + batchSize) *(responses -
      sigmoids).t() + regularization;
}

/**
 * Evaluate the partial gradient of the logistic regression objective
 * function with respect to the individual features in the parameter.
 */
template <typename MatType>
void LogisticRegressionFunction<MatType>::PartialGradient(
    const arma::mat& parameters,
    const size_t j,
    arma::sp_mat& gradient) const
{
  const arma::rowvec diffs = responses - (1 / (1 + arma::exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(arma::size(parameters));

  if (j == 0)
  {
    gradient[j] = -arma::accu(diffs);
  }
  else
  {
    gradient[j] = arma::dot(-predictors.row(j - 1), diffs) + lambda *
      parameters(0, j);
  }
}

} // namespace regression
} // namespace mlpack

#endif
