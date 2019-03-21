/**
 * @file r_squared.hpp
 * @author Gaurav Sharma
 *
 * The R_Squared metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_R_SQUARED_HPP
#define MLPACK_CORE_CV_METRICS_R_SQUARED_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
 * The R-Squared also known as coefficient of determination is a metric of
 * performance for regression algorithms. It is the proportion of the
 * variance in Y being explained by the variation in X, and is given
 * by (1 - SSE/SST), where SSE is sum of square errors and
 * SST is total sum of squares.
 */
class RSquared
{
 public:
  /**
   * Run prediction and calculate the r-squared .
   *
   * @param model A regression model.
   * @param data Column-major data containing test items.
   * @param responses Ground truth (correct) target values for the test items,
   *     should be either a row vector or a column-major matrix.
   */
  template<typename MLAlgorithm, typename DataType, typename ResponsesType>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& responses);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the measurement.
   */
  static const bool NeedsMinimization = false;
};

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "r_squared_impl.hpp"

#endif
