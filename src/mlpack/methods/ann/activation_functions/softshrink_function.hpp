/**
 * @file softshrink_function.hpp
 * @author Lakshya Ojha
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFT_SHRINK_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SOFT_SHRINK_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * SoftShrinkage operator is defined as,
 *         | x - lambda, if x >  lambda
 *  f(x) = | x + lambda, if x < -lambda
 *         | 0, otherwise
 *
 * lambda is set to 0.5 by default.
 *
 *          | 1 , if x >  lambda
 *  f'(x) = | 1 , if x < -lambda
 *          | 0 , otherwise
 */

class SoftShrinkFunction
{
 public:
  /**
   * Computes the Soft Shrinkage function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double Fn(const double x, const double lambda = 0.5)
  {
    if (x > lambda)
    {
      return x - lambda;
    }
    else if (x < -1 * lambda)
    {
      return x + lambda;
    }
    return 0.;
  }
  
  /**
   * Computes the Soft Shrinkage function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  
  template <typename InputVecType, typename OutputVecType>
  static void Fn(const InputVecType &x, OutputVecType &y,
                  const double lambda = 0.5)
  {
    y = x;
    y.transform([](const double x, const double lambda = 0.5){
                  return Fn(x, lambda);});
  }

  /**
   * Computes the first derivative of the Soft Shrinkage function.
   *
   * @param y Input data.
   * @return f'(x)
   */
  
  static double Deriv(const double y, const double lambda = 0.5)
  {
      if (y > 0 || y < 0)
      return 1;

      return 0;
  }
  
  /**
   * Computes the first derivatives of the Soft Shrinkage function.
   * 
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  
  template <typename InputVecType, typename OutputVecType>
  static void Deriv(const InputVecType &y, OutputVecType &x,
                    const double lambda = 0.5)
  {
    x = y ;
    x.transform([](const double cy, const double lambda = 0.5){
                return Deriv(cy,lambda);});
  }

};  // class SoftShrinkFunction
}  // namespace ann
}  // namespace mlpack

#endif
