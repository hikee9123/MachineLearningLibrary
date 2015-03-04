/**
 * @file identity_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the identity function.
 */
#ifndef __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_IDENTITY_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The identity function, defined by
 *
 * @f[
 * f(x) &=& x \\
 * f'(x) &=& 1
 * @f]
 */
class IdentityFunction
{
 public:
  /**
   * Computes the identity function.
   *
   * @param x Input data.
   * @return f(x).
   */
  static double fn(const double x)
  {
    return x;
  }

  /**
   * Computes the identity function.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename InputVecType, typename OutputVecType>
  static void fn(const InputVecType& x, OutputVecType& y)
  {
    y = x;
  }

  /**
   * Computes the first derivative of the identity function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  static double deriv(const double /* unused */)
  {
    return 1.0;
  }

  /**
   * Computes the first derivatives of the identity function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */
  template<typename InputVecType, typename OutputVecType>
  static void deriv(const InputVecType& y, OutputVecType& x)
  {
    //x.ones(y.n_elem);
    x.ones(y.n_rows, y.n_cols);
  }
}; // class IdentityFunction

}; // namespace ann
}; // namespace mlpack

#endif
