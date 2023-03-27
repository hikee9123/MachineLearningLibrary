/**
 * @file methods/ann/layer/hard_tanh.hpp
 * @author Dhawal Arora
 *
 * Definition and implementation of the HardTanH layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_TANH_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>

namespace mlpack {

/**
 * The Hard Tanh activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \left\{
 *   \begin{array}{lr}
 *     max & : x > maxValue \\
 *     min & : x \le minValue \\
 *     x   & : otherwise
 *   \end{array}
 * \right. \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     0 & : x > maxValue \\
 *     0 & : x \le minValue \\
 *     1 & : otherwise
 *   \end{array}
 * \right.
 * @f}
 *
 * @tparam MatType Matrix representation to accept as input
 * (Default: arma::mat).
 */
template <typename MatType =arma::mat>
class HardTanHType : public Layer<MatType>
{
 public:
  /**
   * Create the HardTanH object using the specified parameters. The range
   * of the linear region can be adjusted by specifying the maxValue and
   * minValue. Default (maxValue = 1, minValue = -1).
   *
   * @param maxValue Range of the linear region maximum value.
   * @param minValue Range of the linear region minimum value.
   */
  HardTanHType(const double maxValue = 1, const double minValue = -1);

  //! Clone the HardTanHType object. This handles polymorphism correctly.
  HardTanHType* Clone() const { return new HardTanHType(*this); }

  virtual ~HardTanHType() {}
  //! Copy the given HardTanHType.
  HardTanHType(const HardTanHType& other);
  //! Take ownership of the given HardTanHType.
  HardTanHType(HardTanHType&& other);
  //! Copy the given HardTanHType.
  HardTanHType& operator=(const HardTanHType& other);
  //! Take ownership of the given HardTanHType.
  HardTanHType& operator=(HardTanHType&& other);

  
  
  
  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input, const MatType& gy, MatType& g);

  //! Get the maximum value.
  double const& MaxValue() const { return maxValue; }
  //! Modify the maximum value.
  double& MaxValue() { return maxValue; }

  //! Get the minimum value.
  double const& MinValue() const { return minValue; }
  //! Modify the minimum value.
  double& MinValue() { return minValue; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Maximum value for the HardTanH function.
  double maxValue;

  //! Minimum value for the HardTanH function.
  double minValue;
}; // class HardTanHType

// Convenience typedefs.

// Standard HardTanH layer.
typedef HardTanHType<arma::mat> HardTanH;

} // namespace mlpack

// Include implementation.
#include "hard_tanh_impl.hpp"

#endif
