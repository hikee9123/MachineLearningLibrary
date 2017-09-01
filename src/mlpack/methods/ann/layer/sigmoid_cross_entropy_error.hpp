/**
 * @file sigmoid_cross_entropy_error.hpp
 * @author Kris Singh
 *
 * Definition of the cross-entropy with logit performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SIGMOID_CROSS_ENTROPY_ERROR_HPP
#define MLPACK_METHODS_ANN_LAYER_SIGMOID_CROSS_ENTROPY_ERROR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The SigmoidCrossEntropyError performance function measures the network's
 * performance according to the cross-entropy function.
 * between the input and target distributions.
 * This function calculates the cross entropy
 * given the real values instead of providing the sigmoid activations.
 * The functions is much more numerically stable as can be found from the 
 * formula below.
 * max(x, 0) - x * z + log(1 + exp(-abs(x)))
 * For more detail look here goo.gl/tRjS6j
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class SigmoidCrossEntropyError
{
 public:
  /**
   * Create the SigmoidCrossEntropyError object.
   */
  SigmoidCrossEntropyError();

  /*
   * Computes the Sigmoid CrossEntropy Error functions.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  inline double Forward(const arma::Mat<eT>&& input, const arma::Mat<eT>&& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename eT>
  inline void Backward(const arma::Mat<eT>&& input,
                const arma::Mat<eT>&& target,
                arma::Mat<eT>&& output);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class SigmoidCrossEntropy

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "sigmoid_cross_entropy_error_impl.hpp"

#endif
