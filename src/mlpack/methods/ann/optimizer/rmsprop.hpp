/**
 * @file rmsprop.hpp
 * @author Marcus Edel
 *
 * Implementation of the RmsProp optimizer. RmsProp is an optimizer that
 * utilizes the magnitude of recent gradients to normalize the gradients.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_RMSPROP_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_RMSPROP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * RmsProp is an optimizer that utilizes the magnitude of recent gradients to
 * normalize the gradients. In its basic form, given a step rate \f$ \gamma \f$
 * and a decay term \f$ \alpha \f$ we perform the following updates:
 *
 * \f{eqnarray*}{
 * r_t &=& (1 - \gamma) f'(\Delta_t)^2 + \gamma r_{t - 1} \\
 * v_{t + 1} &=& \frac{\alpha}{\sqrt{r_t}}f'(\Delta_t) \\
 * \Delta_{t + 1} &=& \Delta_t - v_{t + 1}
 * \f}
 *
 * For more information, see the following.
 *
 * @code
 * @misc{tieleman2012,
 *   title={Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine
 *   Learning},
 *   year={2012}
 * }
 * @endcode
 */
template<typename DataType>
class RMSPROP
{
 public:
  /**
   * Construct the RMSPROP optimizer with the given function and parameters.
   *
   * @param function Function to be optimized (minimized).
   * @param lr The learning rate coefficient.
   * @param alpha Constant similar to that used in AdaDelta and Momentum methods.
   * @param eps The eps coefficient to avoid division by zero (numerical
   *        stability).
   */
  RMSPROP(const double lr = 0.01,
          const double alpha = 0.99,
          const double eps = 1e-8) :
      lr(lr),
      alpha(alpha),
      eps(eps)
  {
    // Nothing to do here.
  }

  /**
   * Optimize the given function using RmsProp.
   */
  void Optimize(DataType& weights) 
  {
    if (meanSquaredGad.n_elem == 0)
    {
      meanSquaredGad = weights; 
      meanSquaredGad.zeros();
    }

    Optimize(weights, gradient, meanSquaredGad);
  }

  /*
   * Sum up all gradients and store the results in the gradients storage.
   */
  void Update(DataType const& function_gradients)
  {
    if (gradient.n_elem != 0)
    {
      DataType outputGradient = function_gradients; // function.Gradient();
      gradient += outputGradient;
    }
    else
    {
      gradient = function_gradients;
    }
  }

  /*
   * Reset the gradient storage.
   */
  void Reset()
  {
    gradient.zeros();
  }

  //! Get the gradient.
  DataType& Gradient() const { return gradient; }
  //! Modify the gradient.
  DataType& Gradient() { return gradient; }

 private:
  /**
   * Optimize the given function using RmsProp.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param meanSquaredGradient The moving average over the root mean squared
   *    gradient used to update the weights.
   */
  template<typename eT>
  void Optimize(arma::Cube<eT>& weights,
                arma::Cube<eT>& gradient,
                arma::Cube<eT>& meanSquaredGradient)
  {
    for (size_t s = 0; s < weights.n_slices; s++)
      Optimize(weights.slice(s), gradient.slice(s), meanSquaredGradient.slice(s));
  }

  /**
   * Optimize the given function using RmsProp.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param meanSquaredGradient The moving average over the root mean squared
   *    gradient used to update the weights.
   */
  template<typename eT>
  void Optimize(arma::Mat<eT>& weights,
                arma::Mat<eT>& gradient,
                arma::Mat<eT>& meanSquaredGradient)
  {
    meanSquaredGradient *= alpha;
    meanSquaredGradient += (1 - alpha) * (gradient % gradient);
    weights -= lr * gradient / (arma::sqrt(meanSquaredGradient) + eps);
  }


  //! The value used as learning rate.
  const double lr;

  //! The value used as alpha
  const double alpha;

  //! The value used as eps.
  const double eps;

  //! The current mean squared error of the gradients.
  DataType meanSquaredGad;

  //! The current gradient.
  DataType gradient;
}; // class RMSPROP

} // namespace ann
} // namespace mlpack

#endif
