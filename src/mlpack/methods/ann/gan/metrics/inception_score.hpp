/**
 * @file inception_score.hpp
 * @author Saksham Bansal
 *
 * Definition of Inception Score for Generative Adversarial Networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_METRICS_INCEPTION_SCORE_HPP
#define MLPACK_METHODS_METRICS_INCEPTION_SCORE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {

/**
* Function that computes Inception Score for a set of images produced by a GAN.
*
* For more information, see the following.
*
* @code
* @article{Goodfellow2016,
*   author  = {Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung,
*              Alec Radford, Xi Chen},
*   title   = {Improved Techniques for Training GANs},
*   year    = {2016},
*   url     = {https://arxiv.org/abs/1606.03498},
* }
* @endcode
*
* @param trueOutputs Ground truth sequences.
* @param predOutputs Sequences predicted by model.
* @param tol Minimum absolute difference value
*            which is considered as a model failure.
*/
template<typename ModelType>
double InceptionScore(ModelType Model,
                      arma::mat images);


} // namespace ann
} // namespace mlpack

#include "inception_score_impl.hpp"

#endif
