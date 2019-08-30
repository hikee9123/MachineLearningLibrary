/**
 * @file stochastic_method.hpp
 * @author Marcus Edel
 *
 * Implementation of the stochastic method for use in the Principal Components
 * Analysis method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_STOCHASTIC_METHOD_HPP
#define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_STOCHASTIC_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

namespace mlpack {
namespace pca {

/**
 * Implementation of the Stochastic Policy. The stochastic approximation
 * algorithm is an iterative algorithm, where in each iteration a single
 * sampled point is used to perform an update as done in Stochastic Gradient
 * Descent. For PCA this means iteratively using vectors sampled from a
 * distribution to update the subspace.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Arora2012,
 *   author    = {Arora, Raman and Cotter, Andrew and Livescu, Karen
 *                and Srebro, Nathan},
 *   title     = {Stochastic optimization for PCA and PLS.},
 *   booktitle = {Allerton Conference},
 *   pages     = {861-868},
 *   year      = {2012},
 * }
 * @endcode
 */
template<typename OptimizerType>
class StochasticPolicyType
{
 public:
  /**
   * Use the stochastic gradient method to perform the principal components
   * analysis (PCA).
   *
   * @param optimizer Instantiated optimizer used to update the subspace.
   */
  StochasticPolicyType(const OptimizerType& optimizer = OptimizerType()) :
      optimizer(optimizer)
  {
    /* Nothing to do here. */
  }

  /**
   * Apply Principal Component Analysis to the provided data set using the
   * exact SVD method.
   *
   * @param data Data matrix.
   * @param centeredData Centered data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigVal Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   * @param rank Rank of the decomposition.
   */
  void Apply(const arma::mat& data,
             const arma::mat& centeredData,
             arma::mat& transformedData,
             arma::vec& eigVal,
             arma::mat& eigvec,
             const size_t rank)
  {
    PCAFunction f(centeredData);

    eigvec = arma::randn(centeredData.n_rows, rank);
    optimizer.Optimize(f, eigvec);

    // Project the samples to the principals.
    transformedData = arma::trans(eigvec) * centeredData;

    // Approximate eigenvalues and eigenvectors using Rayleigh–Ritz method.
    arma::mat u, v;
    arma::svd_econ(u, eigVal, v, transformedData);

    // Now we must square the singular values to get the eigenvalues.
    // In addition we must divide by the number of points, because the
    // covariance matrix is X * X' / (N - 1).
    eigVal %= eigVal / (data.n_cols - 1);
  }

  //! Get the optimizer.
  OptimizerType Optimizer() const { return optimizer; }
  //! Modify the optimizer.
  OptimizerType& Optimizer() { return optimizer; }

 private:
  //! Locally stored optimizer instance.
  OptimizerType optimizer;

  /**
   * The principal component analysis (PCA) function for the
   * Principal component analysis (PCA). This is used by various mlpack
   * optimizers to be used for principal component analysis.
   */
  class PCAFunction
  {
   public:
    /**
     * Construct the principal component analysis (PCA) function with the
     * given data.
     *
     * @param data Data matrix.
     */
    PCAFunction(const arma::mat& data) :
        // We promise to be well-behaved... the elements won't be modified.
        data(math::MakeAlias(const_cast<arma::mat&>(data), false))
    {
      // Nothing to do here.
    }

    /**
     * Shuffle the order of function visitation. This may be called by the
     * optimizer.
     */
    void Shuffle()
    {
      // Generate ordering.
      arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
          data.n_cols - 1, data.n_cols));
      arma::mat newData = data.cols(ordering);

      // If we are an alias, make sure we don't write to the original data.
      math::ClearAlias(data);

      // Take ownership of the new data.
      data = std::move(newData);
    }

    //! Return the number of functions.
    size_t NumFunctions() const { return data.n_cols; }

    /*
     * Evaluate the PCA function for a particular batch-size.
     *
     * @param coordinates The function coordinates.
     * @param begin The first function.
     * @param batchSize Number of points to process.
     */
    double Evaluate(const arma::mat& coordinates,
                    const size_t begin,
                    const size_t batchSize)
    {
      return arma::trace(coordinates * data.cols(begin, begin + batchSize - 1) *
          data.cols(begin, begin + batchSize - 1).t() * coordinates);
    }

    /*
     * Evaluate the gradient of the PCA function for a particular batch-size.
     *
     * @param coordinates The function coordinates.
     * @param begin The first function.
     * @param gradient The function gradient.
     * @param batchSize Number of points to process.
     */
    void Gradient(const arma::mat& coordinates,
                  const size_t begin,
                  arma::mat& gradient,
                  const size_t batchSize)
    {
      gradient = data.cols(begin, begin + batchSize - 1) *
          data.cols(begin, begin + batchSize - 1).t() * coordinates;
    }

    /*
     * Evaluate the objective function and gradient of the PCA function for a
     * particular batch-size.
     *
     * @param coordinates The function coordinates.
     * @param begin The first function.
     * @param gradient The function gradient.
     * @param batchSize Number of points to process.
     */
    double EvaluateWithGradient(const arma::mat& coordinates,
                                const size_t begin,
                                arma::mat& gradient,
                                const size_t batchSize)
    {
      gradient = data.cols(begin, begin + batchSize - 1) *
          data.cols(begin, begin + batchSize - 1).t() * coordinates;

      return arma::trace(coordinates * gradient);
    }

   private:
    //! Data matrix..
    arma::mat data;
  };
};

/**
 * Implementation of the PCA Update policy which wraps the existing optimizer
 * update policies.
 */
template<typename UpdatePolicyType = ens::VanillaUpdate>
class PCAUpdate
{
 public:
  /**
   * Construct the wrapper update policy with the given update policy.
   *
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
   *        estimates.
   * @param epsilon Value used to initialise the mean squared gradient
   *        parameter.
   */
  PCAUpdate(const UpdatePolicyType& updatePolicy = UpdatePolicyType()) :
      updatePolicy(updatePolicy)
  {
    /* Nothing to do here. */
  }

  /**
   * Adam specific constructor.
   *
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
   *        estimates.
   * @param epsilon Value used to initialise the mean squared gradient
   *        parameter.
   */
  PCAUpdate(const double epsilon, const double beta1, const double beta2) :
      updatePolicy(UpdatePolicyType(epsilon, beta1, beta2))
  {
    /* Nothing to do here. */
  }

  void Initialize(const size_t rows, const size_t cols)
  {
    updatePolicy.Initialize(rows, cols);
  }

  /**
   * General update step, all parameters are forwarded to the actual
   * update policy.
   *
   * @param iterate Parameters that minimize the function.
   */
  template<typename... Targs>
  void Update(arma::mat& iterate, Targs... fArgs)
  {
    iterate *= -1;
    updatePolicy.Update(iterate, fArgs...);

    arma::mat R;
    arma::qr_econ(iterate, R, iterate);
  }

 private:
  //! The update policy used to update the parameters in each iteration.
  UpdatePolicyType updatePolicy;
};

// Convenience typedefs.

using StochasticSGDPolicy = StochasticPolicyType<
    ens::SGD<PCAUpdate<ens::VanillaUpdate> > >;

using StochasticAdamPolicy = StochasticPolicyType<
    ens::AdamType<PCAUpdate<ens::AdamUpdate> > >;

using StochasticSVRGPolicy = StochasticPolicyType<ens::SVRGType<
    PCAUpdate<ens::SVRGUpdate>, ens::NoDecay> >;

} // namespace pca
} // namespace mlpack

#endif
