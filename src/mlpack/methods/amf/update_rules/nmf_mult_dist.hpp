/**
 * @file nmf_mult_dist.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization.
 */
#ifndef __MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIST_UPDATE_RULES_HPP
#define __MLPACK_METHODS_LMF_UPDATE_RULES_NMF_MULT_DIST_UPDATE_RULES_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * The multiplicative distance update rules for matrices W and H. This follows 
 * a method described in the paper 'Algorithms for Non-negative Matrix Factorization'
 * by D. D. Lee and H. S. Seung. This is a multiplicative rule that ensures
 * that the Frobenius norm \f$ \sqrt{\sum_i \sum_j(V-WH)^2} \f$ is
 * non-increasing between subsequent iterations. Both of the update rules
 * for W and H are defined in this file.
 */
class NMFMultiplicativeDistanceUpdate
{
 public:
  // Empty constructor required for the UpdateRule template. 
  NMFMultiplicativeDistanceUpdate() { }

  template<typename MatType>
  void Initialize(const MatType& dataset, const size_t rank)
  {
        (void)dataset;
        (void)rank;
  }

  /**
   * The update rule for the basis matrix W. The formula used is
   * \f[
   * W_{ia} \leftarrow W_{ia} \frac{(VH^T)_{ia}}{(WHH^T)_{ia}}
   * \f]
   * The function takes in all the matrices and only changes the
   * value of the W matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  template<typename MatType>
  inline static void WUpdate(const MatType& V,
                             arma::mat& W,
                             const arma::mat& H)
  {
    W = (W % (V * H.t())) / (W * H * H.t());
  }

  /**
   * The update rule for the encoding matrix H. The formula used is
   * \f[
   * H_{a\mu} \leftarrow H_{a\mu} \frac{(W^T V)_{a\mu}}{(W^T WH)_{a\mu}}
   * \f]
   * The function takes in all the matrices and only changes the
   * value of the H matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to be updated.
   */
  template<typename MatType>
  inline static void HUpdate(const MatType& V,
                             const arma::mat& W,
                             arma::mat& H)
  {
    H = (H % (W.t() * V)) / (W.t() * W * H);
  }
};

}; // namespace amf
}; // namespace mlpack

#endif
