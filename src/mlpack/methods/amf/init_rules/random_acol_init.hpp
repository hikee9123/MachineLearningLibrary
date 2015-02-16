/**
 * @file random_acol_init.hpp
 * @author Mohan Rajendran
 *
 * Intialization rule for Alternating Matrix Factorization. 
 */
#ifndef __MLPACK_METHODS_LMF_RANDOM_ACOL_INIT_HPP
#define __MLPACK_METHODS_LMF_RANDOM_ACOL_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class initializes the W matrix of the AMF algorithm by averaging p
 * randomly chosen columns of V.  In this case, p is a template parameter.  H is
 * then set randomly This simple initialization is performed by the random 
 * Acol initialization introduced in the paper 'Algorithms, Initializations and 
 * Convergence' by Langville et al.
 *
 * @tparam The number of random columns to average for each column of W.
 */
template<int p = 5>
class RandomAcolInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomAcolInitialization()
  { }

  template<typename MatType>
  inline static void Initialize(const MatType& V,
                                const size_t r,
                                arma::mat& W,
                                arma::mat& H)
  {
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    if (p > m)
    {
      Log::Warn << "Number of random columns is more than the number of columns"
          << "available in the V matrix; weird results may ensue!" << std::endl;
    }

    W.zeros(n, r);

    // Initialize W matrix with random columns.
    for (size_t col = 0; col < r; col++)
    {
      for (size_t randCol = 0; randCol < p; randCol++)
      {
        // .col() does not work in this case, as of Armadillo 3.920.
        W.unsafe_col(col) += V.col(math::RandInt(0, m));
      }
    }

    // Now divide by p.
    W /= p;

    // Initialize H to random values.
    H.randu(r, m);
  }
}; // Class RandomAcolInitialization

}; // namespace amf
}; // namespace mlpack

#endif
