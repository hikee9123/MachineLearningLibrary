#ifndef MDL_PENALTY_DT_IMPL_HPP
#define MDL_PENALTY_DT_IMPL_HPP

#include "mdl_penalty.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
double MDLPenalty<FitnessFunction>::operator()(
    const arma::vec& childCounts,
    const arma::vec& childGains,
    const double delta,
    const size_t numClasses,
    const double numChildren,
    const double sumWeights,
    const double epsilon) const
{
  // Calculate the original gain without penalty.
  const double gain = fitnessFunction(childCounts, childGains, delta,
      numClasses, numChildren, sumWeights, epsilon);

  // Calculate the penalty term using the MDL formula.
  const double penalty = std::log2(numChildren / sumWeights);

  // Return the penalized gain.
  return gain - penalty;
}

} // namespace tree
} // namespace mlpack

#endif // MDL_PENALTY_DT_IMPL_HPP
