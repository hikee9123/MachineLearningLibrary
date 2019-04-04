/**
 * @file feature_selection.hpp
 * @author Jeffin Sam
 *
 * Feature selction based on variance thresholding.
 * Motivated by the idea that low variance features contain less
 * information.
 * Calculate varience of each feature, then drop features 
 * with variance below some threshold
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_FEATURE_SELECTION_HPP
#define MLPACK_CORE_DATA_FEATURE_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 *
 * Feature selector that removes all low-variance features.
 * The idea is when a feature doesn’t vary much within itself,
 * it generally has very little predictive power.
 * Variance Threshold doesn’t consider the relationship of
 * features with the target variable.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{ICETCS2009,
 *   title  = {A VarianceMean Based Feature Selection in Text Classification},
 *   author = {Shen Yin, Zongli Jiang},
 *   year   = {2009}
 * }
 * @endcode
 *
 * @param input Input dataset with actual number of features.
 * @param threshold Threshold for variance.
 * @param output Output matrix with lesser number of features.
 */
template<typename T>
void VarianceSelection(const arma::Mat<T>& input,
              const double threshold,
              arma::Mat<T>& output);

} // namespace data
} // namespace mlpack

// Include implementation.
#include "feature_selection_impl.hpp"

#endif
