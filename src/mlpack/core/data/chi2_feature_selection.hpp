/**
 * @file core/data/chi2_feature_selection.hpp
 * @author Jeffin Sam
 *
 * Feature selction based on Chi-Square Test.
 * This test thus can be used to determine the best features for a given
 * dataset by determining the features on which the output class label
 * is most dependent on.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CHI2_FEATURE_SELECTION_HPP
#define MLPACK_CORE_DATA_CHI2_FEATURE_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {
namespace fs {

/**
 *
 * @code
 * arma::Mat<double> input = loadData();
 * arma::Mat<double> target = loadData();
 * arma::Mat<double> output;
 * size_t outputSize = 1;
 *
 * // removes all low-variance features.
 * data::fs::Chi2Selection(input, target, output, outputSize);
 * @endcode
 *
 * @param input Input dataset with actual number of features.
 * @param target Ouput labels for the respective Input. 
 * @param output Output matrix with lesser number of features.
 * @param outputSize No of features you want in output matrix.
 */
template<typename T>
void Chi2Selection(const arma::Mat<T>& input,
				   const arma::rowvec target,
                   arma::Mat<T>& output,
                   const size_t outputSize);

} // namespace fs
} // namespace data
} // namespace mlpack

// Include implementation.
#include "chi2_feature_selection_impl.hpp"

#endif
