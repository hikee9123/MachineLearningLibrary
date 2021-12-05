/**
 * @file core/cv/metrics/roc_auc_score.hpp
 * @author Sri Madhan M
 *
 * The area under Receiver Operating Characteristic curve (ROC-AUC) score.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ROCAUCSCORE_HPP
#define MLPACK_CORE_CV_METRICS_ROCAUCSCORE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
 * ROC-AUC is a metric of performance for classification algorithms that for
 * binary classification is equal to area under the curve formed by
 * @f$ (fpr, tpr) @f$, where @f$ fpr @f$ and @f$ tpr @f$ are the true positive
 * rate and false positive rate, which is calculated for many different
 * thresholds. For each thresholds, @f$ tpr @f$ and @f$ fpr @f$ are calculated
 * as, @f$ tpr = tp / (tp + fn) @f$ and @f$ fpr = fp / (fp + tn) @f$,
 * where @f$ tp @f$, @f$ tn @f$, @f$ fp @f$ and @f$ fn @f$ are the numbers of
 * true positives, true negatives, false positives and false negatives
 * respectively.
 *
 * @tparam PositiveClass Positives are assumed to have labels equal to this
 *     value. Defaults to 1.
 */
template<size_t PositiveClass = 1>
class ROCAUCScore
{
 public:
  /**
   * Run classification and calculate area under the ROC curve.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm, typename DataType>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const arma::Row<size_t>& labels);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the metric.
   */
  static const bool NeedsMinimization = false;
};

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "roc_auc_score_impl.hpp"

#endif
