/**
 * @file kde_rules_impl.hpp
 * @author Roberto Hueso
 *
 * Implementation of rules for Kernel Density Estimation with generic trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_KDE_RULES_IMPL_HPP
#define MLPACK_METHODS_KDE_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "kde_rules.hpp"

// Used for Monte Carlo estimation.
#include <boost/math/distributions/normal.hpp>

namespace mlpack {
namespace kde {

template<typename MetricType, typename KernelType, typename TreeType>
KDERules<MetricType, KernelType, TreeType>::KDERules(
    const arma::mat& referenceSet,
    const arma::mat& querySet,
    arma::vec& densities,
    const double relError,
    const double absError,
    const double mcProb,
    const size_t initialSampleSize,
    const double mcAccessCoef,
    const double mcBreakCoef,
    MetricType& metric,
    KernelType& kernel,
    const bool monteCarlo,
    const bool sameSet) :
    referenceSet(referenceSet),
    querySet(querySet),
    densities(densities),
    absError(absError),
    relError(relError),
    mcBeta(1 - mcProb),
    initialSampleSize(initialSampleSize),
    mcAccessCoef(mcAccessCoef),
    mcBreakCoef(mcBreakCoef),
    metric(metric),
    kernel(kernel),
    monteCarlo(monteCarlo),
    sameSet(sameSet),
    lastQueryIndex(querySet.n_cols),
    lastReferenceIndex(referenceSet.n_cols),
    baseCases(0),
    scores(0)
{
  // Initialize accumMCAlpha only if Monte Carlo estimations are available.
  if (monteCarlo && kernelIsGaussian)
    accumMCAlpha = arma::vec(querySet.n_cols, arma::fill::zeros);
}

//! The base case.
template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline
double KDERules<MetricType, KernelType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // If reference and query sets are the same we don't want to compute the
  // estimation of a point with itself.
  if (sameSet && (queryIndex == referenceIndex))
    return 0.0;

  // Avoid duplicated calculations.
  if ((lastQueryIndex == queryIndex) && (lastReferenceIndex == referenceIndex))
    return 0.0;

  // Calculations.
  const double distance = metric.Evaluate(querySet.col(queryIndex),
                                          referenceSet.col(referenceIndex));
  densities(queryIndex) += kernel.Evaluate(distance);

  ++baseCases;
  lastQueryIndex = queryIndex;
  lastReferenceIndex = referenceIndex;
  traversalInfo.LastBaseCase() = distance;
  return distance;
}

//! Single-tree scoring function.
template<typename MetricType, typename KernelType, typename TreeType>
inline double KDERules<MetricType, KernelType, TreeType>::
Score(const size_t queryIndex, TreeType& referenceNode)
{
  // Auxiliary variables.
  kde::KDEStat& referenceStat = referenceNode.Stat();
  const arma::vec& queryPoint = querySet.unsafe_col(queryIndex);
  const size_t refNumDesc = referenceNode.NumDescendants();
  double score, minDistance, maxDistance, depthAlpha;
  // Calculations are not duplicated.
  bool alreadyDidRefPoint0 = false;

  // Calculate alpha if Monte Carlo is available.
  if (monteCarlo && kernelIsGaussian)
    depthAlpha = CalculateAlpha(&referenceNode);
  else
    depthAlpha = -1;

  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid &&
      lastQueryIndex == queryIndex &&
      traversalInfo.LastReferenceNode() != NULL &&
      traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0))
  {
    // Don't duplicate calculations.
    alreadyDidRefPoint0 = true;
    const double furthestDescDist = referenceNode.FurthestDescendantDistance();
    minDistance = traversalInfo.LastBaseCase() - furthestDescDist;
    maxDistance = traversalInfo.LastBaseCase() + furthestDescDist;
  }
  else
  {
    // All Calculations are new.
    const math::Range r = referenceNode.RangeDistance(queryPoint);
    minDistance = r.Lo();
    maxDistance = r.Hi();
  }

  const double maxKernel = kernel.Evaluate(minDistance);
  const double minKernel = kernel.Evaluate(maxDistance);
  const double bound = maxKernel - minKernel;

  if (bound <= (absError + relError * minKernel) / referenceSet.n_cols)
  {
    // Estimate values.
    double kernelValue;

    // Calculate kernel value based on reference node centroid.
    if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
      kernelValue = EvaluateKernel(queryIndex, referenceNode.Point(0));
    else
      kernelValue = EvaluateKernel(queryPoint, referenceStat.Centroid());

    if (alreadyDidRefPoint0)
      densities(queryIndex) += (refNumDesc - 1) * kernelValue;
    else
      densities(queryIndex) += refNumDesc * kernelValue;

    // Don't explore this tree branch.
    score = DBL_MAX;

    // Store not used alpha for Monte Carlo.
    if (kernelIsGaussian && monteCarlo)
      accumMCAlpha(queryIndex) += depthAlpha;
  }
  else if (monteCarlo &&
           refNumDesc >= mcAccessCoef * initialSampleSize &&
           kernelIsGaussian)
  {
    // Monte Carlo probabilistic estimation.
    // Calculate z using accumulated alpha if possible.
    const double alpha = depthAlpha + accumMCAlpha(queryIndex);
    const boost::math::normal normalDist;
    const double z =
        std::abs(boost::math::quantile(normalDist, alpha / 2));

    // Auxiliary variables.
    arma::vec sample;
    size_t m = initialSampleSize;
    double meanSample;
    bool useMonteCarloPredictions = true;

    // Resample as long as confidence is not high enough.
    while (m > 0)
    {
      const size_t oldSize = sample.size();
      const size_t newSize = oldSize + m;

      // Don't use probabilistic estimation if this is going to take a similar
      // amount of computations to the exact calculation.
      if (newSize >= mcBreakCoef * refNumDesc)
      {
        useMonteCarloPredictions = false;
        break;
      }

      // Increase the sample size.
      sample.resize(newSize);
      for (size_t i = 0; i < m; ++i)
      {
        // Sample and evaluate random points from the reference node.
        size_t randomPoint;
        if (alreadyDidRefPoint0)
          randomPoint = math::RandInt(1, refNumDesc);
        else
          randomPoint = math::RandInt(0, refNumDesc);

        sample(oldSize + i) =
            EvaluateKernel(queryIndex, referenceNode.Descendant(randomPoint));
      }
      meanSample = arma::mean(sample);
      const double stddev = arma::stddev(sample);
      const double mThreshBase =
          z * stddev * (1 + relError) / (relError * meanSample);
      const size_t mThresh = std::ceil(mThreshBase * mThreshBase);

      if (sample.size() < mThresh)
        m = mThresh - sample.size();
      else
        m = 0;
    }

    if (useMonteCarloPredictions)
    {
      // Confidence is high enough so we can use Monte Carlo estimation.
      if (alreadyDidRefPoint0)
        densities(queryIndex) += (refNumDesc - 1) * meanSample;
      else
        densities(queryIndex) += refNumDesc * meanSample;

      // Prune.
      score = DBL_MAX;

      // Accumulated alpha has been used.
      accumMCAlpha(queryIndex) = 0;
    }
    else
    {
      // Recurse.
      score = minDistance;

      if (referenceNode.IsLeaf())
      {
        // Reclaim not used alpha since the node will be exactly computed.
        accumMCAlpha(queryIndex) += depthAlpha;
      }
    }
  }
  else
  {
    score = minDistance;

    // If node is going to be exactly computed, reclaim not used alpha for
    // Monte Carlo estimations.
    if (kernelIsGaussian && monteCarlo && referenceNode.IsLeaf())
      accumMCAlpha(queryIndex) += depthAlpha;
  }

  ++scores;
  traversalInfo.LastReferenceNode() = &referenceNode;
  traversalInfo.LastScore() = score;
  return score;
}

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
Rescore(const size_t /* queryIndex */,
        TreeType& /* referenceNode */,
        const double oldScore) const
{
  // If it's pruned it continues to be pruned.
  return oldScore;
}

//! Dual-tree scoring function.
template<typename MetricType, typename KernelType, typename TreeType>
inline double KDERules<MetricType, KernelType, TreeType>::
Score(TreeType& queryNode, TreeType& referenceNode)
{
  kde::KDEStat& referenceStat = referenceNode.Stat();
  kde::KDEStat& queryStat = queryNode.Stat();
  const size_t refNumDesc = referenceNode.NumDescendants();
  double score, minDistance, maxDistance, depthAlpha;
  // Calculations are not duplicated.
  bool alreadyDidRefPoint0 = false;

  // Calculate alpha if Monte Carlo is available.
  if (monteCarlo && kernelIsGaussian)
    depthAlpha = CalculateAlpha(&referenceNode);
  else
    depthAlpha = -1;

  // Check if not used Monte Carlo alpha can be reclaimed for this combination
  // of nodes.
  const bool canReclaimAlpha = kernelIsGaussian &&
                               monteCarlo &&
                               referenceNode.IsLeaf() &&
                               queryNode.IsLeaf();

  if (tree::TreeTraits<TreeType>::FirstPointIsCentroid &&
      (traversalInfo.LastQueryNode() != NULL) &&
      (traversalInfo.LastReferenceNode() != NULL) &&
      (traversalInfo.LastQueryNode()->Point(0) == queryNode.Point(0)) &&
      (traversalInfo.LastReferenceNode()->Point(0) == referenceNode.Point(0)))
  {
    // Don't duplicate calculations.
    alreadyDidRefPoint0 = true;
    lastQueryIndex = queryNode.Point(0);
    lastReferenceIndex = referenceNode.Point(0);

    // Calculate min and max distance.
    const double refFurtDescDist = referenceNode.FurthestDescendantDistance();
    const double queryFurtDescDist = queryNode.FurthestDescendantDistance();
    const double sumFurtDescDist = refFurtDescDist + queryFurtDescDist;
    minDistance = traversalInfo.LastBaseCase() - sumFurtDescDist;
    maxDistance = traversalInfo.LastBaseCase() + sumFurtDescDist;
  }
  else
  {
    // All calculations are new.
    const math::Range r = queryNode.RangeDistance(referenceNode);
    minDistance = r.Lo();
    maxDistance = r.Hi();
  }

  const double maxKernel = kernel.Evaluate(minDistance);
  const double minKernel = kernel.Evaluate(maxDistance);
  const double bound = maxKernel - minKernel;

  // If possible, avoid some calculations because of the error tolerance.
  if (bound <= (absError + relError * minKernel) / referenceSet.n_cols)
  {
    // Auxiliary variables.
    double kernelValue;

    // If calculating a center is not required.
    if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    {
      kernelValue = EvaluateKernel(queryNode.Point(0), referenceNode.Point(0));
    }
    // Sadly, we have no choice but to calculate the center.
    else
    {
      kernelValue = EvaluateKernel(queryStat.Centroid(),
                                   referenceStat.Centroid());
    }

    // Sum up estimations.
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
    {
      if (alreadyDidRefPoint0 && i == 0)
        densities(queryNode.Descendant(i)) += (refNumDesc - 1) * kernelValue;
      else
        densities(queryNode.Descendant(i)) += refNumDesc * kernelValue;
    }

    // Prune.
    score = DBL_MAX;

    // Store not used alpha for Monte Carlo.
    if (kernelIsGaussian && monteCarlo)
      queryStat.AccumAlpha() += depthAlpha;
  }
  else if (monteCarlo &&
           refNumDesc >= mcAccessCoef * initialSampleSize &&
           kernelIsGaussian)
  {
    // Monte Carlo probabilistic estimation.
    // Calculate z using accumulated alpha if possible.
    const double alpha = depthAlpha + queryStat.AccumAlpha();
    const boost::math::normal normalDist;
    const double z =
        std::abs(boost::math::quantile(normalDist, alpha / 2));

    // Auxiliary variables.
    arma::vec sample;
    arma::vec means = arma::zeros(queryNode.NumDescendants());
    size_t m;
    double meanSample;
    bool useMonteCarloPredictions = true;

    // Pick a sample for every query node.
    for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
    {
      const size_t queryIndex = queryNode.Descendant(i);
      sample.clear();
      m = initialSampleSize;

      // Resample as long as confidence is not high enough.
      while (m > 0)
      {
        const size_t oldSize = sample.size();
        const size_t newSize = oldSize + m;

        // Don't use probabilistic estimation if this is going to take a similar
        // amount of computations to the exact calculation.
        if (newSize >= mcBreakCoef * refNumDesc)
        {
          useMonteCarloPredictions = false;
          break;
        }

        // Increase the sample size.
        sample.resize(newSize);
        for (size_t i = 0; i < m; ++i)
        {
          // Sample and evaluate random points from the reference node.
          size_t randomPoint;
          if (alreadyDidRefPoint0)
            randomPoint = math::RandInt(1, refNumDesc);
          else
            randomPoint = math::RandInt(0, refNumDesc);

          sample(oldSize + i) =
              EvaluateKernel(queryIndex, referenceNode.Descendant(randomPoint));
        }
        meanSample = arma::mean(sample);
        const double stddev = arma::stddev(sample);
        const double mThreshBase =
            z * stddev * (1 + relError) / (relError * meanSample);
        const size_t mThresh = std::ceil(mThreshBase * mThreshBase);

        if (sample.size() < mThresh)
          m = mThresh - sample.size();
        else
          m = 0;
      }

      // Store mean for the i_th query node descendant point.
      if (useMonteCarloPredictions)
        means(i) = meanSample;
      else
        break;
    }

    if (useMonteCarloPredictions)
    {
      // Confidence is high enough so we can use Monte Carlo estimation.
      for (size_t i = 0; i < queryNode.NumDescendants(); ++i)
      {
        if (alreadyDidRefPoint0 && i == 0)
          densities(queryNode.Descendant(i)) += (refNumDesc - 1) * means(i);
        else
          densities(queryNode.Descendant(i)) += refNumDesc * means(i);
      }

      // Prune.
      score = DBL_MAX;

      // Accumulated alpha has been used.
      queryStat.AccumAlpha() = 0;
    }
    else
    {
      // Recurse.
      score = minDistance;

      if (canReclaimAlpha)
      {
        // Reclaim not used Monte Carlo alpha since the nodes will be
        // exactly computed.
        queryStat.AccumAlpha() += depthAlpha;
      }
    }
  }
  else
  {
    // Recurse.
    score = minDistance;

    // If node is going to be exactly computed, reclaim not used alpha for
    // Monte Carlo estimations.
    if (canReclaimAlpha)
      queryStat.AccumAlpha() += depthAlpha;
  }

  ++scores;
  traversalInfo.LastQueryNode() = &queryNode;
  traversalInfo.LastReferenceNode() = &referenceNode;
  traversalInfo.LastScore() = score;
  return score;
}

//! Dual-tree rescore.
template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
Rescore(TreeType& /*queryNode*/,
        TreeType& /*referenceNode*/,
        const double oldScore) const
{
  // If a branch is pruned then it continues to be pruned.
  return oldScore;
}

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
EvaluateKernel(const size_t queryIndex,
               const size_t referenceIndex) const
{
  return EvaluateKernel(querySet.unsafe_col(queryIndex),
                        referenceSet.unsafe_col(referenceIndex));
}

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
EvaluateKernel(const arma::vec& query, const arma::vec& reference) const
{
  return kernel.Evaluate(metric.Evaluate(query, reference));
}

template<typename MetricType, typename KernelType, typename TreeType>
inline force_inline double KDERules<MetricType, KernelType, TreeType>::
CalculateAlpha(TreeType* node)
{
  KDEStat& stat = node->Stat();

  // If new mcBeta is different from previously computed mcBeta, then alpha for
  // the node is recomputed.
  if (std::abs(stat.MCBeta() - mcBeta) > DBL_EPSILON)
  {
    TreeType* parent = node->Parent();
    if (parent == NULL)
    {
      // If it's the root node then assign mcBeta.
      stat.MCAlpha() = mcBeta;
    }
    else
    {
      // Distribute it's parent alpha between children.
      stat.MCAlpha() = parent->Stat().MCAlpha() / parent->NumChildren();
    }

    // Set beta value for which this alpha is valid.
    stat.MCBeta() = mcBeta;
  }

  return stat.MCAlpha();
}

//! Clean rules base case.
template<typename TreeType>
inline force_inline
double KDECleanRules<TreeType>::BaseCase(const size_t /* queryIndex */,
                                         const size_t /* refIndex */)
{
  return 0;
}

//! Clean rules single-tree score.
template<typename TreeType>
inline force_inline
double KDECleanRules<TreeType>::Score(const size_t /* queryIndex */,
                                      TreeType& referenceNode)
{
  referenceNode.Stat().AccumAlpha() = 0;
  return 0;
}

//! Clean rules double-tree score.
template<typename TreeType>
inline force_inline
double KDECleanRules<TreeType>::Score(TreeType& queryNode,
                                      TreeType& referenceNode)
{
  queryNode.Stat().AccumAlpha() = 0;
  referenceNode.Stat().AccumAlpha() = 0;
  return 0;
}

} // namespace kde
} // namespace mlpack

#endif
