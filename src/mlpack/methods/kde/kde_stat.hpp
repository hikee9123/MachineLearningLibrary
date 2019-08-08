/**
 * @file kde_stat.hpp
 * @author Roberto Hueso
 *
 * Defines TreeStatType for KDE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KDE_STAT_HPP
#define MLPACK_METHODS_KDE_STAT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

namespace mlpack {
namespace kde {

/**
 * Extra data for each node in the tree for the task of kernel density
 * estimation.
 */
class KDEStat
{
 public:
  //! Initialize the statistic.
  KDEStat() :
      validCentroid(false),
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0),
      pointsInTheBase(0),
      minKernelRecon(DBL_MAX),
      maxKernelRecon(DBL_MIN)
  { /* Nothing to do here. */ }

  //! Initialization for a fully initialized node.
  template<typename TreeType>
  KDEStat(TreeType& node) :
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0),
      pointsInTheBase(0),
      minKernelRecon(DBL_MAX),
      maxKernelRecon(DBL_MIN)
  {
    // Calculate centroid if necessary.
    if (!tree::TreeTraits<TreeType>::FirstPointIsCentroid)
    {
      node.Center(centroid);
      validCentroid = true;
    }
    else
    {
      validCentroid = false;
    }
  }

  //! Get the centroid of the node.
  inline const arma::vec& Centroid() const
  {
    if (validCentroid)
      return centroid;
    throw std::logic_error("Centroid must be assigned before requesting its "
                           "value");
  }

  //! Get eigenvectors of the base.
  inline const arma::mat& EigVec() const { return eigVec; }

  //! Get eigenvalues of the base.
  inline const arma::vec& EigVal() const { return eigVal; }

  //! Get row mean of all descendant points of the node.
  inline const arma::vec& Mean() const { return mean; }

  //! Get amount of points for which the base is computed.
  inline size_t PointsInTheBase() const { return pointsInTheBase; }

  //! Modify eigenvectors of the base.
  inline arma::mat& EigVec() { return eigVec; }

  //! Modify eigenvalues of the base.
  inline arma::vec& EigVal() { return eigVal; }

  //! Modify row mean of all descendant points of the node.
  inline arma::vec& Mean() { return mean; }

  //! Modify amount of points for which the base is computed.
  inline size_t& PointsInTheBase() { return pointsInTheBase; }

  //! Get minimum kernel value of the points (p_recon, p) in the current base.
  inline double MinKernelRecon() const { return minKernelRecon; }

  //! Modify minimum kernel value of the points (p_recon, p) in the current
  //! base.
  inline double& MinKernelRecon() { return minKernelRecon; }

  //! Get maximum kernel value of the points (p_recon, p) in the current base.
  inline double MaxKernelRecon() const { return maxKernelRecon; }

  //! Modify maximum kernel value of the points (p_recon, p) in the current
  //! base.
  inline double& MaxKernelRecon() { return maxKernelRecon; }

  //! Get accumulated Monte Carlo alpha of the node.
  inline double MCBeta() const { return mcBeta; }

  //! Modify accumulated Monte Carlo alpha of the node.
  inline double& MCBeta() { return mcBeta; }

  //! Get accumulated Monte Carlo alpha of the node.
  inline double AccumAlpha() const { return accumAlpha; }

  //! Modify accumulated Monte Carlo alpha of the node.
  inline double& AccumAlpha() { return accumAlpha; }

  //! Get Monte Carlo alpha of the node.
  inline double MCAlpha() const { return mcAlpha; }

  //! Modify Monte Carlo alpha of the node.
  inline double& MCAlpha() { return mcAlpha; }

  //! Modify the centroid of the node.
  void SetCentroid(arma::vec newCentroid)
  {
    validCentroid = true;
    centroid = std::move(newCentroid);
  }

  //! Get whether the centroid is valid.
  inline bool ValidCentroid() const { return validCentroid; }

  //! Serialize the statistic to/from an archive.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & BOOST_SERIALIZATION_NVP(centroid);
    ar & BOOST_SERIALIZATION_NVP(validCentroid);

    // Backward compatibility: Old versions of KDEStat did not need to handle
    // alpha values.
    if (version > 0)
    {
      ar & BOOST_SERIALIZATION_NVP(mcBeta);
      ar & BOOST_SERIALIZATION_NVP(mcAlpha);
      ar & BOOST_SERIALIZATION_NVP(accumAlpha);
      ar & BOOST_SERIALIZATION_NVP(eigVec);
      ar & BOOST_SERIALIZATION_NVP(eigVal);
      ar & BOOST_SERIALIZATION_NVP(mean);
      ar & BOOST_SERIALIZATION_NVP(pointsInTheBase);
    }
    else if (Archive::is_loading::value)
    {
      mcBeta = -1;
      mcAlpha = -1;
      accumAlpha = -1;
      eigVec.clear();
      eigVal.clear();
      mean.clear();
      pointsInTheBase = 0;
    }
  }

  double maxK = DBL_MIN;
  double minK = DBL_MAX;

 private:
  //! Node centroid.
  arma::vec centroid;

  //! Whether the centroid is updated or is junk.
  bool validCentroid;

  //! Beta value for which mcAlpha is valid.
  double mcBeta;

  //! Monte Carlo alpha for this node.
  double mcAlpha;

  //! Accumulated not used Monte Carlo alpha in the current node.
  double accumAlpha;

  //! Eigenvectors of the base.
  arma::mat eigVec;

  //! Eigenvalues of the base.
  arma::vec eigVal;

  //! Row mean of all descendant points of the node.
  arma::vec mean;

  //! Amount of points for which the base is computed.
  size_t pointsInTheBase;

  //! Minimum kernel value of the points (p_recon, p) in the current base.
  double minKernelRecon;

  //! Maximum kernel value of the points (p_recon, p) in the current base.
  double maxKernelRecon;
};

} // namespace kde
} // namespace mlpack

//! Set the serialization version of the KDEStat class.
BOOST_TEMPLATE_CLASS_VERSION(template<>, mlpack::kde::KDEStat, 1);

#endif
