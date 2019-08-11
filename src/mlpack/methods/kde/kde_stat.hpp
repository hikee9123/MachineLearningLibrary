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
      accumAlpha(0) { }

  //! Initialization for a fully initialized node.
  template<typename TreeType>
  KDEStat(TreeType& node) :
      mcBeta(0),
      mcAlpha(0),
      accumAlpha(0)
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
    }
    else if (Archive::is_loading::value)
    {
      mcBeta = -1;
      mcAlpha = -1;
      accumAlpha = -1;
    }
  }

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
};

} // namespace kde
} // namespace mlpack

//! Set the serialization version of the KDEStat class.
BOOST_TEMPLATE_CLASS_VERSION(template<>, mlpack::kde::KDEStat, 1);

#endif
