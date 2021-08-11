/**
 * @file methods/decision_tree/multiple_random_dimension_select.hpp
 * @author Ryan Curtin
 *
 * Select a number of random dimensions to pick from.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP
#define MLPACK_METHODS_DECISION_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP

namespace mlpack {
namespace tree {

/**
 * This dimension selection policy allows the selection from a few random
 * dimensions.  The number of random dimensions to use is specified in the
 * constructor.
 */
class MultipleRandomDimensionSelect
{
 public:
  /**
   * Instantiate the MultipleRandomDimensionSelect object.
   *
   * @param numDimensions Number of random dimensions to try for each split.
   */
  MultipleRandomDimensionSelect(const size_t numDimensions = 0) :
        numDimensions(numDimensions),
        dimensions(0)
  { }

  /**
   * Get the current dimension.
   */
  size_t GetDimension(size_t i) const { return values[i]; }

  /**
   * Populates the set of randomly selected dimensions and 
   * return the total number of dimensions to iterate.
   */
  size_t NumDimensions()
  {
    // Reset if possible.
    if (numDimensions == 0 || numDimensions > dimensions)
      numDimensions = (size_t) std::sqrt(dimensions);

    values.set_size(numDimensions);

    // Try setting new values.
    for (size_t i = 0; i < numDimensions; ++i)
    {
      // Generate random different numbers.
      bool unique = false;
      size_t value;
      while (!unique)
      {
        value = math::RandInt(dimensions);

        // Check if we already have the value.
        unique = true;
        for (size_t j = 0; j < i; ++j)
        {
          if (values[j] == value)
          {
            unique = false;
            break;
          }
        }
      }

      values[i] = value;
    }

    return numDimensions;
  }

  //! Get the number of dimensions.
  size_t Dimensions() const { return dimensions; }
  //! Set the number of dimensions.
  size_t& Dimensions() { return dimensions; }

 private:
  //! The number of dimensions.
  size_t numDimensions;
  //! The values we select from.
  arma::Col<size_t> values;
  //! Number of dimensions.
  size_t dimensions;
};

} // namespace tree
} // namespace mlpack

#endif
