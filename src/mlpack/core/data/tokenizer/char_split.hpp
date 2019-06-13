/**
 * @file char_split.hpp
 * @author Jeffin Sam
 *
 * Implementation of Strtok class which tokenizes using single character.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CHAR_SPLIT_HPP
#define MLPACK_CORE_DATA_CHAR_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include "mlpack/core/boost_backport/boost_backport_string_view.hpp"

namespace mlpack {
namespace data {
/**
 * A simple CharSplit class.
 *
 * The class is used to split the documents using a set of characters, that
 * is used as delimiter. The function returns a token, and successive calls,
 * would return many such tokens.
 */
class CharSplit
{
 public:
  /**
  * A constructor to set deimiter.
  * @param Delimiter A set of characters which you want to use as delimiter.
  */
  CharSplit(const std::string& delimiter)
  {
    this->delimiter = delimiter;
  }
  /**
  * A function object which take boost::string_view as input and
  * return a boost::string_view (token).
  * @param str A string to retieve token from.
  */
  boost::string_view operator()(boost::string_view& str) const
  {
    boost::string_view retval;
    boost::string_view delimiterView(delimiter);
    while (retval.empty())
    {
      std::size_t pos = str.find_first_of(delimiterView);
      if (pos == str.npos)
      {
        retval = str;
        str.clear();
        return retval;
      }
      retval = str.substr(0, pos);
      str.remove_prefix(pos + 1);
    }
    return retval;
  }

 private:
  // A set of characters by which the string is tokenized.
  std::string delimiter;
}; // Strtok class

} // namespace data
} // namespace mlpack

#endif
