/**
 * @file string_util.hpp
 * @author Trironk Kiatkungwanglai
 * @author Ryan Birmingham
 *
 * Declares methods that are useful for writing formatting output.
 */
#ifndef MLPACK_CORE_STRING_UTIL_HPP
#define MLPACK_CORE_STRING_UTIL_HPP

#include <string>

namespace mlpack {
namespace util {

//! A utility function that replaces all all newlines with a number of spaces
//! depending on the indentation level.
std::string Indent(std::string input, const size_t howManyTabs = 1);

} // namespace util
} // namespace mlpack

#endif
