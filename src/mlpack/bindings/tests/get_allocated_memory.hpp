/**
 * @file get_allocated_memory.hpp
 * @author Ryan Curtin
 *
 * If the parameter has a type that may need to be deleted, return the address
 * of that object.  Otherwise return NULL.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_ALLOCATED_MEMORY_HPP
#define MLPACK_BINDINGS_CLI_GET_ALLOCATED_MEMORY_HPP

#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace tests {

template<typename T>
void* GetAllocatedMemory(
    const util::ParamData& /* d */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
{
  return NULL;
}

template<typename T>
void* GetAllocatedMemory(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  return NULL;
}

template<typename T>
void* GetAllocatedMemory(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // Here we have a model; return its memory location.
  return *boost::any_cast<T*>(&d.value);
}

template<typename T>
void GetAllocatedMemory(const util::ParamData& d,
                        const void* /* input */,
                        void* output)
{
  *((void**) output) =
      GetAllocatedMemory<typename std::remove_pointer<T>::type>(d);
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
