/**
 * @file load_image.cpp
 * @author Mehul Kumar Nirala
 *
 * Implementation of image loading functionality via STB.
 */
#include "load.hpp"
#include "image_info.hpp"

#ifdef HAS_STB

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace mlpack {
namespace data {

bool LoadImage(const std::string& filename,
               arma::Mat<unsigned char>& matrix,
               ImageInfo& info,
               const bool fatal)
{
  unsigned char* image;

  if (!ImageFormatSupported(filename))
  {
    std::ostringstream oss;
    oss << "Load(): file type " << Extension(filename) << " not supported. ";
    oss << "Currently it supports: ";
    for (auto extension : loadFileTypes)
      oss << " " << extension;
    oss << "." << std::endl;

    if (fatal)
    {
      Log::Fatal << oss.str();
    }
    else
    {
      Log::Warn << oss.str();
    }

    return false;
  }

  // Temporary variables needed as stb_image.h supports int parameters.
  int tempWidth, tempHeight, tempChannels;

  // For grayscale images.
  if (info.Channels() == 1)
  {
    image = stbi_load(filename.c_str(), &tempWidth, &tempHeight, &tempChannels,
        STBI_grey);
  }
  else
  {
    image = stbi_load(filename.c_str(), &tempWidth, &tempHeight, &tempChannels,
        STBI_rgb);
  }

  if (!image)
  {
    if (fatal)
    {
      Log::Fatal << "Load(): failed to load image '" << filename << "': "
          << stbi_failure_reason() << std::endl;
    }
    else
    {
      Log::Warn << "Load(): failed to load image '" << filename << "': "
          << stbi_failure_reason() << std::endl;
    }

    return false;
  }

  info.Width() = tempWidth;
  info.Height() = tempHeight;
  info.Channels() = tempChannels;

  // Copy image into armadillo Mat.
  matrix = arma::Mat<unsigned char>(image, info.Width() * info.Height() *
      info.Channels(), 1, true, true);

  // Free the image pointer.
  free(image);
  return true;
}

} // namespace data
} // namespace mlpack

#else

namespace mlpack {
namespace data {

bool LoadImage(const std::string& /* filename */,
               arma::Mat<unsigned char>& /* matrix */,
               ImageInfo& /* info */,
               const bool fatal)
{
  if (fatal)
  {
    Log::Fatal << "Load(): mlpack was not compiled with STB support, so images "
        << "cannot be loaded!" << std::endl;
  }
  else
  {
    Log::Warn << "Load(): mlpack was not compiled with STB support, so images "
        << "cannot be loaded!" << std::endl;
  }

  return false;
}

} // namespace data
} // namespace mlpack

#endif
