
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_preprocess_preprocess_one_hot_encoding_main.cpp:

Program Listing for File preprocess_one_hot_encoding_main.cpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_preprocess_preprocess_one_hot_encoding_main.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/preprocess/preprocess_one_hot_encoding_main.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/io.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/core.hpp>
   #include <mlpack/core/data/one_hot_encoding.hpp>
   
   // Program Name.
   BINDING_NAME("One Hot Encoding");
   
   // Short description.
   BINDING_SHORT_DESC("A utility to do one-hot encoding on features of dataset.");
   
   // Long description.
   BINDING_LONG_DESC(
       "This utility takes a dataset and a vector of indices and does one-hot "
       "encoding of the respective features at those indices. Indices represent "
       "the IDs of the dimensions to be one-hot encoded."
       "\n\n"
       "The output matrix with encoded features may be saved with the " +
       PRINT_PARAM_STRING("output") + " parameters.");
   
   // Example.
   BINDING_EXAMPLE(
       "So, a simple example where we want to encode 1st and 3rd feature"
       " from dataset " + PRINT_DATASET("X") + " into " +
       PRINT_DATASET("X_output") + " would be"
       "\n\n" +
       PRINT_CALL("preprocess_one_hot_encoding", "input", "X", "output",
           "X_ouput", "dimensions", 1 , "dimensions", 3));
   
   // See also...
   BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
   BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
   BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");
   BINDING_SEE_ALSO("One-hot encoding on Wikipedia",
           "https://en.m.wikipedia.org/wiki/One-hot");
   
   // Define parameters for data.
   PARAM_MATRIX_IN_REQ("input", "Matrix containing data.", "i");
   PARAM_MATRIX_OUT("output", "Matrix to save one-hot encoded features "
       "data to.", "o");
   
   PARAM_VECTOR_IN_REQ(int, "dimensions", "Index of dimensions that"
       "need to be one-hot encoded.", "d");
   
   using namespace mlpack;
   using namespace mlpack::util;
   using namespace arma;
   using namespace std;
   
   static void mlpackMain()
   {
     // Load the data.
     const arma::mat& data = IO::GetParam<arma::mat>("input");
     vector<int>& indices = IO::GetParam<vector<int> >("dimensions");
     vector<size_t> copyIndices(indices.size());
     RequireParamValue<std::vector<int>>("dimensions", [data](std::vector<int> x)
         {
           for (int dim : x)
           {
             if (dim < 0 || (size_t)dim > data.n_rows)
             {
               return false;
             }
           }
           return true;
         }, true, "dimensions must be greater than 0 "
         "and less than the number of dimensions");
     for (size_t i = 0; i < indices.size(); ++i)
     {
       copyIndices[i] = (size_t)indices[i];
     }
     arma::mat output;
     data::OneHotEncoding(data, (arma::Col<size_t>)(copyIndices), output);
     if (IO::HasParam("output"))
       IO::GetParam<arma::mat>("output") = std::move(output);
   }
