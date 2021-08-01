
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_mean_shift_test.cpp:

Program Listing for File mean_shift_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_mean_shift_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/mean_shift_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   #include <mlpack/core.hpp>
   static const std::string testName = "MeanShift";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/mean_shift/mean_shift_main.cpp>
   
   #include "test_helper.hpp"
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct MeanShiftTestFixture
   {
    public:
     MeanShiftTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~MeanShiftTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   static void ResetSettings()
   {
     bindings::tests::CleanMemory();
     IO::ClearSettings();
     IO::RestoreSettings(testName);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftOutputDimensionTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     x.randu(3, 100); // 100 points in 3 dimension
   
     // Input random data points.
     SetInputParam("input", std::move(x));
   
     mlpackMain();
   
     // Now check that the output has 1 extra row for labels.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3 + 1);
     // Check number of output points are the same.
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 100);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftLabelOnlyOutputDimensionTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     x.randu(3, 100); // 100 points in 3 dimension
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     SetInputParam("labels_only", true);
   
     mlpackMain();
   
     // Check that there is only 1 row containing all the labels.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 1);
     // Check number of output points are the same.
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 100);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftInPlaceTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("iris_test.csv", x))
       FAIL("Cannot load test dataset iris_test.csv!");
   
     // Get initial number of rows and columns in file.
     int numRows = x.n_rows;
     int numCols = x.n_cols;
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     SetInputParam("in_place", true);
   
     mlpackMain();
   
     // Now check that the output has 1 extra row for labels.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows ==
         (arma::uword) (numRows + 1));
     // Check number of output points are the same.
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == (arma::uword) numCols);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftForceConvergenceTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("iris_test.csv", x))
       FAIL("Cannot load test dataset iris_test.csv!");
   
     // Input random data points.
     SetInputParam("input", x);
     // Set a very small max_iterations.
     SetInputParam("max_iterations", (int) 1);
   
     mlpackMain();
   
     const int numCentroids1 = IO::GetParam<arma::mat>("centroid").n_cols;
   
     ResetSettings();
   
     // Input same random data points.
     SetInputParam("input", std::move(x));
     // Set the same small max_iterations.
     SetInputParam("max_iterations", (int) 1);
     // Set the force_convergence flag on.
     SetInputParam("force_convergence", true);
   
     mlpackMain();
   
     const int numCentroids2 = IO::GetParam<arma::mat>("centroid").n_cols;
     // Resulting number of centroids should be different.
     REQUIRE(numCentroids1 != numCentroids2);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftRadiusTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("iris_test.csv", x))
       FAIL("Cannot load test dataset iris_test.csv!");
   
     // Input random data points.
     SetInputParam("input", x);
     // Set a small radius.
     SetInputParam("radius", (double) 0.1);
   
     mlpackMain();
   
     const int numCentroids1 = IO::GetParam<arma::mat>("centroid").n_cols;
   
     ResetSettings();
   
     // Input same random data points.
     SetInputParam("input", std::move(x));
     // Set a larger radius.
     SetInputParam("radius", (double) 1.0);
   
     mlpackMain();
   
     const int numCentroids2 = IO::GetParam<arma::mat>("centroid").n_cols;
     // Resulting number of centroids should be different.
     REQUIRE(numCentroids1 != numCentroids2);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftMaxIterationsTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     if (!data::Load("iris_test.csv", x))
       FAIL("Cannot load test dataset iris_test.csv!");
   
     // Input random data points.
     SetInputParam("input", x);
     // Set a small max_iterations.
     SetInputParam("max_iterations", (int) 4);
   
     mlpackMain();
   
     const int numCentroids1 = IO::GetParam<arma::mat>("centroid").n_cols;
   
     ResetSettings();
   
     // Input same random data points.
     SetInputParam("input", std::move(x));
     // Set a larger max_iterations.
     SetInputParam("max_iterations", (int) 20);
   
     mlpackMain();
   
     const int numCentroids2 = IO::GetParam<arma::mat>("centroid").n_cols;
     // Resulting number of centroids should be different.
     REQUIRE(numCentroids1 != numCentroids2);
   }
   
   TEST_CASE_METHOD(
       MeanShiftTestFixture, "MeanShiftInvalidMaxIterationsTest",
       "[MeanShiftMainTest][BindingTests]")
   {
     arma::mat x;
     x.randu(3, 100); // 100 points in 3 dimension
   
     // Input random data points.
     SetInputParam("input", std::move(x));
     // Input invalid max number of iterations.
     SetInputParam("max_iterations", (int) -1);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
