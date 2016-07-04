/**
 * @file pca_main.cpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Main executable to run PCA.
 */
#include <mlpack/core.hpp>

#include "pca.hpp"
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>

using namespace mlpack;
using namespace mlpack::pca;
using namespace std;

// Document program.
PROGRAM_INFO("Principal Components Analysis", "This program performs principal "
    "components analysis on the given dataset using the exact, randomized or "
    "QUIC SVD method. It will transform the data onto its principal components,"
    " optionally performing dimensionality reduction by ignoring the principal "
    "components with the smallest eigenvalues.");

// Parameters for program.
PARAM_STRING_REQ("input_file", "Input dataset to perform PCA on.", "i");
PARAM_STRING_REQ("output_file", "File to save modified dataset to.", "o");

PARAM_INT("new_dimensionality", "Desired dimensionality of output dataset.  If "
    "0, no dimensionality reduction is performed.", "d", 0);
PARAM_DOUBLE("var_to_retain", "Amount of variance to retain; should be between "
    "0 and 1.  If 1, all variance is retained.  Overrides -d.", "r", 0);

PARAM_FLAG("scale", "If set, the data will be scaled before running PCA, such "
    "that the variance of each feature is 1.", "s");

PARAM_STRING("decomposition_method", "Method used for the principal"
    "components analysis: 'exact', 'randomized', 'quic'.", "c", "exact");


//! Run RunPCA on the specified dataset with the given decomposition method.
template<typename DecompositionPolicy>
void RunPCA(arma::mat& dataset,
            const size_t newDimension,
            const size_t scale,
            const double varToRetain)
{
  PCA<DecompositionPolicy> p(scale);

  Log::Info << "Performing PCA on dataset..." << endl;
  double varRetained;

  if (varToRetain != 0)
  {
    if (newDimension != 0)
      Log::Warn << "New dimensionality (-d) ignored because -V was specified."
          << endl;

    varRetained = p.Apply(dataset, varToRetain);
  }
  else
  {
    varRetained = p.Apply(dataset, newDimension);
  }

  Log::Info << (varRetained * 100) << "% of variance retained (" <<
      dataset.n_rows << " dimensions)." << endl;

}

int main(int argc, char** argv)
{
  // Parse commandline.
  CLI::ParseCommandLine(argc, argv);

  // Load input dataset.
  string inputFile = CLI::GetParam<string>("input_file");
  arma::mat dataset;
  data::Load(inputFile, dataset);

  // Find out what dimension we want.
  size_t newDimension = dataset.n_rows; // No reduction, by default.
  if (CLI::GetParam<int>("new_dimensionality") != 0)
  {
    // Validate the parameter.
    newDimension = (size_t) CLI::GetParam<int>("new_dimensionality");
    if (newDimension > dataset.n_rows)
    {
      Log::Fatal << "New dimensionality (" << newDimension
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!" << endl;
    }
  }

  // Get the options for running PCA.
  const size_t scale = CLI::HasParam("scale");
  const double varToRetain = CLI::GetParam<double>("var_to_retain");
  const string decompositionMethod = CLI::GetParam<string>(
      "decomposition_method");

  // Perform PCA.
  if (decompositionMethod == "exact")
  {
<<<<<<< HEAD
    RunPCA<ExactSVDPolicy>(dataset, newDimension, scale, varToRetain);
=======
    if (CLI::GetParam<int>("new_dimensionality") != 0)
      Log::Warn << "New dimensionality (-d) ignored because --var_to_retain was"
          << " specified." << endl;

    varRetained = p.Apply(dataset, CLI::GetParam<double>("var_to_retain"));
>>>>>>> origin/master
  }
  else if(decompositionMethod == "randomized")
  {
    RunPCA<RandomizedSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }
  else if(decompositionMethod == "quic")
  {
    RunPCA<QUICSVDPolicy>(dataset, newDimension, scale, varToRetain);
  }

  // Now save the results.
  string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile, dataset);
}
