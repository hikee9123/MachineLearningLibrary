### mlpack 1.1.0
###### ????-??-?? 
  * Removed overclustering support from k-means because it is not well-tested,
    may be buggy, and is (I think) unused.  If this was support you were using,
    open a bug or get in touch with us; it would not be hard for us to
    reimplement it.

  * Refactored KMeans to allow different types of Lloyd iterations.

  * Added implementations of k-means: Elkan's algorithm, Hamerly's algorithm,
    Pelleg-Moore's algorithm, and the DTNN (dual-tree nearest neighbor)
    algorithm.

### mlpack 1.0.11 
###### 2014-12-11
  * Proper handling of dimension calculation in PCA.

  * Load parameter vectors properly for LinearRegression models.

  * Linker fixes for AugLagrangian specializations under Visual Studio.

  * Add support for observation weights to LinearRegression.

  * MahalanobisDistance<> now takes root of the distance by default and
    therefore satisfies the triangle inequality (TakeRoot now defaults to true).

  * Better handling of optional Armadillo HDF5 dependency.

  * Fixes for numerous intermittent test failures.

  * math::RandomSeed() now sets the random seed for recent (>=3.930) Armadillo
    versions.

  * Handle Newton method convergence better for
    SparseCoding::OptimizeDictionary() and make maximum iterations a parameter.

  * Known bug: CosineTree construction may fail in some cases on i386 systems.
    (#376)

### mlpack 1.0.10
###### 2014-08-29    
  * Bugfix for NeighborSearch regression which caused very slow allknn/allkfn.
    Speeds are now restored to approximately 1.0.8 speeds, with significant
    improvement for the cover tree.

  * Detect dependencies correctly when ARMA_USE_WRAPPER is not being defined
    (i.e., libarmadillo.so does not exist).

  * Bugfix for compilation under Visual Studio.

### mlpack 1.0.9
###### 2014-07-28
  * GMM initialization is now safer and provides a working GMM when constructed
    with only the dimensionality and number of Gaussians (#314).

  * Check for division by 0 in Forward-Backward Algorithm in HMMs (#314).

  * Fix MaxVarianceNewCluster (used when re-initializing clusters for k-means)
    (#314).

  * Fixed implementation of Viterbi algorithm in HMM::Predict() (#316).

  * Significant speedups for dual-tree algorithms using the cover tree (#243, #329) including a faster implementation of FastMKS.

  * Fix for LRSDP optimizer so that it compiles and can be used (#325).

  * CF (collaborative filtering) now expects users and items to be zero-indexed,
    not one-indexed (#324).

  * CF::GetRecommendations() API change: now requires the number of
    recommendations as the first parameter.  The number of users in the local
    neighborhood should be specified with CF::NumUsersForSimilarity().

  * Removed incorrect PeriodicHRectBound (#30).

  * Refactor LRSDP into LRSDP class and standalone function to be optimized
    (#318).

  * Fix for centering in kernel PCA (#355).

  * Added simulated annealing (SA) optimizer, contributed by Zhihao Lou.

  * HMMs now support initial state probabilities; these can be set in the
    constructor, trained, or set manually with HMM::Initial() (#315).

  * Added Nyström method for kernel matrix approximation by Marcus Edel.

  * Kernel PCA now supports using Nyström method for approximation.

  * Ball trees now work with dual-tree algorithms, via the BallBound<> bound
    structure (#320); fixed by Yash Vadalia.

  * The NMF class is now AMF<>, and supports far more types of factorizations,
    by Sumedh Ghaisas.

  * A QUIC-SVD implementation has returned, written by Siddharth Agrawal and
    based on older code from Mudit Gupta.

  * Added perceptron and decision stump by Udit Saxena (these are weak learners
    for an eventual AdaBoost class).

  * Sparse autoencoder added by Siddharth Agrawal.

### mlpack 1.0.8
###### 2014-01-06    
  * Memory leak in NeighborSearch index-mapping code fixed (#310).

  * GMMs can be trained using the existing model as a starting point by
    specifying an additional boolean parameter to GMM::Estimate() (#308).

  * Logistic regression implementation added in methods/logistic_regression (see
    also #305).

  * L-BFGS optimizer now returns its function via Function().

  * Version information is now obtainable via mlpack::util::GetVersion() or the
    __MLPACK_VERSION_MAJOR, __MLPACK_VERSION_MINOR, and  __MLPACK_VERSION_PATCH
    macros (#309).

  * Fix typos in allkfn and allkrann output.

### mlpack 1.0.7
###### 2013-10-04    
  * Cover tree support for range search (range_search), rank-approximate nearest
    neighbors (allkrann), minimum spanning tree calculation (emst), and FastMKS
    (fastmks).

  * Dual-tree FastMKS implementation added and tested.

  * Added collaborative filtering package (cf) that can provide recommendations
    when given users and items.

  * Fix for correctness of Kernel PCA (kernel_pca) (#280).

  * Speedups for PCA and Kernel PCA (#204).

  * Fix for correctness of Neighborhood Components Analysis (NCA) (#289).

  * Minor speedups for dual-tree algorithms.

  * Fix for Naive Bayes Classifier (nbc) (#279).

  * Added a ridge regression option to LinearRegression (linear_regression)
    (#298).

  * Gaussian Mixture Models (gmm::GMM<>) now support arbitrary covariance matrix
    constraints (#294).

  * MVU (mvu) removed because it is known to not work (#189).

  * Minor updates and fixes for kernels (in mlpack::kernel).

### mlpack 1.0.6
###### 2013-06-13    
  * Minor bugfix so that FastMKS gets built.

### mlpack 1.0.5
###### 2013-05-01   
  * Speedups of cover tree traversers (#243).

  * Addition of rank-approximate nearest neighbors (RANN), found in
    src/mlpack/methods/rann/.

  * Addition of fast exact max-kernel search (FastMKS), found in
    src/mlpack/methods/fastmks/.

  * Fix for EM covariance estimation; this should improve GMM training time.

  * More parameters for GMM estimation.

  * Force GMM and GaussianDistribution covariance matrices to be positive
    definite, so that training converges much more often.

  * Add parameter for the tolerance of the Baum-Welch algorithm for HMM
    training.

  * Fix for compilation with clang compiler.

  * Fix for k-furthest-neighbor-search.

### mlpack 1.0.4
###### 2013-02-08    
  * Force minimum Armadillo version to 2.4.2.

  * Better output of class types to streams; a class with a ToString() method
    implemented can be sent to a stream with operator<<.  See #164.

  * Change return type of GMM::Estimate() to double (#266).

  * Style fixes for k-means and RADICAL.

  * Handle size_t support correctly with Armadillo 3.6.2 (#267).

  * Add locality-sensitive hashing (LSH), found in src/mlpack/methods/lsh/.

  * Better tests for SGD (stochastic gradient descent) and NCA (neighborhood
    components analysis).

### mlpack 1.0.3
###### 2012-09-16    

  * Remove internal sparse matrix support because Armadillo 3.4.0 now includes
    it.  When using Armadillo versions older than 3.4.0, sparse matrix support
    is not available.

  * NCA (neighborhood components analysis) now support an arbitrary optimizer
    (#254), including stochastic gradient descent (#258).

### mlpack 1.0.2
###### 2012-08-15    
  * Added density estimation trees, found in src/mlpack/methods/det/.

  * Added non-negative matrix factorization, found in src/mlpack/methods/nmf/.

  * Added experimental cover tree implementation, found in
    src/mlpack/core/tree/cover_tree/ (#156).

  * Better reporting of boost::program_options errors (#231).

  * Fix for timers on Windows (#218, #217).

  * Fix for allknn and allkfn output (#210).

  * Sparse coding dictionary initialization is now a template parameter (#226).

### mlpack 1.0.1
###### 2012-03-03    
  * Added kernel principal components analysis (kernel PCA), found in
    src/mlpack/methods/kernel_pca/ (#47).

  * Fix for Lovasz-Theta AugLagrangian tests (#188).

  * Fixes for allknn output (#191, #192).

  * Added range search executable (#198).

  * Adapted citations in documentation to BiBTeX; no citations in -h output
    (#201).

  * Stop use of 'const char*' and prefer 'std::string' (#183).

  * Support seeds for random numbers (#182).

### mlpack 1.0.0
###### 2011-12-17    
  * Initial release.  See any resolved tickets numbered less than #196 or
    execute this query:
    http://www.mlpack.org/trac/query?status=closed&milestone=mlpack+1.0.0
