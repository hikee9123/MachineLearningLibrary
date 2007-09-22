#ifndef MULTIBODY_H
#define MULTIBODY_H

#include <values.h>

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/series_expansion_aux.h"

template<typename TKernel>
class NaiveMultibody {

  FORBID_COPY(NaiveMultibody);

 private:

  /** Temporary space for storing indices selected for exhaustive computation
   */
  ArrayList<int> exhaustive_indices_;

  /** Temporary space for storing pairwise distances */
  Matrix distmat_;

  /** dataset for the tree */
  Matrix data_;

  /** kernel function */
  TKernel kernel_;

  /** the total number of n-tuples to consider */
  double total_num_tuples_;

  /** potential estimate */
  double potential_e_;

  /** exhaustive computer */
  void NMultibody(int level) {
    
    int num_nodes = 3;
    int start_index = 0;
    double result = 0;

    if(level < num_nodes) {
      
      if(level == 0) {
	start_index = 0;
      }
      else {
	start_index = exhaustive_indices_[level - 1] + 1;
      }
      
      for(index_t i = start_index; i < data_.n_cols() - 
	    (num_nodes - level - 1); i++) {
	
	exhaustive_indices_[level] = i;
	NMultibody(level + 1);
      }
    }
    else {
      for(index_t i = 0; i < num_nodes; i++) {
	const double *i_col = data_.GetColumnPtr(exhaustive_indices_[i]);
	for(index_t j = i + 1; j < num_nodes; j++) {
	  const double *j_col = data_.GetColumnPtr(exhaustive_indices_[j]);
	  distmat_.set(i, j, la::DistanceSqEuclidean(data_.n_rows(), i_col,
						     j_col));
	}
      }

      result = 1e-27 * kernel_.EvalUnnormOnSq(distmat_.get(0, 1)) *
        kernel_.EvalUnnormOnSq(distmat_.get(0, 2)) *
        kernel_.EvalUnnormOnSq(distmat_.get(1, 2));

      potential_e_ += result;
    }
  }

 public:

  NaiveMultibody() {}
  
  ~NaiveMultibody() {}

  void Init(const Matrix &data, double bandwidth) {
    data_.Alias(data);
    distmat_.Init(3, 3);
    exhaustive_indices_.Init(3);
    kernel_.Init(bandwidth);
    total_num_tuples_ = 0;

    potential_e_ = 0;
  }

  void Compute() {

    NMultibody(0);

    printf("Got potential sum %g\n", potential_e_);
  }

};

template<typename TKernel, typename TKernelDerivative>
class MultitreeMultibody {

  FORBID_COPY(MultitreeMultibody);

public:

  /** Statatistics stored in each node of the tree */
  class MultibodyStat {
    
  public:
    
    /** Summed up potential for query points in this node */
    double potential_;

    /**
     * Extra amount of error that can be spent for the query points in
     * this node.
     */
    double extra_token_;

    /**
     * Far field expansion created by the reference points in this node.
     */
    FarFieldExpansion<TKernel, TKernelDerivative> farfield_expansion_;

    /**
     * Local expansion stored in this node.
     */
    LocalExpansion<TKernel, TKernelDerivative> local_expansion_;

    /** Initialize the statistics */
    void Init() {
    }

    void Init(double bandwidth, SeriesExpansionAux *sea) {
      potential_ = 0;
      extra_token_ = 0;
      farfield_expansion_.Init(bandwidth, sea);
      local_expansion_.Init(bandwidth, sea);
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
      Init();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const MultibodyStat& left_stat, 
	      const MultibodyStat& right_stat) {
      Init();
    }

    void Init(double bandwidth, const Vector& center, 
	      SeriesExpansionAux *sea) {
      
      potential_ = 0;
      extra_token_ = 0;
      farfield_expansion_.Init(bandwidth, center, sea);
      local_expansion_.Init(bandwidth, center, sea);
    }

    MultibodyStat() { }
    
    ~MultibodyStat() {}

  };

  typedef BinarySpaceTree<DHrectBound<2>, Matrix, MultibodyStat> Tree;

  typedef TKernel Kernel;
  
  typedef TKernelDerivative KernelDerivative;

  // constructor/destructor
  MultitreeMultibody() {}
  
  ~MultitreeMultibody() { delete root_; }

  // getters/setters
  
  const Matrix &get_data() const { return data_; }

  // interesting functions...

  /** Main computation */
  void Compute(double tau) {

    ArrayList<Tree *> root_nodes;
    root_nodes.Init(3);

    // store node pointers
    for(index_t i = 0; i < 3; i++) {
      root_nodes[i] = root_;
    }
    
    total_num_tuples_ = math::BinomialCoefficient(data_.n_cols(), 3);
    tau_ = tau;

    // run and do timing for multitree multibody
    MTMultibody(root_nodes, total_num_tuples_);

    printf("Total potential estimate: %g\n", potential_e_);
  }

  void InitExpansionObjects(Tree *node) {
    
    if(node != NULL) {
      Vector far_center;
      Vector local_center;
      far_center.Alias(node->stat().farfield_expansion_.get_center());
      local_center.Alias(node->stat().local_expansion_.get_center());
      node->bound().CalculateMidpoint(&far_center);
      node->bound().CalculateMidpoint(&local_center);
      
      node->stat().Init(sqrt(kernel_.bandwidth_sq()), &sea_);
    }

    if(!node->is_leaf()) {
      InitExpansionObjects(node->left());
      InitExpansionObjects(node->right());
    }
  }

  /** Initialize the kernel object, and build the tree */
  void Init(double bandwidth) {

    fx_timer_start(NULL, "tree_d");
    tree::LoadKdTree(NULL, &data_, &root_, NULL);

    sea_.Init(10, data_.n_rows());
    
    kernel_.Init(bandwidth);

    InitExpansionObjects(root_);

    fx_timer_stop(NULL, "tree_d");

    // Warning, I should fix this so that it generalizes to any number of
    // tuples... May involves coming up with a solution that involves
    // templates...
    non_leaf_indices_.Init(3);
    distmat_.Init(3, 3);
    exhaustive_indices_.Init(3);

    potential_l_ = potential_e_ = 0;
    extra_token_ = 0;
  }

private:
  
  /** The current list of non-leaf indices */
  ArrayList<int> non_leaf_indices_;

  /** Temporary space for storing indices selected for exhaustive computation 
   */
  ArrayList<int> exhaustive_indices_;

  /** Temporary space for storing pairwise distances */
  Matrix distmat_;

  /** pointer to the root of the tree */
  Tree *root_;

  /** dataset for the tree */
  Matrix data_;

  /** series approximation auxiliary computations */
  SeriesExpansionAux sea_;

  /** kernel function */
  Kernel kernel_;

  /** the total number of n-tuples to consider */
  double total_num_tuples_;
  
  /**
   * Extra amount of error that can be spent for the query points in
   * this node.
   */
  double extra_token_;

  /** potential estimate */
  double potential_e_;
  
  /** Running lower bound on the potential */
  double potential_l_;

  /** approximation relative error bound */
  double tau_;

  int as_indexes_strictly_surround_bs(Tree *a, Tree *b) {
    return (a->begin() < b->begin() && a->end() >= b->end()) ||
      (a->begin() <= b->begin() && a->end() > b->end());
  }

  /** 
   * Compute the total number of n-tuples by recursively splitting up
   * the i-th node 
   */
  double two_ttn(int b, ArrayList<Tree *> nodes, int i) {

    double result = 0.0;
    Tree *kni = nodes[i];
    nodes[i] = kni->left();
    result += ttn(b, nodes);
    nodes[i] = kni->right();
    result += ttn(b, nodes);
    nodes[i] = kni;
    return result;
  }

  /** Compute the total number of n-tuples */
  double ttn(int b, ArrayList<Tree *> nodes) {
    
    Tree *bkn = nodes[b];
    double result;
    int n = nodes.size();

    if(b == n - 1) {
      result = (double) bkn->count();
    }
    else {
      int j;
      int conflict = 0;
      int simple_product = 1;
      
      result = (double) bkn->count();
      
      for(j = b + 1 ; j < n && !conflict; j++) {
	Tree *knj = nodes[j];
	
	if (bkn->begin() >= knj->end() - 1) {
	  conflict = 1;
	}
	else if(nodes[j - 1]->end() - 1 > knj->begin()) {
	  simple_product = 0;
	}
      }
      
      if(conflict) {
	result = 0.0;
      }
      else if(simple_product) {
	for(j = b + 1; j < n; j++) {
	  result *= nodes[j]->count();
	}
      }
      else {
	int jdiff = -1; 

	// undefined... will eventually point to the
	// lowest j > b such that nodes[j] is different from
	// bkn	
	for(j = b + 1; jdiff < 0 && j < n; j++) {
	  Tree *knj = nodes[j];
	  if(bkn->begin() != knj->begin() ||
	     bkn->end() - 1 != knj->end() - 1) {
	    jdiff = j;
	  }
	}

	if(jdiff < 0) {
	  result = math::BinomialCoefficient(bkn->count(), n - b);
	}
	else {
	  Tree *dkn = nodes[jdiff];

	  if(dkn->begin() >= bkn->end() - 1) {
	    result = math::BinomialCoefficient(bkn->count(), jdiff - b);
	    if(result > 0.0) {
	      result *= ttn(jdiff, nodes);
	    }
	  }
	  else if(as_indexes_strictly_surround_bs(bkn, dkn)) {
	    result = two_ttn(b, nodes, b);
	  }
	  else if(as_indexes_strictly_surround_bs(dkn, bkn)) {
	    result = two_ttn(b, nodes, jdiff);
	  }
	}
      }
    }
    return result;
  }

  /** Heuristic for node splitting - find the node with most points */
  int FindSplitNode(ArrayList<Tree *> nodes) {

    int global_index = -1;
    double global_min = MAXDOUBLE;

    for(index_t i = 0; i < non_leaf_indices_.size(); i++) {

      int non_leaf_index = non_leaf_indices_[i];
      double minimum_side_length = MAXDOUBLE;

      // find out the minimum side length
      for(index_t j = 0; j < data_.n_rows(); j++) {
	
	DRange range = nodes[non_leaf_index]->bound().get(j);
	double side_length = range.width();
	
	if(side_length < minimum_side_length) {
	  minimum_side_length = side_length;
	}
      }
      if(minimum_side_length < global_min) {
	global_min = minimum_side_length;
	global_index = non_leaf_index;
      }
    }
    return global_index;
  }

  /** Pruning rule */
  int Prunable(ArrayList<Tree *> nodes, double num_tuples) {

    int i, j;
    double min_potential, max_potential;
    double dsqd_ij_min, dsqd_ij_max, dsqd_ik_min, dsqd_ik_max, dsqd_jk_min,
      dsqd_jk_max;
    int num_nodes = nodes.size();
    double lower_change;
    double error, estimate;
    double dmin = 0, dmax = 0;
    
    // compute pairwise bounding box distances
    for(i = 0; i < num_nodes - 1; i++) {
      Tree *node_i = nodes[i];

      for(j = i + 1; j < num_nodes; j++) {
	Tree *node_j = nodes[j];
	
	dmin = node_i->bound().MinDistanceSq(node_j->bound());
	dmax = node_i->bound().MaxDistanceSq(node_j->bound());

	distmat_.set(i, j, dmin);
	distmat_.set(j, i, dmax);
      }
    }

    dsqd_ij_min = distmat_.get(0, 1);
    dsqd_ij_max = distmat_.get(1, 0);
    dsqd_ik_min = distmat_.get(0, 2);
    dsqd_ik_max = distmat_.get(2, 0);
    dsqd_jk_min = distmat_.get(1, 2);
    dsqd_jk_max = distmat_.get(2, 1);
    
    min_potential = 1e-27 * kernel_.EvalUnnormOnSq(dsqd_ij_max) *
      kernel_.EvalUnnormOnSq(dsqd_ik_max) *
      kernel_.EvalUnnormOnSq(dsqd_jk_max);
    max_potential = 1e-27 * kernel_.EvalUnnormOnSq(dsqd_ij_min) *
      kernel_.EvalUnnormOnSq(dsqd_ik_min) *
      kernel_.EvalUnnormOnSq(dsqd_jk_min);

    lower_change = num_tuples * min_potential;
    
    error = 0.5 * (max_potential - min_potential);
    
    estimate = 0.5 * num_tuples * (min_potential + max_potential);

    // compute whether the error is below the threshold
    if(error <= tau_ * (potential_l_ + lower_change) / 
       total_num_tuples_ + extra_token_) {
      
      potential_l_ += lower_change;
      potential_e_ += estimate;
      extra_token_ = num_tuples * (potential_l_ * tau_ / total_num_tuples_ - error) 
	+ extra_token_;

      DEBUG_ASSERT(extra_token_ >= 0);
      return 1;
    }
    return 0;
  }

  /** Base exhaustive case */
  void MTMultibodyBase(ArrayList<Tree *> nodes, int level) {

    int start_index;
    double result;
    int num_nodes = nodes.size();

    if(level < num_nodes) {
      
      /* run over each point in this node */
      if(level > 0) {
	if(nodes[level - 1] == nodes[level]) {
	  start_index = exhaustive_indices_[level - 1] + 1;
	}
	else {
	  start_index = nodes[level]->begin();
	}
      }
      else {
	start_index = nodes[level]->begin();
      }
      
      for(index_t i = start_index; i < (nodes[level])->end(); i++) {	
	exhaustive_indices_[level] = i;
	MTMultibodyBase(nodes, level + 1);
      }
    }
    else {

      // complete the table of distance computation
      for(index_t i = 0; i < num_nodes; i++) {
	const double *i_col = data_.GetColumnPtr(exhaustive_indices_[i]);

	for(index_t j = i + 1; j < num_nodes; j++) {
	  
	  const double *j_col = data_.GetColumnPtr(exhaustive_indices_[j]);
	  distmat_.set(i, j, la::DistanceSqEuclidean(data_.n_rows(), i_col, 
						     j_col));
	}
      }
      
      result = 1e-27 * kernel_.EvalUnnormOnSq(distmat_.get(0, 1)) *
	kernel_.EvalUnnormOnSq(distmat_.get(0, 2)) *
	kernel_.EvalUnnormOnSq(distmat_.get(1, 2));
      
      potential_e_ += result;
      potential_l_ += result;
    }
  }

  /** Main multitree recursion */
  void MTMultibody(ArrayList<Tree *> nodes, double num_tuples) {
    
    if(Prunable(nodes, num_tuples)) {
      return;
    }

    // figure out which ones are non-leaves
    non_leaf_indices_.Resize(0);
    for(index_t i = 0; i < 3; i++) {
      if(!(nodes[i]->is_leaf())) {
	non_leaf_indices_.AddBackItem(i);
      }
    }
    
    // all leaves, then base case
    if(non_leaf_indices_.size() == 0) {
      MTMultibodyBase(nodes, 0);
      return;
    }
    
    // else, split an internal node and recurse
    else {
      int split_index;
      double new_num_tuples;
      
      // copy to new nodes
      ArrayList<Tree *> new_nodes;
      new_nodes.Init(3);
      for(index_t i = 0; i < 3; i++) {
	new_nodes[i] = nodes[i];
      }

      // apply splitting heuristic
      split_index = FindSplitNode(nodes);

      // recurse to the left
      new_nodes[split_index] = nodes[split_index]->left();
      new_num_tuples = ttn(0, new_nodes);
      
      if(new_num_tuples > 0) {
	MTMultibody(new_nodes, new_num_tuples);
      }
      
      // recurse to the right
      new_nodes[split_index] = nodes[split_index]->right();
      new_num_tuples = ttn(0, new_nodes);
      
      if(new_num_tuples > 0) {
	MTMultibody(new_nodes, new_num_tuples);
      }
    }
  }

};

#endif
