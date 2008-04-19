#ifndef TREE_COVER_TREE_H
#define TREE_COVER_TREE_H

#include <fastlib/fastlib.h>

#include "cover_tree.h"

namespace ctree {

  const double BASE = 1.3;
  const double inverse_log_base = 1.0 / log(BASE);
  const index_t NEG_INF = (int) log(0);

  inline double scaled_distance(index_t scale) {
    return pow(BASE, scale);
  }

  inline index_t scale_of_distance(double distance) {
    return (index_t) ceil(log(distance)*inverse_log_base);
  }

  
  class NodeDistances {
    
  private:
    // the point not needed, we just use the index when we 
    // make an arraylist for each of the point
    // Vector point_;
    index_t point_;
    ArrayList<double> distances_;
    
  public:
    
    NodeDistances() {
      distances_.Init(0);
    }
    
    ~NodeDistances() {
    }

    index_t point() {
      return point_;
    }

    ArrayList<double>& distances() {
      return distances_;
    }

    double distances(index_t in) {
      return distances_[in];
    }

    void add_distance(double dist) {
      distances_.PushBackCopy(dist);
      return;
    }

    void Init(index_t point, double dist) {
      point_ = point;
      distances_.PushBackCopy(dist);
      return;
    }
  };
  

  double max_set(ArrayList<NodeDistances*>& set) {

    double max = 0.0;
    for (index_t i = 0; i < set.size(); i++) {
      if(max < set[i]->distances().back()) {
	max = set[i]->distances().back();
      }
    }
    return max;
  }

  void print_space(index_t n) {
    for (index_t i = 0; i < n; i++) {
      printf("\t");
    }
    return;
  }

  void print_tree(index_t depth, CoverTreeNode *top_node) {
    print_space(depth);
    if (top_node->num_of_children() > 0) {
      print_space(depth);
      NOTIFY("Point %"LI"d:\n", top_node->point()+1);
      NOTIFY("scale_depth = %"LI"d, max_dist = %lf, children = %"LI"d\n",
	     top_node->scale_depth(), top_node->max_dist_to_grandchild(),
	     top_node->num_of_children());
      for (index_t i = 0; i < top_node->num_of_children(); i++) {
	print_tree(depth+1, top_node->child(i));
      }
    }
    return;
  }

  void PrintTree(CoverTreeNode *top_node) {
    print_tree(0, top_node);
    return;
  }

  // here we assume that both the point_set and the far set are
  // already initialized
  void split_far(ArrayList<NodeDistances*> point_set, 
		 ArrayList<NodeDistances*> far,
		 index_t scale) {

    double bound = scaled_distance(scale);
    ArrayList<NodeDistances*> near;
    index_t initial_size = far.size();

    near.Init(0);
    for (index_t i = 0; i < point_set.size(); i++) {
      if (point_set[i]->distances().back() > bound) {
	far.PushBackCopy(point_set[i]);
      }
      else {
	near.PushBackCopy(point_set[i]);
      }
    }

    DEBUG_ASSERT_MSG(point_set.size() == 
		     far.size() - initial_size + near.size(), 
		     "split_far: point set size doesn't add up\n");

    point_set.Destruct();
    point_set.Steal(&near);

    return;
  }

  // here we assume that the point_set and the near set are 
  // already initialized
  void split_near(index_t point, const Matrix& data,
		  ArrayList<NodeDistances*> point_set,
		  ArrayList<NodeDistances*> near,
		  index_t scale) {

    double bound = scaled_distance(scale);
    ArrayList<NodeDistances*> far;
    index_t initial_size = near.size();

    far.Init(0);
    for (index_t i = 0; i < point_set.size(); i++) {

      Vector p1, p2;

      data.MakeColumnVector(point, &p1);
      data.MakeColumnVector(point_set->point(), &p2);
      double dist = sqrt(la::DistanceSqEuclidean(p1, p2));
      if (dist > bound) {
	far.PushBackCopy(point_set[i]);
      }
      else {
	point_set[i]->add_distance(dist);
	near.PushBackCopy(point_set[i]);
      }
    }

    DEBUG_ASSERT_MSG(point_set.size() == 
		     near.size() - initial_size + far.size(),
		     "split_near: point set doesn't add up\n");

    point_set.Destruct();
    point_set.Steal(&far);

    return;
  }

  CoverTreeNode *private_make_tree(index_t point, const Matrix& data,
				   index_t current_scale,
				   index_t max_scale, 
				   ArrayList<NodeDistances*> point_set,
				   ArrayList<NodeDistances*> consumed_set) {

    // no other point so leaf in explicit tree
    if (point_set.size() == 0) { 
      CoverTreeNode *node = new CoverTreeNode();
      node->MakeLeafNode(point);
      return node;
    }
    else {
      double max_dist = max_set(point_set);
      index_t next_scale = min(current_scale - 1, scale_of_distance(max_dist));

      // At the -INF level so all points are nodes
      // and we have point with zero distances
      if (next_scale == NEG_INF) { 

	ArrayList<CoverTreeNode*> children;

	children.Init(0);
	children.PushBack()->MakeLeafNode(point);

	for (index_t i = 0; i < point_set.size(); i++) {
	  children.PushBack()->MakeLeafNode(point_set[i]->point());
	  consumed_set.PushBackCopy(point_set[i]);
	}

	DEBUG_ASSERT(children.size() == point_set.size());
	point_set.Resize(0);
	CoverTreeNode *node = new CoverTreeNode();
	node->MakeNode(point, 0.0, 100, &children);

	return node;
      }

      // otherwise you need to recurse
      else {

	ArrayList<NodeDistances*> far;

	far.Init(0);
	split_far(point_set, far, current_scale);

	CoverTreeNode *child = private_make_tree(point, data, 
						 next_scale, max_scale,
						 point_set, consumed_set);

	if (point_set.size() == 0) {

	  point_set.Destruct();
	  point_set.Steal(&far);

	  return child;
	}

	else {

	  ArrayList<CoverTreeNode*> children;
	  ArrayList<NodeDistances*> new_point_set, new_consumed_set;

	  children.Init(0);
	  new_point_set.Init(0);
	  new_consumed_set.Init(0);
	  children.PushBackCopy(child);

	  while (point_set.size() != 0) {

	    index_t new_point;
	    double new_dist = point_set.back()->distances().back();

	    new_point = point_set.back()->point();
	    // remember to check here what to use, PushBackRaw() or AddBack()
	    // so that we can use PopBackInit(Element *dest)
	    point_set.PopBackInit(consumed_set.PushBackRaw()); 

	    split_near(new_point, data, point_set, new_point_set, current_scale);
	    split_near(new_point, data, far, new_point_set, current_scale);

	    CoverTreeNode *child_node = private_make_tree(new_point, data,
							  next_scale,
							  max_scale,
							  new_point_set, 
							  new_consumed_set);

	    child_node->set_dist_to_parent(new_dist);
	    children.PushBackCopy(child_node);

	    double bound = scaled_distance(current_scale);

	    for (index_t i = 0; i < new_point_set.size(); i++) {

	      new_point_set[i]->distances().PopBack();
	      if (new_point_set[i]->distances().back() > bound) {
		point_set.PushBackCopy(new_point_set[i]);
	      }
	      else {
		far.PushBackCopy(new_point_set[i]);
	      }
	    }

	    while (new_consumed_set.size() != 0) {
	      new_consumed_set.back()->distances().PopBack();
	      new_consumed_set.PopBackInit(consumed_set.PushBackRaw());
	    }
	  }

	  point_set.Destruct();
	  point_set.Steal(&far);

	  CoverTreeNode *node = new CoverTreeNode();

	  node->MakeNode(point, max_set(consumed_set), 
			 max_scale - current_scale, &children);

	  return node;
	}
      }
    }
  }

  CoverTreeNode *MakeCoverTree(const Matrix& dataset) {

    index_t n = dataset.n_cols();
    DEBUG_ASSERT(n > 0);
    ArrayList<NodeDistances*> point_set, consumed_set;
    Vector root_point;

    dataset.MakeColumnVector(0, &root_point);
    point_set.Init(0);
    consumed_set.Init(0);

    for (index_t i = 1; i < n; i++) {
      NodeDistances *node_distances;
      Vector point;
      double dist;

      dataset.MakeColumnVector(i, &point);
      dist = sqrt(la::DistanceSqEuclidean(root_point, point));
      node_distances->Init(i, dist);

      point_set.PushBackCopy(node_distances);
    }
    DEBUG_ASSERT(point_set.size() == n - 1);

    double max_dist = max_set(point_set);
    index_t max_scale = scale_of_distance(max_dist);

    CoverTreeNode *root_node;
    root_node = private_make_tree(0, dataset, max_scale,
				  max_scale, point_set,
				  consumed_set);

    return root_node;
  }

};
#endif
