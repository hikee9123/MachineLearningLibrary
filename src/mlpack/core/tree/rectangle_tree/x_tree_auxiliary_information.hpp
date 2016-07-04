/**
 * @file no_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the XTreeAuxiliaryInformation class, a class that provides
 * some x-tree specific information about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP

namespace mlpack {
namespace tree {

/**
 * The XTreeAuxiliaryInformation class provides information specific to X trees
 * for each node in a RectangleTree.
 */
template<typename TreeType>
class XTreeAuxiliaryInformation
{
 public:
  //! Default constructor
  XTreeAuxiliaryInformation() :
    normalNodeMaxNumChildren(0),
    splitHistory(0)
  { };

  /**
   * Construct this whith the specified node.
   *
   * @param node The node that stores this auxiliary information.
   */
  XTreeAuxiliaryInformation(const TreeType* node) :
      normalNodeMaxNumChildren(node->Parent() ?
          node->Parent()->AuxiliaryInfo().NormalNodeMaxNumChildren() :
          node->MaxNumChildren()),
      splitHistory(node->Bound().Dim())
  { };

  /**
   * Create an auxiliary information object by copying from the other node.
   *
   * @param other The node from which the information will be copied.
   */
  XTreeAuxiliaryInformation(const TreeType& other) :
      normalNodeMaxNumChildren(
          other.AuxiliaryInfo().NormalNodeMaxNumChildren()),
      splitHistory(other.AuxiliaryInfo().SplitHistory())
  { };

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method allows the auxiliary information the option of manipulating the
   * tree in order to perform the insertion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param node The node in which the point is being inserted.
   * @param point The global number of the point being inserted.
   */
  bool HandlePointInsertion(TreeType* /* node */, const size_t /* point */)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method allows the auxiliary information the option of manipulating the
   * tree in order to perform the insertion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param node The node in which the nodeToInsert is being inserted.
   * @param nodeToInsert The node being inserted.
   * @param insertionLevel The level of the tree at which the nodeToInsert
   *        should be inserted.
   */
  bool HandleNodeInsertion(TreeType* /* node */,
                           TreeType* /* nodeToInsert */,
                           bool /* insertionLevel */)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the deletion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   * @param node The node from which the point is being deleted.
   * @param localIndex The local index of the point being deleted.
   */
  bool HandlePointDeletion(TreeType* , const size_t)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the deletion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   * @param node The node from which the node is being deleted.
   * @param nodeIndex The local index of the node being deleted.
   */
  bool HandleNodeRemoval(TreeType* , const size_t)
  {
    return false;
  }

  /**
   * Some tree types require to propagate the information upward.
   * This method should return false if this is not the case. If true is
   * returned, the update will be propogated upward.
   * @param node The node in which the auxiliary information being update.
   */
  bool UpdateAuxiliaryInfo(TreeType* )
  {
    return false;
  }

  /**
   * Nullify the auxiliary information in order to prevent an invalid free.
   */
  void NullifyData()
  { }

  /**
   * The X tree requires that the tree records it's "split history".  To make
   * this easy, we use the following structure.
   */
  typedef struct SplitHistoryStruct
  {
    int lastDimension;
    std::vector<bool> history;

    SplitHistoryStruct(int dim) : lastDimension(0), history(dim)
    {
      for (int i = 0; i < dim; i++)
        history[i] = false;
    }

    template<typename Archive>
    void Serialize(Archive& ar, const unsigned int /* version */)
    {
      ar & data::CreateNVP(lastDimension, "lastDimension");
      ar & data::CreateNVP(history, "history");
    }
  } SplitHistoryStruct;

 private:
    //! The max number of child nodes a non-leaf normal node can have.
  size_t normalNodeMaxNumChildren;
  //! A struct to store the "split history" for X trees.
  SplitHistoryStruct splitHistory;

 public:
  //! Return the maximum number of a normal node's children.
  size_t NormalNodeMaxNumChildren() const { return normalNodeMaxNumChildren; }
  //! Modify the maximum number of a normal node's children.
  size_t& NormalNodeMaxNumChildren() { return normalNodeMaxNumChildren; }
  //! Return the split history of the node assosiated with this object.
  const SplitHistoryStruct& SplitHistory() const { return splitHistory; }
  //! Modify the split history of the node assosiated with this object.
  SplitHistoryStruct& SplitHistory() { return splitHistory; }

  /**
   * Serialize the information.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using data::CreateNVP;

    ar & CreateNVP(normalNodeMaxNumChildren, "normalNodeMaxNumChildren");
    ar & CreateNVP(splitHistory, "splitHistory");
  }

};

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_AUXILIARY_INFORMATION_HPP
