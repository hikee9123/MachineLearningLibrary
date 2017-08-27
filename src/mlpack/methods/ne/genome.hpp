/**
 * @file genome.hpp
 * @author Bang Liu
 *
 * Definition of the Genome class.
 */
#ifndef MLPACK_METHODS_NE_GENOME_HPP
#define MLPACK_METHODS_NE_GENOME_HPP

#include <cassert>
#include <map>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "utils.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a genome.
 * A genome is consist of a group of genes.
 */
class Genome 
{
 public:
  //! Neurons.
  std::vector<NeuronGene> neuronGenes;

  //! Links.
  std::vector<LinkGene> linkGenes;

  /**
   * Default constructor.
   */
  Genome(): 
    id(-1),
    numInput(0),
    numOutput(0),
    fitness(DBL_MAX)
  {}
  
  /**
   * Construct the Genome object with the given parameters.
   *
   * @param id Genome identifier.
   * @param neuronGenes List of genes to construct the genome.
   * @param linkGenes List of links to construct the genome.
   * @param numInput Number of input genes.
   * @param numOutput Number of output genes.
   * @param fitness Initial fitness.
   */
  Genome(int id,
  	     const std::vector<NeuronGene>& neuronGenes,
         const std::vector<LinkGene>& linkGenes,
         int numInput,
         int numOutput,
         double fitness):
    id(id),
    neuronGenes(neuronGenes),
    linkGenes(linkGenes),
    numInput(numInput),
    numOutput(numOutput),
    fitness(fitness)
  {}

  // /**
  //  * Copy constructor.
  //  *
  //  * @param genome The copied genome.
  //  */
  // Genome(const Genome& genome)
  // {
  //   id = genome.id;
  //   neuronGenes = genome.neuronGenes;
  //   linkGenes = genome.linkGenes;
  //   numInput = genome.numInput;
  //   numOutput = genome.numOutput;
  //   fitness = genome.fitness;
  // }

  // *
  //  * Destructor.
   
  // ~Genome() {}

  // /**
  //  * Operator =.
  //  *
  //  * @param genome The genome to be compared with.
  //  */
  // Genome& operator =(const Genome& genome)
  // {
  //   if (this != &genome)
  //   {
  //     id = genome.id;
  //     neuronGenes = genome.neuronGenes;
  //     linkGenes = genome.linkGenes;
  //     numInput = genome.numInput;
  //     numOutput = genome.numOutput;
  //     fitness = genome.fitness;
  //   }

  //   return *this;
  // }

  /**
   * Get genome id.
   */
  int Id() const { return id; }

  /**
   * Set genome id.
   */
  void Id(int id) { this->id = id; }

  /**
   * Get input length.
   */
  int NumInput() { return numInput; }

  /**
   * Set input length.
   */
  void NumInput(int numInput) { this->numInput = numInput; }

  /**
   * Get output length.
   */
  int NumOutput() { return numOutput; }

  /**
   * Set output length.
   */
  void NumOutput(int numOutput) { this->numOutput = numOutput; }

  
  double& Fitness() { return fitness; }

  double Fitness() const { return fitness; }

  /**
   * Set fitness.
   */
  void Fitness(double fitness) { this->fitness = fitness; }

  /**
   * Get neuron number.
   */
  int NumNeuron() const { return neuronGenes.size(); }
  
  /**
   * Get link number.
   */
  int NumLink() const { return linkGenes.size(); }

  /**
   * Get input length.
   */
  int GetNumInput()
  {
    int numInput = 0;
    for (int i = 0; i < neuronGenes.size(); ++i)
    {
      if (neuronGenes[i].Type() == INPUT || neuronGenes[i].Type() == BIAS)
        ++numInput;
    }

    return numInput;
  }

  /**
   * Get output length.
   */
  int GetNumOutput()
  {
    int numOutput = 0;
    for (int i = 0; i < neuronGenes.size(); ++i)
    {  
      if (neuronGenes[i].Type() == OUTPUT)
        ++numOutput;
    }

    return numOutput;
  }

  /**
   * Whether specified neuron id exist in this genome.
   *
   * @param id Check whether id exist in this genome.
   */
  bool HasNeuronId(int id) const
  {
    for (int i = 0; i < NumNeuron(); ++i)
    {
      if (neuronGenes[i].Id() == id)
      {
        return true;
      }
    }
    
    return false;
  }

  /**
   * Get neuron by id.
   *
   * @param id The id of the retrieved neuron.
   * @param neuronGene Return parameter. The retrieved neuron.
   */
  void GetNeuronById(int id, NeuronGene& neuronGene)
  {
    assert(HasNeuronId(id));

    for (int i = 0; i < NumNeuron(); ++i)
    {
      if (neuronGenes[i].Id() == id)
      {
        neuronGene = neuronGenes[i];
        return;
      }
    }
  }

  /**
   * Get neuron's index in the neuron array of this genome by id.
   *
   * @param id Neuron's id.
   */
  int GetNeuronIndex(int id) const
  {
    for(int i = 0; i < NumNeuron(); ++i)
    {
      if (neuronGenes[i].Id() == id)
        return i;
    }

    return -1;  // Id start from 0. -1 means not found.
  }

  /**
   * Get link index in the link array of this genome by innovation id.
   *
   * @param innovId Link's innovation id.
   */
  int GetLinkIndex(int innovId) const
  {
    for(int i = 0; i < NumLink(); ++i)
    {
      if (linkGenes[i].InnovationId() == innovId)
        return i;
    }

    return -1;  // Id start from 0. -1 means not found.
  }

  /**
   * Whether a link exist and enabled.
   *
   * @param innovId Link's innovation id.
   */
  bool ContainEnabledLink(int innovId) const
  {
    for(int i = 0; i < NumLink(); ++i)
    {
      if (linkGenes[i].InnovationId() == innovId &&
          linkGenes[i].Enabled())
        return true;
    }

    return false;
  }

  /**
   * Whether link exist.
   *
   * @param innovId Link's innovation id.
   */
  bool ContainLink(int innovId) const
  {
    for(int i = 0; i < NumLink(); ++i)
    {
      if (linkGenes[i].InnovationId() == innovId)
        return true;
    }

    return false;
  }

  /**
   * Set all the neurons' input and output to be zero.
   */
  void Flush()
  {
    for (int i = 0; i < neuronGenes.size(); ++i)
    {
      neuronGenes[i].Activation(0);
      neuronGenes[i].Input(0);
    }
  }

  /**
   * Sort link genes by toNeuron's depth.
   *
   * Links will be sort by the value of its toNeuron's depth,
   * from 0 (input) to 1 (output). This will be helpful for the
   * calculation of genome's activation.
   */
  void SortLinkGenes()
  {
    struct DepthAndLink
    {
      double depth;
      LinkGene link;
 
      DepthAndLink(double d, LinkGene& l) : depth(d), link(l) {}

      bool operator < (const DepthAndLink& dL) const
      {
        return (depth < dL.depth);
      }
    };

    std::vector<double> toNeuronDepths;
    for (int i = 0; i < linkGenes.size(); ++i)
    {
      NeuronGene toNeuron;
      GetNeuronById(linkGenes[i].ToNeuronId(), toNeuron);
      toNeuronDepths.push_back(toNeuron.Depth());
    }

    std::vector<DepthAndLink> depthAndLinks;
    int linkGenesSize = linkGenes.size();
    for (int i = 0; i < linkGenesSize; ++i)
    {
      depthAndLinks.push_back(DepthAndLink(toNeuronDepths[i], linkGenes[i]));
    }

    std::sort(depthAndLinks.begin(), depthAndLinks.end());

    for (int i = 0; i < linkGenesSize; ++i)
    {
      linkGenes[i] = depthAndLinks[i].link;
    }
  }

  /**
   * Activate genome.
   * 
   * Calculate genome's output given input.
   * The last dimension of input is always 1 for bias. If 0, then means no bias.
   *
   * @param input Input of the genome.
   */
  template<typename VecType>
  void Classify(const VecType& point, VecType& response)
  {
    std::vector<double> input;
    for (size_t i = 0; i < point.n_elem; ++i)
    {
      input.push_back(point(i));
    }
    input.push_back(1.0);

    Activate(input);
    std::vector<double> output;
    Output(output);

    response = VecType(output.size());
    for (size_t i = 0; i < output.size(); ++i)
    {
      response(i) = output[i];
    }
  }

  void Activate(std::vector<double>& input)
  {
    assert(input.size() == numInput);

    SortLinkGenes();
    
    // Set all neurons' input to be 0.
    for (int i = 0; i < NumNeuron(); ++i)
    {
      neuronGenes[i].Input(0);
    }
    
    // Set input neurons.
    for (int i = 0; i < numInput; ++i)
    {
      neuronGenes[i].Input(input[i]);  // assume INPUT, BIAS, OUTPUT, HIDDEN sequence
      neuronGenes[i].Activation(input[i]);
    }

    // Activate hidden and output neurons.
    for (int i = 0; i < NumLink(); ++i)
    {
      int toNeuronIdx = GetNeuronIndex(linkGenes[i].ToNeuronId());
      int fromNeuronIdx = GetNeuronIndex(linkGenes[i].FromNeuronId());
      double input = neuronGenes[toNeuronIdx].Input() + 
                     neuronGenes[fromNeuronIdx].Activation() * 
                     linkGenes[i].Weight() *
                     linkGenes[i].Enabled();
      neuronGenes[toNeuronIdx].Input(input);
        
      if ( (i == NumLink() - 1) ||
           (GetNeuronIndex(linkGenes[i + 1].ToNeuronId()) != toNeuronIdx))
      {
        neuronGenes[toNeuronIdx].CalcActivation();
      }
    }
  }

  /**
   * Get output vector.
   *
   * @param output Return parameter, store the output vector.
   */
  void Output(std::vector<double>& output)
  {
    output.clear();
    for (int i = 0; i < numOutput; ++i)
    {
      output.push_back(neuronGenes[numInput + i].Activation());
    }
  }

  /**
   * Set random link weights between [lo, hi].
   *
   * @param lo Low bound of random weight.
   * @param hi High bound of random weight.
   */
  void RandomizeWeights(const double lo, const double hi)
  {
    for (int i = 0; i < linkGenes.size(); ++i)
    {
      double weight = mlpack::math::Random(lo, hi);
      linkGenes[i].Weight(weight); 
    }
  }

  /**
   * Add link to link list.
   *
   * @param linkGene The new link to add.
   */
  void AddLink(LinkGene& linkGene)
  {
    linkGenes.push_back(linkGene);
  }

  /**
   * Add hidden neuron to neuron list.
   *
   * @param neuronGene The new hidden neuron to add.
   */
  void AddHiddenNeuron(NeuronGene& neuronGene)
  {
    if (neuronGene.Type() == HIDDEN)
    {
      neuronGenes.push_back(neuronGene);
    }
  }

  /**
   * Print genome structure information.
   */
  void PrintGenome()
  {
    printf("---------------------------Genome Start---------------------------\n");
    const char* enumNeuronTypetring[] = { "NONE", "INPUT", "BIAS", "HIDDEN", "OUTPUT" };
    const char* enumActivationFuncTypeString[] = { "SIGMOID", "TANH", "LINEAR", "RELU" };  //NOTICE: keep the same with the enum type.
    const char* boolEnabledString[] = { "False", "True" };

    std::cout << "Neurons: " << neuronGenes.size() << std::endl;
    for(size_t i = 0; i < neuronGenes.size(); ++i)
    {
      printf("  Gene:(id=%i, type=%s, activation func=%s, input=%f, response=%f, depth=%.3f)\n",
             neuronGenes[i].Id(),
             enumNeuronTypetring[neuronGenes[i].Type()],
             enumActivationFuncTypeString[neuronGenes[i].ActFuncType()],
             neuronGenes[i].Input(),
             neuronGenes[i].Activation(),
             neuronGenes[i].Depth());
    }

    std::cout << "Links: " << linkGenes.size() << std::endl;
    for(size_t i = 0; i < linkGenes.size(); ++i)
    {
      printf("  Link:(from=%i, to=%i, weight=%f, enabled=%s, innovation=%i)\n",
             linkGenes[i].FromNeuronId(),
             linkGenes[i].ToNeuronId(),
             linkGenes[i].Weight(),
             boolEnabledString[linkGenes[i].Enabled()],
             linkGenes[i].InnovationId());
    }
    printf("----------------------------Genome End----------------------------\n");
  }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(id, "id");
    ar & data::CreateNVP(numInput, "numInput");
    ar & data::CreateNVP(numOutput, "numOutput");
    ar & data::CreateNVP(fitness, "fitness");

    ar & data::CreateNVP(neuronGenes, "neuronGenes");
    ar & data::CreateNVP(linkGenes, "linkGenes");
  }

  /**
 * Non-intrusive serialization for Neighbor Search class. We need this
 * definition because we are going to use the serialize function for boost
 * variant, which will look for a serialize function for its member types.
 */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

 private:
  //! Genome id.
  int id;

  //! Input length (include bias).
  int numInput;

  //! Output length.
  int numOutput;

  //! Genome fitness.
  double fitness;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENOME_HPP
