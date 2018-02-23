/**
 * @file policygrad.hpp
 * @author Rohan Raj
 *
 * Definition of the PolicyGradient class, which implements policy gradient network in Mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_POLICY_GRADIENT_LEARNING_HPP
#define MLPACK_METHODS_RL_POLICY_GRADIENT_LEARNING_HPP

#include <mlpack/prereqs.hpp>

#include "replay/random_replay.hpp"
#include "training_config.hpp"

namespace mlpack {
namespace rl {
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
class PolicyGradient
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  PolicyGradient(TrainingConfig config,
            NetworkType network,
            PolicyType policy,
            ReplayType replayMethod,
            UpdaterType updater = UpdaterType(),
            EnvironmentType environment = EnvironmentType());
  // check parameters
  /**
   * Execute a step in an episode.
   * @return Reward for the step.
   */
  double Step();

  /**
   * Execute an episode.
   * @return Return of the episode.
   */
  double Episode();

  /**
   * @return Total steps from beginning.
   */
  const size_t& TotalSteps() const { return totalSteps; }

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }
  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }

 private:

  //! Locally-stored hyper-parameters.
  TrainingConfig config;

  //! Locally-stored learning network.
  NetworkType learningNetwork;

  //! Locally-stored target network.
  NetworkType targetNetwork;

  //! Locally-stored updater.
  UpdaterType updater;

  //! Locally-stored behavior policy.
  PolicyType policy;

  //! Locally-stored experience method.
  ReplayType replayMethod;

  //! Locally-stored reinforcement learning task.
  EnvironmentType environment;

  //! Total steps from the beginning of the task.
  size_t totalSteps;

  //! Locally-stored current state of the agent.
  StateType state;

  //! Locally-stored flag indicating training mode or test mode.
  bool deterministic;
};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "policygrad_impl.hpp"
#endif
