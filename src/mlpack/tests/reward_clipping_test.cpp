/**
 * @file reward_clipping_test.hpp
 * @author Shashank Shekhar
 *
 * Test for the reward clipping wrapper for reinforcement learning environments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/continuous_mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobat.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>
#include <mlpack/methods/reinforcement_learning/environment/reward_clipping.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>
#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop_update.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(RewardClippingTest);

// Test checking that reward clipping works with vanilla update.
BOOST_AUTO_TEST_CASE(ClippedRewardTest)
{
  Pendulum task;
  RewardClipping<Pendulum> rewardClipping(task, -2.0, +2.0);
  
  RewardClipping<Pendulum>::State state = rewardClipping.InitialSample();
  RewardClipping<Pendulum>::Action action;
  action.action[0] = math::Random(-1.0, 1.0);
  double reward = rewardClipping.Sample(state, action);

  BOOST_REQUIRE(reward <= 2.0);
  BOOST_REQUIRE(reward >= -2.0);
}

//! Test DQN in Acrobat task.
BOOST_AUTO_TEST_CASE(RewardClippedAcrobatWithDQN)
{
    // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
      GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 64);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(64, 32);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(32, 3);

  // Set up the policy and replay method.
  GreedyPolicy<RewardClipping<Acrobat>> policy(1.0, 1000, 0.1);
  RandomReplay<RewardClipping<Acrobat>> replayMethod(20, 10000);

  // Set up Acrobat task and reward clipping wrapper
  Acrobat task;
  RewardClipping<Acrobat> rewardClipping(task, -2.0, +2.0);

  // Set up update rule
  AdamUpdate update;

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.99;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 400;

  // Set up DQN agent.
  QLearning<decltype(rewardClipping), decltype(model), AdamUpdate,
            decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy),
      std::move(replayMethod), std::move(update), std::move(rewardClipping));

  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > 1000)
    {
      Log::Debug << "Acrobat with DQN failed." << std::endl;
      converged = false;
      break;
    }

    /**
     * I am using a thresold of -380 to check convegence.
     */
    Log::Debug << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > -380.00)
    {
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 20; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }
  }

  BOOST_REQUIRE(converged);
}

BOOST_AUTO_TEST_SUITE_END();