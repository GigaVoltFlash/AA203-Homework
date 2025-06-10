from .Agent import Agent
from typing import List
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch, datetime
import pdb

class ReinforceAgent(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self.policy_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        self.agent_name = "reinforce"

    def build_network(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def policy(self, state: np.ndarray, train : bool=False) -> int:
        state = torch.tensor(state, dtype=torch.float)
        action_probs = self.policy_network(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        if train:
            log_prob = action_distribution.log_prob(action)
            return action, log_prob
        else:
            return action

    def train(self, env: gym.wrappers, num_episodes: int=2000) -> None:
        reward_history = []
        for episode in range(num_episodes):
            obs, info = env.reset(seed=1738)
            terminated, truncated = False, False

            log_probs = []
            rewards = []

            while not terminated and not truncated:
                action, log_prob = self.policy(obs, train=True)
                obs, reward, terminated, truncated, info = env.step(action.item())
                log_probs.append(log_prob)
                rewards.append(reward)

            self.learn(rewards, log_probs)
            total_reward = sum(rewards)
            reward_history.append(total_reward)
            print(f"Episode {episode+1}: Total Reward = {total_reward}")

        self.plot_rewards(reward_history)

    def learn(self, rewards: list, log_probs: list) -> None:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Implement the naive REINFORCE, REINFORCE with causality trick and REINFORCE with causality trick + a baseline to 'center' the returns.
        ###     2) After you've finished your implementation, please comment out all sections but the section you wish to evaluate for training.
        ###
        ### Please see the following docs for support:
        ###     torch.stack: https://docs.pytorch.org/docs/stable/generated/torch.stack.html

        # 1) Naive REINFORCE
        gamma_array = torch.pow(self.gamma, torch.arange(len(rewards)))
        log_probs_tensor = torch.stack(log_probs)
        rewards_tensor = torch.tensor(rewards)
        log_probs_sum = torch.sum(log_probs_tensor)
        rewards_sum = torch.sum(torch.mul(rewards_tensor, gamma_array))
        loss = -torch.mean(rewards_sum * log_probs_sum)
        
        # 2) REINFORCE with causality trick
        gamma_array = torch.pow(self.gamma, torch.arange(len(rewards)))
        log_probs_tensor = torch.stack(log_probs)
        rewards_tensor = torch.tensor(rewards)
        rewards_discounted = torch.mul(rewards_tensor, gamma_array)
        rewards_after_t = torch.flip(torch.cumsum(torch.flip(rewards_discounted, dims=[0]), dim=0), dims=[0])
        log_probs_reward_weighted = torch.mul(log_probs_tensor, rewards_after_t)
        cost_gradient = torch.sum(log_probs_reward_weighted)
        loss = -torch.mean(cost_gradient)

        # 3) REINFORCE with causality trick and baseline to "center" the returns
        gamma_array = torch.pow(self.gamma, torch.arange(len(rewards)))
        log_probs_tensor = torch.stack(log_probs)
        rewards_tensor = torch.tensor(rewards)
        rewards_discounted = torch.mul(rewards_tensor, gamma_array)
        mean_rewards_discounted = torch.mean(rewards_discounted)
        rewards_after_t = torch.flip(torch.cumsum(torch.flip(rewards_discounted, dims=[0]), dim=0), dims=[0])
        log_probs_reward_weighted = torch.mul(log_probs_tensor, rewards_after_t - mean_rewards_discounted)
        cost_gradient = torch.sum(log_probs_reward_weighted)
        loss = -torch.mean(cost_gradient)
        ###########################################################################

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    @staticmethod
    def plot_rewards(reward_history: List[int]) -> None:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure()
        plt.plot(reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Reward Curve')
        filename = f"reward_curve_{current_time}.png"
        plt.savefig(filename)
        plt.show()
        print(f"Saved reward curve as {filename}")