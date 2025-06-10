from .Agent import Agent, Transition
from collections import deque
from typing import Union
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import math, torch, random
import matplotlib.pyplot as plt
import pdb

class QLearning(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24, use_gpu : bool=False) -> None:
        super().__init__(state_dim, action_dim, hidden_dim, use_gpu)
        self.policy_network = self.build_network().to(self.device)
        self.buffer = deque([], maxlen=1000) # empty replay buffer with the 1000 most recent transitions
        self.agent_name = "qlearning"

        self.iteration = 0

    def eps_threshold(self) -> float: # epsilon threshold for e-greedy exploration
        eps_start, eps_end, eps_decay = 0.9, 0.05, 1000
        self.iteration += 1
        return eps_end + (eps_start - eps_end) * math.exp(-1 * self.iteration / eps_decay)

    def build_network(self) -> torch.nn.Module:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Construct and return a Multi-Layer Perceptron (MLP) representing the Q function. Recall that a Q function accepts
        ###     two arguments i.e., a state and action pair. For this implementation, your Q function will process an observation
        ###     of the state and produce an estimate of the expected, discounted reward for each available action as an output -- allowing you to 
        ###     select the prediction assosciated with either action.
        ###     2) Use a hidden layer dimension as specified by 'self.hidden_dim'.
        ###     3) Our solution implements a three layer MLP with ReLU activations on the first two hidden units.
        ###     But you are welcome to experiment with your own network definition!
        ###
        ### Please see the following docs for support:
        ###     nn.Sequential: https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        ###     nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ###     nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.action_dim))
        return model
        ###########################################################################
    
    def policy(self, state : Union[np.ndarray, torch.tensor], train : bool=False) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).unsqueeze(0).to(self.device)
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) If train == True, sample from the policy with e-greedy exploration with a decaying epsilon threshold. We've already
        ###     implemented a function you can use to call the exploration threshold at any instance in the iteration i.e., 'self.eps_treshold()'.
        ###     2) If train == False, sample the action with the highest Q value as predicted by your network.
        ###     HINT: An exemplar implementation will need to use torch.no_grad() in the solution.
        ###
        ### Please see the following docs for support:
        ###     random.random: https://docs.python.org/3/library/random.html#random.random
        ###     torch.randint: https://docs.pytorch.org/docs/stable/generated/torch.randint.html
        ###     torch.argmax: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
        ###     torch.no_grad(): https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
        if train:
            rand_num = random.random()
            if rand_num < self.eps_threshold():
                policy_action = torch.randint(self.action_dim, (1,))
            else:
                policy_action = torch.argmax(self.policy_network(state))
        else:
            with torch.no_grad():
                policy_action = torch.argmax(self.policy_network(state))
        return policy_action
        ###########################################################################
    
    def sample_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(self.buffer) > self.batch_size
        samples = random.sample(self.buffer, self.batch_size)

        states, actions, targets = [], [], []
        for i in range(self.batch_size):
            s, a, r, sp = samples[i].state.to(self.device), samples[i].action.item(), samples[i].reward.to(self.device), samples[i].next_state if samples[i].next_state is None else samples[i].next_state.to(self.device)
            states.append(s)
            actions.append(a)
            with torch.no_grad():
                targets.append(r if sp is None else r + self.gamma*torch.max(self.policy_network(sp)))
        
        return torch.cat(states), torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1), torch.cat(targets).unsqueeze(1)

    def train(self, env : gym.wrappers, num_episodes : int=300) -> None:
        ### WRITE YOUR CODE BELOW ###################################################
        ###     1) Implementing the training algorithm according to Algorithm 1 on page 5 in "Playing Atari with Deep Reinforcement Learning".
        ###     2) Importantly, only take a gradient step on your memory buffer if the buffer size exceeds the batch size hyperparameter. 
        ###     HINT: In our implementation, we used the AdamW optimizer.
        ###     HINT: Use the custom 'Transition' data structure to push observed (s, a, r, s') transitions to the memory buffer. Then, 
        ###     you can sample from the buffer simply by calling 'self.sample_buffer()'.
        ###     HINT: In our implementation, we clip the value of gradients to 100, which is optional.
        ###
        ### Please see the following docs for support:
        ###     torch.optim.AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        ###     torch.nn.MSELoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        ###     torch.nn.utils.clip_grad_value_: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=1e-3)
        rewards_over_time = []
        # plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Q-Learning Training Progress')

        for i in tqdm(range(num_episodes), desc="Training Q-Learning Agent"):
            # pdb.set_trace()
            s, _ = env.reset()
            s = torch.tensor(s)
            terminated_bool = False
            truncated_bool = False
            full_reward = 0
            while not terminated_bool and not truncated_bool:
                action = self.policy(s, True)
                s_next, reward, terminated_bool, truncated_bool, _ = env.step(action.item())
                full_reward += reward
                s_next = torch.tensor(s_next)
                if terminated_bool or truncated_bool:
                    transition = Transition(s, action, torch.tensor([reward]), None)
                else:
                    transition = Transition(s, action, torch.tensor([reward]), s_next)
                self.buffer.append(transition)
                if len(self.buffer) > self.batch_size:
                    s_sample, a_sample, target = self.sample_buffer()
                    s_sample = s_sample.reshape(self.batch_size, -1)
                    mse = torch.nn.MSELoss()
                    q_values = self.policy_network(s_sample)
                    q_selected = q_values.gather(1, a_sample)
                    loss = mse(q_selected, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
                    optimizer.step()
                s = s_next
            rewards_over_time.append(full_reward)

        # Update plot
        plt.plot(np.arange(len(rewards_over_time)), rewards_over_time)
        plt.savefig('qlearning_reward_over_time.png')
        
        ###########################################################################