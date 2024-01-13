import gymnasium as gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import torch.nn as nn
from itertools import count
from features_extractor import MinigridFeaturesExtractor

import torch as T
from torch import optim
import torch.nn.functional as F
from collections import deque , namedtuple
from itertools import count
# set up matplotlib
from IPython import display

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class replay_memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(input_dims, 128) # First hidden layer
        self.fc2 = nn.Linear(128, 128)        # Second hidden layer
        self.fc3 = nn.Linear(128, n_actions)  # Output layer

    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=1)
        return action_probs


class Agent(object):
    def __init__(self, input_dims, n_actions,  device, env, label, LR=0.001, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005):
        self.policy_network = PolicyNetwork(input_dims, n_actions).to(device)
        self.target_network = PolicyNetwork(input_dims, n_actions).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.memory = replay_memory(100000)
        self.steps_done = 0
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.LR = LR
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.LR, amsgrad=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.env = env
        self.label = label
        self.feature_extractor = MinigridFeaturesExtractor(env.observation_space[self.label], features_dim=64)
        self.current_state = None


    def save_checkpoint(self):
        T.save(self.policy_network.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.policy_network.load_state_dict(T.load(self.chkpt_file))
        
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with T.no_grad():
                feature_vector = self.feature_extractor(state).view(-1, 64)
                action_probs = self.policy_network(feature_vector)
                return self.policy_network(action_probs).to(self.device).max(1)[1].view(1, 1)
        else:
            return T.tensor([self.env.action_space[self.label].sample()], device=self.device, dtype=T.long)

    def process_step(self, action, observation, reward, done):
        next_state = self.feature_extractor(observation)

        # Convert the necessary components to tensors
        action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float)
        new_state_tensor = torch.tensor(observation, device=self.device, dtype=torch.float).unsqueeze(0)
        next_states = feature_extractor(next_states_tensor)

        # Store the transition in memory
        self.memory.push(self.current_state, action_tensor, next_states, reward_tensor)

        # Update the current state
        self.current_state = next_states

        # Perform optimization if the memory is sufficiently populated
        if len(self.memory) > self.batch_size:
            self.optimize_model()

        # If the episode is done, reset the current state
        if done:
            self.current_state = None  # Or set to initial state based on your environment

        # Soft update the target network
        self.soft_update_target_network()
    
    def soft_update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            
 

        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=T.bool)
        non_final_next_states = T.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)



        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_network
        state_action_values = []
        for agent_state in state_batch:
            agent_state_action_values = self.policy_network(agent_state).gather(1, action_batch)
            state_action_values.append(agent_state_action_values)
        state_action_values = T.cat(state_action_values)
        # state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_network; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = T.zeros(self.batch_size, device=device)
        with T.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.loss_record.append(loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        T.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

class MultiAgentSystem:
    def __init__(self, env, num_agents, device):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        self.agents = [Agent(input_dims=64, n_actions=env.action_space[0].n, device=self.device, env=env, label=i) for i in range(num_agents)]
        self.episode_durations = []
        self.loss_record = []
        self.episode_rewards = []
        self.max_score = -math.inf
        self.min_score = math.inf
        self.avg_score = 0
    
    def collect_actions(self, states):
        actions = {}
        for agent_id, (agent, state) in enumerate(zip(self.agents, states)):
            action = agent.select_action(state)
            actions[agent_id] = action
        return actions
    
    def optimise_models(self):
        for agent in self.agents:
            agent.optimize_model()
    
    def train(self, num_episodes):
        pbar = tqdm(range(1,num_episodes))
        for i_episode in pbar:
            # Initialize the self.environment and get it's state
            states, info = self.env.reset()
            accumulated_reward = 0

            for t in count():

                actions = self.collect_actions(states)
                observation, reward, terminated, truncated, _ = self.env.step(actions)
                for agent in self.agents:
                    action = actions[agent.label]
                    accumulated_reward += reward[agent.label]
                    reward = reward[agent.label]
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = observation[agent.label]

                    states = next_state

                    agent.process_step(action, next_state, reward, done)

                    if done:
                        self.episode_rewards.append(accumulated_reward)
                        self.episode_durations.append(t + 1)
                        break


    
    # Run episode
    def run_episode(self,  num_episodes=1):
        total_reward = []
        for episode in range(num_episodes):
            episode_reward = 0
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            terminated = False
            while not terminated: 
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
            total_reward.append(episode_reward)

        return np.mean(total_reward)








