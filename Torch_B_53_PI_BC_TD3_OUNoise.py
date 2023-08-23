import sys
IN_COLAB = "google.colab" in sys.modules

import copy
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

if IN_COLAB and not os.path.exists("demo.pkl"):
    # download demo.pkl
    !wget https://raw.githubusercontent.com/mrsyee/pg-is-all-you-need/master/demo.pkl

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        state_size: int, 
        size: int, 
        batch_size: int = 32, 
    ):
        """Initialize."""
        self.state_memory = np.zeros([size, state_size], dtype=np.float32)
        self.action_memory = np.zeros([size], dtype=np.float32)
        self.reward_memory = np.zeros([size], dtype=np.float32)
        self.next_state_memory = np.zeros([size, state_size], dtype=np.float32)
        self.done_memory = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        """Store the transition in buffer."""
        self.state_memory[self.ptr] = obs
        self.action_memory[self.ptr] = act
        self.reward_memory[self.ptr] = rew
        self.next_state_memory[self.ptr] = next_obs
        self.done_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def extend(
        self, 
        transitions: List[Tuple],
    ):
        """Store the multi transitions in buffer."""
        for transition in transitions:
            self.store(*transition)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        
        return dict(obs=self.state_memory[idxs],
                    next_obs=self.next_state_memory[idxs],
                    acts=self.action_memory[idxs],
                    rews=self.reward_memory[idxs],
                    done=self.done_memory[idxs],
                    )

    def __len__(self) -> int:
        return self.size

# PrioritizedReplayBuffer
# Not Defined

class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self, 
        size: int, 
        mu: float = 0.0, 
        theta: float = 0.15, 
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    def __init__(
        self, 
        state_size: int, 
        action_size: int,
        init_w: float = 3e-3,
    ):
        """Initialization."""
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        # set the hidden layers
        self.hidden1 = nn.Linear(self.state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        
        self.out = nn.Linear(128, self.action_size)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()
        
        return action
    
class CriticQ(nn.Module):
    def __init__(
        self, 
        state_size: int, 
        init_w: float = 3e-3,
    ):
        """Initialize."""
        super(CriticQ, self).__init__()
        
        self.hidden1 = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        return value

class TD3Agent:
    """TD3Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        actor1 (nn.Module): target actor model to select actions
        actor2 (nn.Module): target actor model to select actions
        actor_target1 (nn.Module): actor model to predict next actions
        actor_target2 (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic1 (nn.Module): critic model to predict state values
        critic2 (nn.Module): critic model to predict state values
        critic1_target (nn.Module): target critic model to predict state values
        critic2_target (nn.Module): target critic model to predict state values        
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        demo_memory (ReplayBuffer): replay memory for demonstration
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        lambda1 (float): weight for policy gradient loss
        lambda2 (float): weight for behavior cloning loss
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        policy_update_freq (int): update actor every time critic updates this times
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        demo_batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        demo: list,
        gamma: float = 0.99,
        tau: float = 5e-3,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        initial_random_steps: int = 1e4,
        # loss parameters
        lambda1: float = 1e-3,
        lambda2: int = 1.0,
        policy_update_freq: int = 2,
    ):
        """
        Initialization.
        """
        self.env = env
        # network parameters
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        # hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        
        # loss parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2 / demo_batch_size
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # actor
        self.actor = Actor(self.state_size, self.action_size
                          ).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size
                          ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1 = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic2 = CriticQ(self.state_size + self.action_size).to(self.device)
        
        self.critic1_target = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # buffer
        self.memory = ReplayBuffer(
             self.state_size, memory_size, batch_size
        )
        
        # demo buffer
        self.demo_memory = ReplayBuffer(self.state_size, len(demo), demo_batch_size)
        self.demo_memory.extend(demo) 
        
        # noise
        self.exploration_noise = OUNoise(
            self.action_size,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )
        self.target_policy_noise = OUNoise(
            self.action_size,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )
        self.target_policy_noise_clip = target_policy_noise_clip
        
        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4,
        )
        self.critic_optimizer = optim.Adam(critic_parameters, lr=1e-3,
        )
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0
        
        # update step for actor
        self.update_step = 0
        
        # mode: train / test
        self.is_test = False

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(torch.FloatTensor(state).to(self.device)
                                        ).detach().cpu().numpy()
        
        # add noise for exploration during training
        if not self.is_test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state, selected_action]
        
        return selected_action

    def train_step(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device     = self.device  # for shortening the following lines
        
        '''
        sample from replay buffer
        '''
        samples    = self.memory.sample_batch()
        state      = torch.FloatTensor(samples["obs"]).to(device)
        action     = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward     = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        done       = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
         
        '''
        sample from demo buffer
        '''
        d_samples    = self.demo_memory.sample_batch()
        d_state      = torch.FloatTensor(d_samples["obs"]).to(device)
        d_next_state = torch.FloatTensor(d_samples["next_obs"]).to(device)
        d_action     = torch.FloatTensor(d_samples["acts"].reshape(-1, 1)).to(device)
        d_reward     = torch.FloatTensor(d_samples["rews"].reshape(-1, 1)).to(device)
        d_done       = torch.FloatTensor(d_samples["done"].reshape(-1, 1)).to(device)
        
        '''
        Critic loss
        '''
        masks = 1 - done
        
        # get actions with noise
        noise = torch.FloatTensor(self.target_policy_noise.sample()).to(device)
        clipped_noise = torch.clamp(
            noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        
        next_P_targs = (self.actor_target(next_state) + clipped_noise).clamp(-1.0, 1.0)

        curr_Q1s = self.critic1(state, action)
        curr_Q2s = self.critic2(state, action)
        # min (Q_1', Q_2')
        next_Q1_targs = self.critic1_target(next_state, next_P_targs)
        next_Q2_targs = self.critic2_target(next_state, next_P_targs)
        next_values = torch.min(next_Q1_targs, next_Q2_targs)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        expected_Qs = reward + self.gamma * next_values * masks
        expected_Qs = expected_Qs.detach()

        # critic loss
        critic1_loss = F.mse_loss(curr_Q1s, expected_Qs)
        critic2_loss = F.mse_loss(curr_Q2s, expected_Qs)
        
        # train critic
        critic_loss = critic1_loss + critic2_loss
        
        '''
        Reset Critic Gradient
        '''
        self.critic_optimizer.zero_grad()
        
        '''
        Update Critic -Q,one-step
        '''
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.total_step % self.policy_update_freq == 0:
            '''
            PG Loss
            '''
            pg_loss = -self.critic1(state, self.actor(state)).mean()
            
            '''
            BC Loss
            '''
            pred_action = self.actor(d_state)       
            qf_mask = torch.gt(
                self.critic1(d_state, d_action),
                self.critic1(d_state, pred_action),
            ).to(device)
            qf_mask = qf_mask.float()
            n_qf_mask = int(qf_mask.sum().item())

            if n_qf_mask == 0:
                bc_loss = torch.zeros(1, device=device)
            else:
                bc_loss = (
                    torch.mul(pred_action, qf_mask) - torch.mul(d_action, qf_mask)
                ).pow(2).sum() / n_qf_mask
            
            '''
            Actor Loss
            '''
            actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss
            
            # train actor

            '''
            Reset Actor Gradient
            '''
            self.actor_optimizer.zero_grad()
            
            '''
            Update Actor,n-tep
            '''
            actor_loss.backward()
            self.actor_optimizer.step()

            '''
            target update
            '''
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        return actor_loss.data, critic_loss.data
        
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic1_target.parameters(), self.critic1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic2_target.parameters(), self.critic2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
    
class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

# environment
env_name = "Pendulum-v1"
env = gym.make(env_name)
env = ActionNormalizer(env)

import pickle

# load demo on replay memory
demo_path = "demo.pkl"
with open(demo_path, "rb") as f:
    demo = pickle.load(f)

'''
Hyper Parameters
'''

memory_size = 2000
demo_batch_size = 128
ou_noise_theta = 1.0
ou_noise_sigma = 0.1
initial_random_steps = 5000

max_episodes = 300
batch_size = 32

'''
Agent Define
'''

agent = TD3Agent(
    env, 
    memory_size, 
    batch_size, 
    demo_batch_size,
    ou_noise_theta,
    ou_noise_sigma,
    demo,
    initial_random_steps=initial_random_steps,
)

if __name__ == "__main__":
    
    """Train the agent."""
    agent.is_test = False
    
    actor_losses  = []
    critic_losses = []
    scores        = []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
        while not done:
            '''
            Get Action
            '''
            action = agent.get_action(state)
            
            '''
            Execute Action and Observe
            '''
            next_state, reward, done, _ = agent.env.step(action)
            
            '''
            Store Transitions
            '''          
            agent.transition += [reward, next_state, done]
            agent.memory.store(*agent.transition)
            
            '''
            Update state
            '''     
            state = next_state
            episode_reward += reward

            # if episode ends
            if done:
                state = agent.env.reset()
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
            # if training is ready
            if (len(agent.memory) >= agent.batch_size):
                actor_loss, critic_loss = agent.train_step()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

    
