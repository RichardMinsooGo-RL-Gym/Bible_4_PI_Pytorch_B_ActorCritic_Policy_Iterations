import sys
IN_COLAB = "google.colab" in sys.modules

import random
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.distributions import Normal

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        # set the hidden layers
        self.hidden = nn.Linear(self.state_size, 128)
        self.mu_layer = nn.Linear(128, self.action_size)
        self.log_std_layer = nn.Linear(128, self.action_size)
        
        self.mu_layer = initialize_uniformly(self.mu_layer)
        self.log_std_layer = initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden(state))
        
        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = dist.sample()
        
        return action, dist
    
class CriticV(nn.Module):
    def __init__(
        self, 
        state_size: int, 
    ):
        """Initialize."""
        super(CriticV, self).__init__()
        
        self.hidden = nn.Linear(state_size, 128)
        self.out = nn.Linear(128, 1)
        
        self.out = initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        value = F.relu(self.hidden(state))
        value = self.out(value)
        return value

class A2CAgent:
    """A2CAgent interacting with environment.
        
    Attributes:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
        self, 
        env: gym.Env,
        gamma: float, 
        entropy_weight: float,
    ):
        """Initialize."""
        # networks
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        self.env = env
        # hyperparameters
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # actor
        self.actor = Actor(state_size, action_size
                          ).to(self.device)
        self.critic = CriticV(state_size
                          ).to(self.device)
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4,
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3,
        )
        
        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        
        # mode: train / test
        self.is_test = False

    def get_action(self, state):
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action
        
        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
        
            self.transition = [state, log_prob]
        
        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def train_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_state = torch.FloatTensor(next_state).to(self.device)
        curr_Q = self.critic(state)
        next_Q = self.critic(next_state)
        expected_Q  = reward + self.gamma * next_Q * mask
        
        # train critic
        critic_loss = F.smooth_l1_loss(curr_Q, expected_Q.detach())
        
        # advantage = Q_t - V(s_t)
        advantage = (expected_Q - curr_Q).detach()  # not backpropagated
        actor_loss = -advantage * log_prob
        actor_loss += self.entropy_weight * -log_prob  # entropy maximization
        
        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()
    
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
env_name = "Pendulum-v0"
env = gym.make(env_name)
env = ActionNormalizer(env)

# parameters
gamma = 0.9
entropy_weight = 1e-2

max_episodes = 300

# train
agent = A2CAgent(
    env, 
    gamma, 
    entropy_weight,
)

if __name__ == "__main__":
    
    """Train the agent."""
    agent.is_test = False
    
    actor_losses  = []
    critic_losses = []
    scores        = []
    
    for episode in range(max_episodes):
        state = agent.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            # next_state, reward, done = agent.step(action)
            """Take an action and return the response of the env."""
            next_state, reward, done, _ = agent.env.step(action)
            agent.transition.extend([next_state, reward, done])           
            
            state = next_state
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
            actor_loss, critic_loss = agent.train_step()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)


