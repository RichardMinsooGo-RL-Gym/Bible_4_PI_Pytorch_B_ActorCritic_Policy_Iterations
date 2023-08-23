import sys
IN_COLAB = "google.colab" in sys.modules

from torch.distributions import Normal
import random
from typing import Dict, List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

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

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        
        return dict(
            obs=self.state_memory[indices],
            next_obs=self.next_state_memory[indices],
            acts=self.action_memory[indices],
            rews=self.reward_memory[indices],
            done=self.done_memory[indices],
        )

    def __len__(self) -> int:
        return self.size

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(nn.Module):
    def __init__(
        self, 
        state_size: int, 
        action_size: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """Initialization."""
        super(Actor, self).__init__()
        
        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # set the hidden layers
        self.hidden1 = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        
        # set log_std layer
        self.log_std_layer = nn.Linear(128, action_size)
        
        # set mean layer
        self.mu_layer = nn.Linear(128, action_size)
        self.mu_layer = init_layer_uniform(self.mu_layer)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        
        x = F.relu(self.hidden2(x))
        
        # get mean
        mu = self.mu_layer(x).tanh()
        
        # get std
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)
        
        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()
        
        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
class CriticQ(nn.Module):
    def __init__(
        self, 
        state_size: int, 
    ):
        """Initialize."""
        super(CriticQ, self).__init__()
        
        self.hidden1 = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        return value
    
class CriticV(nn.Module):
    def __init__(self, state_size: int):
        """Initialize."""
        super(CriticV, self).__init__()
        
        self.hidden1 = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        
        return value

class SACAgent:
    """SAC agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        vf (nn.Module): critic model to predict state values
        target_value_net (nn.Module): target critic model to predict state values
        value_optimizer (Optimizer): optimizer for training vf
        critic_1 (nn.Module): critic model to predict state-action values
        critic_2 (nn.Module): critic model to predict state-action values
        critic_1_optimizer (Optimizer): optimizer for training critic_1
        critic_2_optimizer (Optimizer): optimizer for training critic_2
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        policy_update_freq (int): policy update frequency
        device (torch.device): cpu / gpu
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optimizer (Optimizer): optimizer for alpha
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        policy_update_freq: int = 2,
    ):
        """
        Initialization.
        """
        self.env = env
        # networks
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        # hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # actor
        self.actor = Actor(self.state_size, self.action_size
                          ).to(self.device)
        
        # q function
        self.critic1 = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic2 = CriticQ(self.state_size + self.action_size).to(self.device)
        
        # v function
        self.value_net = CriticV(self.state_size).to(self.device)
        self.target_value_net = CriticV(self.state_size).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        # buffer
        self.memory = ReplayBuffer(
            self.state_size, memory_size, batch_size
        )
        
        # automatic entropy tuning
        self.target_entropy = -np.prod((self.action_size,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4,
        )
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0
        
        # mode: train / test
        self.is_test = False

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(torch.FloatTensor(state).to(self.device)
                                        )[0].detach().cpu().numpy()
        
        
        self.transition = [state, selected_action]
        
        return selected_action

    def train_step(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device     = self.device  # for shortening the following lines
        
        # sample from replay buffer
        samples    = self.memory.sample_batch()
        state      = torch.FloatTensor(samples["obs"]).to(device)
        action     = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward     = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        done       = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_actions, new_log_Pis = self.actor(state)
        
        # train alpha (dual problem)
        alpha_loss = (
            -self.log_alpha.exp() * (new_log_Pis + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        alpha = self.log_alpha.exp()  # used for the actor loss calculation
        
        # q function loss
        masks = 1 - done
        curr_Q1s = self.critic1(state, action)
        curr_Q2s = self.critic2(state, action)
        next_V_targs = self.target_value_net(next_state)
        expected_Qs = reward + self.gamma * next_V_targs * masks
        expected_Qs = expected_Qs.detach()

        # critic loss
        critic1_loss = F.mse_loss(curr_Q1s, expected_Qs)
        critic2_loss = F.mse_loss(curr_Q2s, expected_Qs)
        
        # v function loss
        curr_Vs = self.value_net(state)
        q_pred = torch.min(
            self.critic1(state, new_actions), self.critic2(state, new_actions)
        )
        expected_Vs = q_pred - alpha * new_log_Pis
        value_loss = F.mse_loss(curr_Vs, expected_Vs.detach())
        
        if self.total_step % self.policy_update_freq == 0:
            # actor loss
            expected_Ps = q_pred - curr_Vs.detach()
            actor_loss = (alpha * new_log_Pis - expected_Ps).mean()
            
            # train actor

            # update policy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # target update (vf)
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
            
        # train Q functions
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        critic_loss = critic1_loss + critic2_loss

        # train V function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return actor_loss.data, critic_loss.data, value_loss.data, alpha_loss.data

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
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

# parameters
memory_size = 2000
initial_random_steps = 5000

max_episodes = 300
batch_size = 32

# train
agent = SACAgent(
    env, 
    memory_size, 
    batch_size, 
    initial_random_steps=initial_random_steps,
)

if __name__ == "__main__":
    
    """Train the agent."""
    agent.is_test = False
    
    actor_losses  = []
    critic_losses = []
    scores        = []
    value_losses, alpha_losses = [], []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
        while not done:
            action = agent.get_action(state)
            # next_state, reward, done = agent.step(action)
            """Take an action and return the response of the env."""
            next_state, reward, done, _ = agent.env.step(action)
            agent.transition += [reward, next_state, done]
            agent.memory.store(*agent.transition)
            
            state = next_state
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
            # if training is ready
            if (len(agent.memory) >= agent.batch_size):
                losses = agent.train_step()
                actor_losses.append(losses[0])
                critic_losses.append(losses[1])
                value_losses.append(losses[2])
                alpha_losses.append(losses[3])

    
