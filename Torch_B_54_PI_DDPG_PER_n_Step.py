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

if IN_COLAB and not os.path.exists("segment_tree.py") and not os.path.exists("demo.pkl"):
    # download segment tree module
    !wget https://raw.githubusercontent.com/curt-park/rainbow-is-all-you-need/master/segment_tree.py
    # download demo.pkl
    !wget https://raw.githubusercontent.com/mrsyee/pg-is-all-you-need/master/demo.pkl
        
from segment_tree import MinSegmentTree, SumSegmentTree

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer:
    """A numpy replay buffer with demonstrations."""

    def __init__(
        self, 
        state_size: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 3, 
        gamma: float = 0.99,
    ):
        """Initialize."""
        self.state_memory = np.zeros([size, state_size], dtype=np.float32)
        self.action_memory = np.zeros([size], dtype=np.float32)
        self.reward_memory = np.zeros([size], dtype=np.float32)
        self.next_state_memory = np.zeros([size, state_size], dtype=np.float32)
        self.done_memory = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store the transition in buffer."""
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)
        
        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.state_memory[self.ptr] = obs
        self.action_memory[self.ptr] = act
        self.reward_memory[self.ptr] = rew
        self.next_state_memory[self.ptr] = next_obs
        self.done_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self, idxs: List[int] = None) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size
        
        if idxs is None:
            idxs = np.random.choice(
                len(self), size=self.batch_size, replace=False
            )
            
        return dict(obs=self.state_memory[idxs],
                    next_obs=self.next_state_memory[idxs],
                    acts=self.action_memory[idxs],
                    rews=self.reward_memory[idxs],
                    done=self.done_memory[idxs],
                    # for N-step learning
                    idxs=idxs,
                    )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self.gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size

# PrioritizedReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer with demonstrations.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        state_size: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6, 
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            state_size, size, batch_size, 
            n_step, gamma, 
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.state_memory[indices]
        next_obs = self.next_state_memory[indices]
        acts = self.action_memory[indices]
        rews = self.reward_memory[indices]
        done = self.done_memory[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

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

class DDPGfDAgent:
    """DDPGfDAgent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        n_step (int): the number of multi step
        use_n_step (bool): whether to use n_step memory
        prior_eps (float): guarantees every transitions can be sampled
        lambda1 (float): n-step return weight
        lambda2 (float): l2 regularization weight
        lambda3 (float): actor loss contribution of prior weight
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        # PER parameters
        alpha: float = 0.3,
        beta: float = 1.0,
        prior_eps: float = 1e-6,
        # N-step Learning
        n_step: int = 3,
        # loss parameters
        lambda1: float = 1.0,  # N-step return weight
        lambda2: float = 1e-4, # l2 regularization weight
        lambda3: float = 1.0,  # actor loss contribution of prior weight
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
        
        # loss parameters
        self.lambda1 = lambda1
        self.lambda3 = lambda3
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # actor
        self.actor = Actor(self.state_size, self.action_size
                          ).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size
                          ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticQ(self.state_size + self.action_size
                          ).to(self.device)
        self.critic_target = CriticQ(self.state_size + self.action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            self.state_size, memory_size, batch_size, alpha
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                self.state_size, 
                memory_size, 
                batch_size, 
                n_step=n_step,
                gamma=gamma
            )
        # noise
        self.exploration_noise = OUNoise(
            self.action_size,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4,
            weight_decay=lambda2,
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3,
            weight_decay=lambda2,
        )
        
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
                                        ).detach().cpu().numpy()
        
        # add noise for exploration during training
        if not self.is_test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state, selected_action]
        
        return selected_action

    def _get_critic_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return element-wise critic loss."""
        device     = self.device  # for shortening the following lines
        state      = torch.FloatTensor(samples["obs"]).to(device)
        action     = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward     = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        done       = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        masks = 1 - done
        
        next_P_targs = self.actor_target(next_state)
        curr_Qs = self.critic(state, action)
        next_Q_targs = self.critic_target(next_state, next_P_targs)
        expected_Qs  = reward + self.gamma * next_Q_targs * masks
        expected_Qs  = expected_Qs.to(device).detach()
        
        # train critic
        critic_loss_element_wise = (curr_Qs - expected_Qs).pow(2)

        return critic_loss_element_wise
    
    def train_step(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        
        samples = self.memory.sample_batch(self.beta)
        state      = torch.FloatTensor(samples["obs"]).to(device)
        action     = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        weights    = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(device)
        indices    = samples["indices"]
        
        # train critic
        # 1-step loss
        critic_loss_element_wise = self._get_critic_loss(samples, self.gamma)
        critic_loss = torch.mean(critic_loss_element_wise * weights)
        
        # n-step loss
        if self.use_n_step:
            samples_n = self.memory_n.sample_batch(indices)
            n_gamma = self.gamma ** self.n_step
            critic_loss_n_element_wise = self._get_critic_loss(
                samples_n, n_gamma
            )
            
            # to update loss and priorities
            critic_loss_element_wise += (
                critic_loss_n_element_wise * self.lambda1
            )
            critic_loss = torch.mean(critic_loss_element_wise * weights) 
        
        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # train actor
        actor_loss_element_wise = -self.critic(state, self.actor(state))
        actor_loss = torch.mean(actor_loss_element_wise * weights)
        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # target update
        self._target_soft_update()
        
        # PER: update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.prior_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        self.memory.update_priorities(indices, new_priorities)
        
        return actor_loss.data, critic_loss.data
        
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
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
ou_noise_theta = 1.0
ou_noise_sigma = 0.1
initial_random_steps = 5000
n_step = 3

max_episodes = 300
batch_size = 32

# train
agent = DDPGfDAgent(
    env, 
    memory_size, 
    batch_size, 
    ou_noise_theta,
    ou_noise_sigma,
    n_step=n_step,
    initial_random_steps=initial_random_steps,
)

if __name__ == "__main__":
    
    """Train the agent."""
    agent.is_test = False
    
    actor_losses  = []
    critic_losses = []
    scores        = []
    frame_idx = 0
    num_frames= 100000
    
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
            
            # N-step transition
            transition = agent.transition
            if agent.use_n_step:
                transition =agent.memory_n.store(*agent.transition)

            # add a single step transition
            if transition:
                agent.memory.store(*transition)
            
            state = next_state
            episode_reward += reward

            frame_idx += 1
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            agent.beta = agent.beta + fraction * (1.0 - agent.beta)
            
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

    
