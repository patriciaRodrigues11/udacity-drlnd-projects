import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from model import DuelingQNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 4e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.6             # reliance of sampling on prioritization
BETA = 0.4              # reliance of importance sampling weight on priortization

device = torch.device('cuda')

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Initialize learning step for updating beta
        self.learn_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get prioritized subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, BETA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Choose action values according to local model
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, beta):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            beta (float): reliance of importance sampling weight on priortization
        """

        # Beta will reach 1 after 25,000 training steps (~325 episodes)
        b = min(1.0, beta + self.learn_step * (1.0 - beta) / 25000)
        self.learn_step += 1
        
        states, actions, rewards, next_states, dones, probabilities, indices = experiences

#         # Get max predicted actions (for next states) from local model
#         next_local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
#         # Evaluate the max predicted actions from the local model on the target model
#         # based on Double DQN
#         Q_targets_next_values = self.qnetwork_target(next_states).detach().gather(1, next_local_actions)
#         # Compute Q targets for current states
#         Q_targets = rewards + (gamma * Q_targets_next_values * (1 - dones))

#         # Get expected Q values from local
#         Q_expected = self.qnetwork_local(states).gather(1, actions)
        
                ## Double DQN
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        next_actions = self.qnetwork_local(next_states).argmax(-1, keepdim=True)
        Q_targets_next = self.qnetwork_target(next_states).gather(-1, next_actions)
        Q_targets = rewards + GAMMA * Q_targets_next * (1-dones)

        # Compute and update new priorities
        new_priorities = (abs(Q_expected - Q_targets) + 0.2).detach()
        self.memory.update_priority(new_priorities, indices)

        # Compute and apply importance sampling weights to TD Errors
        ISweights = (((1 / len(self.memory)) * (1 / probabilities)) ** b)
        max_ISweight = torch.max(ISweights)
        ISweights /= max_ISweight
        Q_targets *= ISweights
        Q_expected *= ISweights

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss = loss
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience objects."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): reliance of sampling on prioritization
        """
        self.action_size = action_size
        self.memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.alpha = alpha
        self.max_priority = 0
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority=10):
        """Add a new experience to memory."""
        # Assign priority of new experiences to max priority to insure they are played at least once
        if len(self.memory) > self.batch_size + 5:
            e = self.experience(state, action, reward, next_state, done, self.max_priority)
        else:
            e = self.experience(state, action, reward, next_state, done, int(priority) ** self.alpha)
        self.memory.add(e)

    def update_priority(self, new_priorities, indices):
        """Updates priority of experience after learning."""
        for new_priority, index in zip(new_priorities, indices):
            index = int(index.item())
            old_e = self.memory[index]
            new_p = new_priority.item() ** self.alpha
            new_e = self.experience(old_e.state, old_e.action, old_e.reward, old_e.next_state, old_e.done, new_p)
            self.memory.update(index, new_e)
            if new_p > self.max_priority:
                self.max_priority = new_p
    
    def sample(self):
        """Sample a batch of experiences from memory based on TD Error priority.
           Return indices of sampled experiences in order to update their
           priorities after learning from them.
        """
        experiences = []
        indices = []
        sub_array_size = self.memory.get_sum() / self.batch_size
        for i in range(self.batch_size):
            choice = np.random.uniform(sub_array_size * i, sub_array_size * (i + 1))
            e, index = self.memory.retrieve(1, choice)
            experiences.append(e)
            indices.append(index)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        probabilities = torch.from_numpy(np.vstack([e.priority / self.memory.get_sum() for e in experiences])).float().to(device)
        indices = torch.from_numpy(np.vstack([i for i in indices])).int().to(device)
        
        return states, actions, rewards, next_states, dones, probabilities, indices

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
class SumTree:
    """
    Leaf nodes hold experiences and intermediate nodes store experience priority sums.
    
    Adapted from: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    """
    
    def __init__(self, maxlen):
        """Initialize a SumTree object.
        Params
        ======
            maxlen (int): maximum size of replay buffer
        """
        self.sumList = np.zeros(maxlen*2)
        self.experiences = np.zeros(maxlen*2, dtype=object)
        self.maxlen = maxlen
        self.currentSize = 0
        # Set insertion marker for next item as first leaf
        self.tail = ((len(self.sumList)-1) // 2) + 1
        

    def add(self, experience):
        """Add experience to array and experience priority to sumList."""
        if self.tail == len(self.sumList):
            self.tail = ((len(self.sumList)-1) // 2) + 1
        self.experiences[self.tail] = experience
        old = self.sumList[self.tail]
        self.sumList[self.tail] = experience.priority
        if old == 0:
            change = experience.priority
            self.currentSize += 1
        else:
            change = experience.priority - old
        self.propagate(self.tail, change)
        
        self.tail += 1

    def propagate(self, index, change):
        """Updates sum tree to reflect change in priority of leaf."""
        parent = index // 2
        if parent == 0:
            return
        self.sumList[parent] += change
        self.propagate(parent, change)

    def get_sum(self):
        """Return total sum of priorities."""
        return self.sumList[1]

    def retrieve(self, start_index, num):
        """Return experience at index in which walking the array and summing the probabilities equals num."""
        # Return experience if we reach leaf node
        if self.left(start_index) > len(self.sumList) - 1:
            return self.experiences[start_index], start_index
        # If left sum is greater than num, we look in left subtree
        if self.sumList[self.left(start_index)] >= num:
            return self.retrieve(self.left(start_index), num)
        # If left sum is not greater than num, we subtract the left sum and look in right subtree
        return self.retrieve(self.right(start_index), num - self.sumList[self.left(start_index)])

    def update(self, index, experience):
        """Updates experience with new priority."""
        
        self.experiences[index] = experience
        old_e_priority = self.sumList[index]
        self.sumList[index] = experience.priority
        change = experience.priority - old_e_priority
        self.propagate(index, change)
        

    def left(self, index):
        return index * 2

    def right(self, index):
        return index * 2 + 1

    def __getitem__(self, index):
        return self.experiences[index]

    def __len__(self):
        return self.currentSize