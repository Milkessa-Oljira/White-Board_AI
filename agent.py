import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Convert state and next_state to numpy arrays if they're not already
        state_array = {
            'image_array': np.array(state['image_array']),
            'numeric': np.array(state['numeric'])
        }
        next_state_array = {
            'image_array': np.array(next_state['image_array']),
            'numeric': np.array(next_state['numeric'])
        }
        self.buffer.append((state_array, action, reward, next_state_array, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        
        # Separate the components
        states = [s[0] for s in samples]
        actions = [s[1] for s in samples]
        rewards = [s[2] for s in samples]
        next_states = [s[3] for s in samples]
        dones = [s[4] for s in samples]
        
        # Combine the dictionaries
        state_batch = {
            'image_array': np.stack([s['image_array'] for s in states]),
            'numeric': np.stack([s['numeric'] for s in states])
        }
        next_state_batch = {
            'image_array': np.stack([s['image_array'] for s in next_states]),
            'numeric': np.stack([s['numeric'] for s in next_states])
        }
        
        return (
            state_batch,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            next_state_batch,
            np.array(dones, dtype=np.float32)
        )
    
    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)

class CNNEncoder(nn.Module):
    def __init__(self, height, width):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output size
        h = (((height - 8)//4 + 1) - 4)//2 + 1 - 3 + 1
        w = (((width - 8)//4 + 1) - 4)//2 + 1 - 3 + 1
        self.fc = nn.Linear(64 * h * w, 512)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 255.0  # Normalize and reshape for CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std for all continuous actions
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Additional layers for discrete actions
        self.joint_select = nn.Linear(hidden_dim, 2)  # 2 options
        self.pen_mode = nn.Linear(hidden_dim, 3)      # 3 options
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Continuous actions
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        # Discrete actions
        joint_prob = F.softmax(self.joint_select(x), dim=-1)
        pen_prob = F.softmax(self.pen_mode(x), dim=-1)
        
        return mean, log_std, joint_prob, pen_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2

class SACAgent:
    def __init__(self, env_dims, device="cpu", alpha=0.2):
        self.device = torch.device(device)
        self.alpha = alpha  # entropy coefficient
        
        # Initialize dimensions
        self.image_height = env_dims['image_height']
        self.image_width = env_dims['image_width']
        self.numeric_dim = env_dims['numeric_dim']
        self.action_dim = 6  # 2 force vectors (2D each) + 2 discrete actions
        
        # Initialize networks
        self.cnn = CNNEncoder(self.image_height, self.image_width).to(device)
        state_dim = 512 + self.numeric_dim  # CNN output + numeric input
        
        self.actor = Actor(state_dim, 4).to(device)  # 4 for continuous actions
        self.critic = Critic(state_dim, self.action_dim).to(device)
        self.critic_target = Critic(state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training parameters
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            # Process state
            image = torch.FloatTensor(state['image_array']).unsqueeze(0).to(self.device)
            numeric = torch.FloatTensor(state['numeric']).unsqueeze(0).to(self.device)
            
            # Get CNN features
            image_features = self.cnn(image)
            state_features = torch.cat([image_features, numeric], dim=1)
            
            # Get action distributions
            mean, log_std, joint_prob, pen_prob = self.actor(state_features)
            std = log_std.exp()
            
            if evaluate:
                continuous_action = mean
                joint_select = torch.argmax(joint_prob, dim=1)
                pen_mode = torch.argmax(pen_prob, dim=1)
            else:
                # Sample continuous actions
                normal = Normal(mean, std)
                continuous_action = normal.rsample()
                
                # Sample discrete actions
                joint_select = torch.multinomial(joint_prob, 1)
                pen_mode = torch.multinomial(pen_prob, 1)
            
            # Combine actions
            continuous_action = torch.tanh(continuous_action)
            action_array = np.zeros(6)
            action_array[0] = joint_select.cpu().numpy()[0]
            action_array[1] = pen_mode.cpu().numpy()[0]
            action_array[2:6] = continuous_action.cpu().numpy()[0]
            
            return action_array
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_tensors = {
            'image_array': torch.FloatTensor(state_batch['image_array']).to(self.device),
            'numeric': torch.FloatTensor(state_batch['numeric']).to(self.device)
        }
        next_state_tensors = {
            'image_array': torch.FloatTensor(next_state_batch['image_array']).to(self.device),
            'numeric': torch.FloatTensor(next_state_batch['numeric']).to(self.device)
        }
        action_tensors = torch.FloatTensor(action_batch).to(self.device)
        reward_tensors = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_tensors = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Process state features
        current_features = torch.cat([
            self.cnn(state_tensors['image_array']),
            state_tensors['numeric']
        ], dim=1)
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(current_features, action_tensors)

        # Process next state features
        with torch.no_grad():
            next_features = torch.cat([
                self.cnn(next_state_tensors['image_array']),
                next_state_tensors['numeric']
            ], dim=1)
            
            next_mean, next_log_std, next_joint_prob, next_pen_prob = self.actor(next_features)
            next_std = next_log_std.exp()
            next_normal = Normal(next_mean, next_std)
            next_action = torch.tanh(next_normal.rsample())
            
            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_features, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_tensors + (1 - done_tensors) * self.gamma * target_Q

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        mean, log_std, joint_prob, pen_prob = self.actor(current_features)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = torch.tanh(normal.rsample())
        
        Q1, Q2 = self.critic(current_features, action)
        Q = torch.min(Q1, Q2)
        
        actor_loss = -Q.mean()
        
        # Add entropy term
        entropy = normal.entropy().mean()
        actor_loss += -self.alpha * entropy

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'cnn_state_dict': self.cnn.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])