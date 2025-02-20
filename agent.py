import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
import threading
import queue
import gymnasium as gym

# Behavioral Memory Network (BMN)
class BMN(nn.Module):
    def __init__(self, obs_dim, act_dim, embed_dim=64, n_heads=4, n_layers=2):
        super(BMN, self).__init__()
        self.obs_encoder = nn.Linear(obs_dim, embed_dim)
        self.act_encoder = nn.Linear(act_dim, embed_dim)
        self.reward_encoder = nn.Linear(1, embed_dim)
        self.transformer = nn.Transformer(embed_dim, n_heads, n_layers, batch_first=True)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, obs, act, rew):
        obs = obs.unsqueeze(1)  # [batch, 1, obs_dim]
        act = act.unsqueeze(1)  # [batch, 1, act_dim]
        rew = rew.unsqueeze(1)  # [batch, 1, 1]
        
        obs_embed = self.obs_encoder(obs)
        act_embed = self.act_encoder(act)
        rew_embed = self.reward_encoder(rew)
        combined = obs_embed + act_embed + rew_embed
        memory = self.transformer(combined, combined)
        return self.fc_out(memory[:, -1])

# Curiosity Module
class CuriosityModule(nn.Module):
    def __init__(self, visual_shape, act_dim, hidden_dim=128):
        super(CuriosityModule, self).__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2), nn.ReLU(),  # (16, 436, 498)
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU()  # (32, 216, 247)
        )
        with torch.no_grad():
            test_input = torch.zeros(1, 3, visual_shape[0], visual_shape[1])
            conv_out = self.visual_encoder(test_input)
            flat_size = conv_out.numel() // conv_out.shape[0]
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(flat_size, hidden_dim)
        self.act_encoder = nn.Linear(act_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, visual, act, next_visual):
        v_enc = self.linear(self.flatten(self.visual_encoder(visual)))
        a_enc = self.act_encoder(act)
        combined = torch.cat([v_enc, a_enc], dim=-1)
        pred = self.predictor(combined)
        next_v_enc = self.linear(self.flatten(self.visual_encoder(next_visual)))
        curiosity = torch.mean((pred - next_v_enc) ** 2)
        return curiosity

# Intention and Motor Policies
class IntentionPolicy(nn.Module):
    def __init__(self, embed_dim, n_intentions=4):
        super(IntentionPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, n_intentions), nn.Softmax(dim=-1)
        )

    def forward(self, embed):
        return self.net(embed)

class MotorPolicy(nn.Module):
    def __init__(self, embed_dim, intention_dim, act_dim=6):
        super(MotorPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + intention_dim, 128), nn.ReLU(),
            nn.Linear(128, act_dim * 2)  # Mean and log_std
        )

    def forward(self, embed, intention):
        x = torch.cat([embed, intention], dim=-1)
        out = self.net(x)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        dist = torch.distributions.Normal(mean, log_std.exp())
        action_raw = dist.sample()
        return action_raw, dist

# HGABL Agent
class HGABLAgent:
    def __init__(self):
        """Initialize the agent with environment, models, and threading."""
        self.env = gym.make('WhiteboardEnv-v0', render_mode="human").unwrapped
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions
        self.visual_dim = 128
        obs_dim = self.env.observation_space["numeric"].shape[0] + self.visual_dim  # 4 + 128
        act_dim = 6  # joint_select, pen_mode, pen_force (2), vision_force (2)
        visual_shape = (self.env.board_height_pix, self.env.board_width_pix)  # (875, 1000)
        self.n_intentions = 4

        # Models
        self.bmn = BMN(obs_dim, act_dim).to(self.device)
        self.curiosity = CuriosityModule(visual_shape, act_dim).to(self.device)
        self.intention_policy = IntentionPolicy(64, self.n_intentions).to(self.device)
        self.motor_policy = MotorPolicy(64, self.n_intentions).to(self.device)

        # Optimizers
        self.bmn_opt = optim.Adam(self.bmn.parameters(), lr=1e-4)
        self.curiosity_opt = optim.Adam(self.curiosity.parameters(), lr=1e-4)
        self.intention_opt = optim.Adam(self.intention_policy.parameters(), lr=1e-4)
        self.motor_opt = optim.Adam(self.motor_policy.parameters(), lr=1e-4)

        # Buffers
        self.memory = deque(maxlen=1000)
        self.behavior_embeddings = []

        # Threading for learning
        self.learn_queue = queue.Queue()
        self.learn_thread = threading.Thread(target=self._learn_thread)
        self.learn_thread.daemon = True
        self.learn_thread.start()

        self.step_count = 0

    def preprocess_visual(self, visual):
        """Preprocess visual input to a 128-dimensional embedding."""
        visual_array = np.transpose(visual, (2, 0, 1))  # [channels, height, width]
        visual_tensor = torch.FloatTensor(visual_array).unsqueeze(0).to(self.device) / 255.0
        return self.curiosity.linear(self.curiosity.flatten(self.curiosity.visual_encoder(visual_tensor)))

    def preprocess_raw_visual(self, visual):
        """Preprocess raw visual input for curiosity module."""
        visual_array = np.transpose(visual, (2, 0, 1))  # [channels, height, width]
        return torch.FloatTensor(visual_array).to(self.device) / 255.0

    def get_full_obs(self, obs):
        """Combine numeric and visual observations into a full observation tensor."""
        numeric = torch.FloatTensor(obs["numeric"]).unsqueeze(0).to(self.device)
        visual_embed = self.preprocess_visual(obs["image_array"])
        return torch.cat([numeric, visual_embed], dim=-1)

    def act(self, obs, behavior_embed):
        """Generate an action based on the current observation and behavior embedding."""
        with torch.no_grad():
            intention_probs = self.intention_policy(behavior_embed)
            action_raw, _ = self.motor_policy(behavior_embed, intention_probs)
        
        action = {
            "joint_select": int((action_raw[0, 0] > 0).item()),
            "pen_mode": int((action_raw[0, 1] > 0).item()),
            "pen_force": action_raw[0, 2:4].cpu().numpy().astype(np.float32),
            "vision_force": action_raw[0, 4:6].cpu().numpy().astype(np.float32)
        }
        return action, intention_probs

    def _learn_thread(self):
        """Background thread to process learning batches from the queue."""
        while True:
            try:
                batch = self.learn_queue.get(timeout=1.0)
                self.learn(batch)
            except queue.Empty:
                continue

    def learn(self, batch):
        """Update models based on a batch of experiences."""
        obs, act, rew, next_obs = zip(*batch)
        
        # Convert to tensors efficiently
        obs_full = torch.cat([self.get_full_obs(o) for o in obs]).to(self.device)
        obs_visual = torch.stack([self.preprocess_raw_visual(o["image_array"]) for o in obs]).to(self.device)
        act_array = np.array([
            np.concatenate([np.array([a["joint_select"], a["pen_mode"]]), a["pen_force"], a["vision_force"]])
            for a in act
        ])
        act_tensor = torch.FloatTensor(act_array).to(self.device)
        rew_tensor = torch.FloatTensor(np.array(rew)).unsqueeze(-1).to(self.device)
        next_visual = torch.stack([self.preprocess_raw_visual(o["image_array"]) for o in next_obs]).to(self.device)

        # Update BMN
        behavior_embed = self.bmn(obs_full, act_tensor, rew_tensor)
        self.bmn_opt.zero_grad()
        bmn_loss = torch.mean(behavior_embed ** 2)  # Regularization
        bmn_loss.backward()
        self.bmn_opt.step()
        behavior_embed = behavior_embed.detach()

        # Curiosity Reward
        curiosity_loss = self.curiosity(obs_visual, act_tensor, next_visual)
        self.curiosity_opt.zero_grad()
        curiosity_loss.backward()
        self.curiosity_opt.step()
        curiosity_reward = curiosity_loss.detach().cpu().numpy()

        # Human Feedback Amplification
        total_reward = rew_tensor + 0.1 * curiosity_reward

        # Policy Update
        intention_probs = self.intention_policy(behavior_embed)
        action_raw, motor_dist = self.motor_policy(behavior_embed, intention_probs)
        intention_loss = -total_reward.mean() * torch.log(intention_probs + 1e-6).mean()
        motor_loss = -total_reward.mean() * motor_dist.log_prob(act_tensor).mean()

        # Combined loss for efficiency
        total_loss = intention_loss + motor_loss

        self.intention_opt.zero_grad()
        self.motor_opt.zero_grad()
        total_loss.backward()
        self.intention_opt.step()
        self.motor_opt.step()

        if rew_tensor.abs().sum() > 0:
            self.behavior_embeddings.append((behavior_embed.detach(), total_reward.mean().item()))

    def run(self):
        """Main loop to run the agent."""
        obs, _ = self.env.reset()
        behavior_embed = torch.zeros(1, 64).to(self.device)
        clock = pygame.time.Clock()

        while True:
            # Handle human input
            reward = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        reward = 1
                    elif event.key == pygame.K_2:
                        reward = -1
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.env.human_pen_active = True
                        self.env.human_drawing = True
                        self.env.human_cursor_pos = np.array(event.pos) / self.env.pixels_per_cm
                    elif event.button == 3:
                        self.env.human_scrolling = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.env.human_drawing = False
                    elif event.button == 3:
                        self.env.human_scrolling = False
                elif event.type == pygame.MOUSEMOTION and self.env.human_pen_active:
                    self.env.human_cursor_pos = np.array(event.pos) / self.env.pixels_per_cm

            # Agent action
            action, intention_probs = self.act(obs, behavior_embed)
            next_obs, _, done, _, info = self.env.step(action)

            # Store experience
            self.memory.append((obs, action, reward, next_obs))

            # Queue learning periodically
            self.step_count += 1
            if self.step_count % 10 == 0 and len(self.memory) >= 32:
                indices = np.random.choice(len(self.memory), 32, replace=False)
                batch = [self.memory[i] for i in indices]
                self.learn_queue.put(batch)

            # Render and update
            self.env.render()
            obs = next_obs
            clock.tick(self.env.metadata["render_fps"])
    
    def save_state(self, file_path):
        """Save the agent's state to a file."""
        state = {
            'bmn': self.bmn.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            'intention_policy': self.intention_policy.state_dict(),
            'motor_policy': self.motor_policy.state_dict(),
            'bmn_opt': self.bmn_opt.state_dict(),
            'curiosity_opt': self.curiosity_opt.state_dict(),
            'intention_opt': self.intention_opt.state_dict(),
            'motor_opt': self.motor_opt.state_dict(),
            'memory': list(self.memory),
            'step_count': self.step_count,
        }
        torch.save(state, file_path)

    def load_state(self, file_path):
        """Load the agent's state from a file."""
        state = torch.load(file_path)
        self.bmn.load_state_dict(state['bmn'])
        self.curiosity.load_state_dict(state['curiosity'])
        self.intention_policy.load_state_dict(state['intention_policy'])
        self.motor_policy.load_state_dict(state['motor_policy'])
        self.bmn_opt.load_state_dict(state['bmn_opt'])
        self.curiosity_opt.load_state_dict(state['curiosity_opt'])
        self.intention_opt.load_state_dict(state['intention_opt'])
        self.motor_opt.load_state_dict(state['motor_opt'])
        self.memory = deque(state['memory'], maxlen=1000)
        self.step_count = state['step_count']