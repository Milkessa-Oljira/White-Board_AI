#!/usr/bin/env python
"""
Full implementation of a multi-agent RL scenario with two joints (pen and vision)
on a whiteboard environment. The AI uses DDPG to select actions from a continuous
action space that includes a joint selector, pen mode (when applicable), and force
components (DX, DY). The simulation includes realistic physics (friction, muscle force
limits, boundary constraints), a partial (focused) vision modality, and reward signals
from both the environment and (via placeholder) human feedback.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pygame

# =====================
# Environment Constants
# =====================

# Board dimensions in cm:
BOARD_WIDTH_CM = 200
BOARD_HEIGHT_CM = 125

# Resolution: 
PIXELS_PER_CM = 5
BOARD_WIDTH_PIX = BOARD_WIDTH_CM * PIXELS_PER_CM  # 1000 pixels
BOARD_HEIGHT_PIX = BOARD_HEIGHT_CM * PIXELS_PER_CM  # 625 pixels

# Focus window dimensions (clear region): 
FOCUS_SIZE_CM = 38
FOCUS_SIZE_PIX = FOCUS_SIZE_CM * PIXELS_PER_CM

# Pen joint parameters
PEN_MASS = 0.05  # in kg
PEN_FRICTION_COEFF = 0.8  # friction coefficient for pen movement
SCROLL_EXTRA_FRICTION = 1.2  # extra friction factor when in scrolling mode
MAX_FORCE = 5.0  # maximum muscle-like force (N)

# Vision joint parameters
EYE_MASS = 0.02  # in kg
EYE_FRICTION_COEFF = 0.7  # friction for vision joint (without extra friction)
# The reaction force is proportional to the weight of the eye
# (We simulate this via the mass and friction)

# Boundary limits in cm
MIN_X, MAX_X = 0, BOARD_WIDTH_CM
MIN_Y, MAX_Y = 0, BOARD_HEIGHT_CM

# Pen modes (only valid if moving pen)
PEN_MODE_MOVE = 0  # moving without writing
PEN_MODE_WRITE = 1  # writing mode
PEN_MODE_SCROLL = 2  # scrolling mode

# a rectangle where text is displayed by the environment
FORBIDDEN_REGION = [10, 15, 5, 10]  # [x_min, x_max, y_min, y_max] in cm

# =====================
# Environment Class
# =====================

class WhiteboardEnv:
    def __init__(self):
        # Board dimensions in cm and pixels
        self.board_width_cm = BOARD_WIDTH_CM 
        self.board_height_cm = BOARD_HEIGHT_CM
        self.pixels_per_cm = PIXELS_PER_CM  
        self.board_width_pix = self.board_width_cm * self.pixels_per_cm
        self.board_height_pix = self.board_height_cm * self.pixels_per_cm 

        self.focus_size_cm = FOCUS_SIZE_CM
        self.focus_size_pix = self.focus_size_cm * self.pixels_per_cm

        self.dt = 1.0 / 60.0

        # Create a blank white board (BGR image)
        self.board_image = pygame.Surface((self.board_width_pix, self.board_height_pix))
        self.board_image.fill((255, 255, 255))
        # Board vertical scroll offset (in pixels; for simplicity we only track vertical scrolling)
        self.board_offset_y = 0

        # ----- AI Joint States -----
        # AI pen (controlled by DDPG) state: position (cm) and velocity (cm/s)
        self.ai_pen_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.ai_pen_vel = np.zeros(2, dtype=np.float32)
        # Vision joint state for AI (controls the focus area)
        self.vision_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.vision_vel = np.zeros(2, dtype=np.float32)
        # Last applied forces (for tactile feedback)
        self.last_ai_pen_force = np.zeros(2, dtype=np.float32)
        self.last_vision_force = np.zeros(2, dtype=np.float32)

        # ----- Human Joint States -----
        # Human pen state (controlled via mouse)
        self.human_pen_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.human_pen_vel = np.zeros(2, dtype=np.float32)
        # Human input flags and cursor (in cm)
        self.human_pen_active = False   # True if the mouse is within board boundaries
        self.human_drawing = False      # True when left mouse button is held down
        self.human_scrolling = False    # True if human scrolling is active (e.g. via a key)
        self.human_cursor_pos = None    # (x, y) position in cm of the current cursor

        # Continuous environment: never done on its own.
        self.done = False

    def reset(self):
        # Reset the board (clear drawings) and all joint states.
        self.board_image = pygame.Surface((self.board_width_pix, self.board_height_pix))
        self.board_image.fill((255, 255, 255))
        self.board_offset_y = 0

        self.ai_pen_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.ai_pen_vel = np.zeros(2, dtype=np.float32)
        self.vision_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.vision_vel = np.zeros(2, dtype=np.float32)
        self.last_ai_pen_force = np.zeros(2, dtype=np.float32)
        self.last_vision_force = np.zeros(2, dtype=np.float32)

        self.human_pen_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.human_pen_vel = np.zeros(2, dtype=np.float32)
        self.human_pen_active = False
        self.human_drawing = False
        self.human_scrolling = False
        self.human_cursor_pos = None

        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns a dictionary with:
          - 'image': the composite board image with a clear focus (the 75x75 cm window)
                     and a blurred periphery.
          - 'numeric': an 8-dimensional vector: [ai_pen_pos (x,y), vision_pos (x,y),
                        pen tactile feedback, vision tactile feedback], with positions normalized.
        """
        composite_img = self._create_composite_image()

        # Tactile feedback is approximated as friction forces proportional to velocity.
        ai_pen_force = -0.8 * self.ai_pen_vel   # using a friction coefficient of 0.8
        vision_force = -0.7 * self.vision_vel     # using a friction coefficient of 0.7

        numeric_state = np.concatenate([self.ai_pen_pos, self.vision_pos, ai_pen_force, vision_force])
        # Normalize positions by board dimensions
        numeric_state[:4] = numeric_state[:4] / np.array([self.board_width_cm, self.board_height_cm,
                                                           self.board_width_cm, self.board_height_cm])
        # Normalize forces by MAX_FORCE (assumed to be 5.0 N)
        numeric_state[4:] = numeric_state[4:] / 5.0

        return {'image': composite_img, 'numeric': numeric_state}

    def _create_composite_image(self):
        """
        Create a composite image where the entire board is blurred (using smoothscale as a simple blur)
        except for a clear focus window (centered on the AI's vision position).
        """
        # Create a blurred version by downscaling then upscaling.
        small_size = (self.board_width_pix // 10, self.board_height_pix // 10)
        small = pygame.transform.smoothscale(self.board_image, small_size)
        blurred = pygame.transform.smoothscale(small, (self.board_width_pix, self.board_height_pix))
        
        # Determine focus window in pixels.
        center_x = int(self.vision_pos[0] * self.pixels_per_cm)
        center_y = int(self.vision_pos[1] * self.pixels_per_cm)
        half_focus = self.focus_size_pix // 2
        x1 = max(center_x - half_focus, 0)
        y1 = max(center_y - half_focus, 0)
        width = min(self.focus_size_pix, self.board_width_pix - x1)
        height = min(self.focus_size_pix, self.board_height_pix - y1)
        focus_rect = pygame.Rect(x1, y1, width, height)
        
        # Copy the non-blurred focus region from the original board.
        focus_surface = self.board_image.subsurface(focus_rect).copy()
        blurred.blit(focus_surface, (x1, y1))
        return blurred

    def step(self, action):
        """
        Processes an action (a 4-D continuous vector):
          action[0]: Joint selection. If < 0.5, control the AI pen; if >= 0.5, control the vision joint.
          action[1]: For AI pen control, determines the mode:
                     - < 1/3: move (without writing),
                     - [1/3, 2/3): writing,
                     - >= 2/3: scrolling.
          action[2]: Force in x direction (in N), clipped to [-5, 5].
          action[3]: Force in y direction (in N), clipped to [-5, 5].
        Only one joint is updated per time step.
        """
        # Update human input (pen position, drawing, scrolling)
        self._update_human_pen()
        self._update_human_scrolling()

        # --- New Action Decoding for 6-Dim Action Space ---
        # action[0]: joint selection scalar.
        # action[1]: pen mode scalar (used if pen is selected).
        # action[2:4]: pen force (x, y).
        # action[4:6]: vision force (x, y).
        joint_select = action[0]
        if joint_select < 0.5:
            # Control the AI pen.
            mode_raw = action[1]
            if mode_raw < 1/3:
                mode = 0   # move (no writing)
            elif mode_raw < 2/3:
                mode = 1   # writing mode
            else:
                mode = 2   # scrolling mode
            pen_force = np.clip(np.array(action[2:4]), -5.0, 5.0)
            vision_force = np.zeros(2, dtype=np.float32)
        else:
            # Control the vision joint.
            mode = None  # mode is ignored for vision.
            vision_force = np.clip(np.array(action[4:6]), -5.0, 5.0)
            pen_force = np.zeros(2, dtype=np.float32)

        self.last_ai_pen_force = pen_force
        self.last_vision_force = vision_force

        # Update AI joints.
        self._update_joint('ai_pen', pen_force, mode)
        self._update_joint('vision', vision_force)

        # Compute reward (for example, based on writing location, etc.)
        reward = self._compute_reward(mode)

        # Continuous environment: never terminates on its own.
        self.done = False

        next_state = self.get_state()
        info = {'mode': mode}
        return next_state, reward, self.done, info

    def _update_joint(self, joint, applied_force, mode=None):
        """
        Updates the specified joint’s state using Euler integration.
          - For the AI pen: mass=0.05 kg, friction=0.8.
            In scrolling mode, an extra friction of 1.2 is added.
          - For the vision joint: mass=0.02 kg, friction=0.7.
        """
        if joint == 'ai_pen':
            mass = 0.05
            friction = 0.8
            vel = self.ai_pen_vel
            pos = self.ai_pen_pos
        elif joint == 'vision':
            mass = 0.02
            friction = 0.7
            vel = self.vision_vel
            pos = self.vision_pos
        else:
            raise ValueError("Invalid joint name.")

        extra_friction = 0
        if joint == 'ai_pen' and mode == 2:  # scrolling mode
            extra_friction = 1.2

        total_friction = friction + extra_friction
        acceleration = (applied_force - total_friction * vel) / mass
        new_vel = vel + acceleration * self.dt
        new_pos = pos + new_vel * self.dt

        # Enforce board boundaries.
        new_pos[0] = np.clip(new_pos[0], 0, self.board_width_cm)
        new_pos[1] = np.clip(new_pos[1], 0, self.board_height_cm)
        if new_pos[0] in [0, self.board_width_cm]:
            new_vel[0] = 0
        if new_pos[1] in [0, self.board_height_cm]:
            new_vel[1] = 0

        if joint == 'ai_pen':
            self.ai_pen_vel = new_vel
            self.ai_pen_pos = new_pos
            if mode == 1:
                self._draw_at_ai_pen()
        else:
            self.vision_vel = new_vel
            self.vision_pos = new_pos

        # If AI pen is in scrolling mode, update board offset (simulate scrolling).
        if joint == 'ai_pen' and mode == 2:
            delta_offset = int(new_vel[1] * self.pixels_per_cm * self.dt)
            self.board_offset_y += delta_offset

    def _draw_at_ai_pen(self):
        x = int(self.ai_pen_pos[0] * self.pixels_per_cm)
        y = int(self.ai_pen_pos[1] * self.pixels_per_cm)
        pygame.draw.circle(self.board_image, (0, 0, 0), (x, y), 3)

    def _compute_reward(self, mode):
        """
        Computes a reward signal. For example, if the AI pen is in writing mode and writes within a
        forbidden region (e.g., x between 100 and 200 cm and y between 50 and 100 cm), a negative reward
        is issued; otherwise, a small positive reward is given.
        """
        reward = 0.0
        # if mode == 1:
        #     x, y = self.ai_pen_pos
        #     if 100 <= x <= 200 and 50 <= y <= 100:
        #         reward = -1.0
        #     else:
        #         reward = 0.5
        return reward

    def _update_human_pen(self):
        """
        Updates the human pen position based on the mouse cursor.
        If the cursor is within the board and drawing is active, the pen position is updated
        and a continuous blue stroke is drawn.
        """
        if self.human_pen_active and self.human_cursor_pos is not None:
            target = np.array(self.human_cursor_pos, dtype=np.float32)
            self.human_pen_pos = target.copy()
            if self.human_drawing:
                self._draw_at_human_pen()
            else:
                # If not drawing, reset the last drawn position.
                self.last_human_draw_pos = None

    def _draw_at_human_pen(self):
        x = int(self.human_pen_pos[0] * self.pixels_per_cm)
        y = int(self.human_pen_pos[1] * self.pixels_per_cm)
        pygame.draw.circle(self.board_image, (0, 0, 255), (x, y), 3)

    def _update_human_scrolling(self):
        """
        If human scrolling is active (e.g., via a key press), update the board offset.
        Here we use a fixed scroll delta.
        """
        if self.human_scrolling:
            self.board_offset_y += 5

    def render(self, screen):
        """
        Renders the board for human viewing using Pygame.
        The board image is larger than the display, so we extract a viewport based on board_offset_y.
        The human sees:
        - The board (with drawings).
        - A blue circle indicating the human pen.
        - A black circle for the AI pen.
        - A red transparent square (with border) showing the AI focus window.
        """
        # Start with a copy of the board image.
        board_to_show = self.board_image.copy()
        
        # Draw human pen indicator.
        if self.human_pen_active and self.human_cursor_pos is not None:
            x = int(self.human_pen_pos[0] * self.pixels_per_cm)
            y = int(self.human_pen_pos[1] * self.pixels_per_cm)
            pygame.draw.circle(board_to_show, (0, 0, 255), (x, y), 5, 2)
        
        # Draw AI pen indicator.
        ai_x = int(self.ai_pen_pos[0] * self.pixels_per_cm)
        ai_y = int(self.ai_pen_pos[1] * self.pixels_per_cm)
        pygame.draw.circle(board_to_show, (0, 0, 0), (ai_x, ai_y), 5)
        
        # Draw AI focus tracker as a red transparent square.
        tracker_size = self.focus_size_pix  # e.g., 380 pixels if focus_size_cm=38 and 10 pixels/cm
        tracker_x = int(self.vision_pos[0] * self.pixels_per_cm) - tracker_size // 2
        tracker_y = int(self.vision_pos[1] * self.pixels_per_cm) - tracker_size // 2
        tracker_surface = pygame.Surface((tracker_size, tracker_size), pygame.SRCALPHA)
        tracker_surface.fill((255, 0, 0, 77))  # about 30% opacity
        board_to_show.blit(tracker_surface, (tracker_x, tracker_y))
        pygame.draw.rect(board_to_show, (255, 0, 0), (tracker_x, tracker_y, tracker_size, tracker_size), 2)
        
        # Determine viewport based on board_offset_y.
        screen_width, screen_height = screen.get_size()
        # Ensure board_offset_y is within valid range.
        max_offset = max(0, self.board_height_pix - screen_height)
        self.board_offset_y = np.clip(self.board_offset_y, 0, max_offset)
        viewport_rect = pygame.Rect(0, self.board_offset_y, screen_width, screen_height)
        viewport = board_to_show.subsurface(viewport_rect)
        
        # Blit the viewport to the screen.
        screen.blit(viewport, (0, 0))
        pygame.display.flip()

# =====================
# Replay Buffer Class
# =====================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

# =====================
# DDPG Network Definitions
# =====================

# The Actor takes as input the state (both an image and numeric vector) and outputs a 4-dim action.
class Actor(nn.Module):
    def __init__(self, numeric_input_dim, action_dim):
        super(Actor, self).__init__()
        # CNN for image processing
        # Assume input image shape: (3, H, W) with H=BOARD_HEIGHT_PIX, W=BOARD_WIDTH_PIX.
        # For speed, we resize the image to a smaller size (e.g., 84x84).
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # out: 16 x 40 x 40 (if input 84x84)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        # Compute conv output size (assume input resized to 84x84)
        conv_output_size = 32 * 7 * 7

        # MLP for numeric input
        self.fc_numeric = nn.Sequential(
            nn.Linear(numeric_input_dim, 32),
            nn.ReLU(),
        )

        # Combine image and numeric features
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size + 32, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # output between -1 and 1; we will scale appropriately.
        )

    def forward(self, img, numeric):
        # img is expected to be (batch, 3, 84, 84)
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        num_feat = self.fc_numeric(numeric)
        x = torch.cat([x, num_feat], dim=1)
        action = self.fc(x)
        return action

# The Critic takes state and action and outputs a scalar Q-value.
class Critic(nn.Module):
    def __init__(self, numeric_input_dim, action_dim):
        super(Critic, self).__init__()
        # CNN for image processing (same as actor)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        conv_output_size = 32 * 7 * 7

        self.fc_numeric = nn.Sequential(
            nn.Linear(numeric_input_dim, 32),
            nn.ReLU(),
        )
        # Combine image, numeric, and action features
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size + 32 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, img, numeric, action):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        num_feat = self.fc_numeric(numeric)
        x = torch.cat([x, num_feat, action], dim=1)
        q_value = self.fc(x)
        return q_value

# =====================
# DDPG Agent Class
# =====================

class DDPGAgent:
    def __init__(self, numeric_input_dim=8, action_dim=4, device='cpu'):
        self.device = device
        self.actor = Actor(numeric_input_dim, action_dim).to(self.device)
        self.actor_target = Actor(numeric_input_dim, action_dim).to(self.device)
        self.critic = Critic(numeric_input_dim, action_dim).to(self.device)
        self.critic_target = Critic(numeric_input_dim, action_dim).to(self.device)

        # Copy initial parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005  # for soft update

    def select_action(self, state, noise_scale=0.1):
        """
        Given state (dictionary with 'image' and 'numeric'), select an action.
        The image is now a pygame.Surface. We resize it to 84x84 using pygame and convert it
        to a numpy array.
        """
        # Get the image surface from state.
        img_surface = state['image']
        # Resize the surface to 84x84 using pygame.
        resized_surface = pygame.transform.smoothscale(img_surface, (84, 84))
        # Convert the resized surface to a numpy array.
        # pygame.surfarray.array3d returns an array with shape (width, height, channels).
        img_array = pygame.surfarray.array3d(resized_surface)
        # Transpose to get shape (height, width, channels).
        img_array = np.transpose(img_array, (1, 0, 2))
        
        # Convert to torch tensor and adjust dimensions to (batch, channels, height, width)
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1).unsqueeze(0) / 255.0
        numeric = torch.FloatTensor(state['numeric']).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(img_tensor.to(self.device), numeric.to(self.device)).cpu().data.numpy().flatten()
        self.actor.train()
        
        # Add exploration noise.
        action += noise_scale * np.random.randn(*action.shape)
        
        # Scale the outputs:
        # action[0]: joint selection in [0, 1]
        joint_sel = (action[0] + 1) / 2.0
        # action[1]: pen mode (if used), scaled to [0, 1]
        pen_mode = (action[1] + 1) / 2.0
        # action[2:4]: pen forces scaled to [-MAX_FORCE, MAX_FORCE]
        pen_forces = action[2:4] * MAX_FORCE
        # action[4:6]: vision forces scaled to [-MAX_FORCE, MAX_FORCE]
        vision_forces = action[4:6] * MAX_FORCE
        
        return np.concatenate(([joint_sel, pen_mode], pen_forces, vision_forces))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        """Update actor and critic networks using transitions from replay buffer.
        This method is called only when a reward is received.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        # Sample a batch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Preprocess image and numeric parts for both current and next states:
        img_batch = []
        numeric_batch = []
        next_img_batch = []
        next_numeric_batch = []
        
        for s in state_batch:
            # Resize image using pygame
            resized_surface = pygame.transform.smoothscale(s['image'], (84, 84))
            # Convert to numpy array; pygame.surfarray.array3d returns shape (width, height, channels)
            img_array = pygame.surfarray.array3d(resized_surface)
            # Transpose to (height, width, channels)
            img_array = np.transpose(img_array, (1, 0, 2))
            img_batch.append(img_array)
            numeric_batch.append(s['numeric'])
        
        for s in next_state_batch:
            resized_surface = pygame.transform.smoothscale(s['image'], (84, 84))
            next_img_array = pygame.surfarray.array3d(resized_surface)
            next_img_array = np.transpose(next_img_array, (1, 0, 2))
            next_img_batch.append(next_img_array)
            next_numeric_batch.append(s['numeric'])
        
        img_batch = torch.FloatTensor(np.array(img_batch)).permute(0, 3, 1, 2).to(self.device) / 255.0
        numeric_batch = torch.FloatTensor(np.array(numeric_batch)).to(self.device)
        next_img_batch = torch.FloatTensor(np.array(next_img_batch)).permute(0, 3, 1, 2).to(self.device) / 255.0
        next_numeric_batch = torch.FloatTensor(np.array(next_numeric_batch)).to(self.device)

        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Critic update:
        with torch.no_grad():
            next_actions = self.actor_target(next_img_batch, next_numeric_batch)
            next_q = self.critic_target(next_img_batch, next_numeric_batch, next_actions)
            target_q = reward_batch + self.gamma * (1 - done_batch) * next_q
        current_q = self.critic(img_batch, numeric_batch, action_batch)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update:
        actor_loss = -self.critic(img_batch, numeric_batch, self.actor(img_batch, numeric_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks:
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)

# =====================
# Main Training Loop
# =====================

def main():
    pygame.init()
    # Create the environment.
    env = WhiteboardEnv()
    state = env.reset()
    
    # Create a display window (e.g., 800x600) so scrolling is visible.
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Whiteboard")
    
    clock = pygame.time.Clock()
    agent = DDPGAgent(numeric_input_dim=8, action_dim=6, device=torch.device("cpu"))
    
    running = True
    keyboard_reward = 0.0  # default reward
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                # Convert viewport coordinates to board coordinates.
                board_y = y + env.board_offset_y
                if 0 <= x < env.board_width_pix and 0 <= board_y < env.board_height_pix:
                    env.human_pen_active = True
                    env.human_cursor_pos = (x / env.pixels_per_cm, board_y / env.pixels_per_cm)
                else:
                    env.human_pen_active = False
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    env.human_drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    env.human_drawing = False
                    
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll the board: event.y is +1 (up) or -1 (down)
                env.board_offset_y -= event.y * 20  # adjust scroll speed as needed
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    keyboard_reward = -1.0
                    print("Keyboard reward: -1")
                elif event.key == pygame.K_2:
                    keyboard_reward = 1.0
                    print("Keyboard reward: +1")
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_1, pygame.K_2):
                    keyboard_reward = 0.0
        
        # Agent selects an action based on the current state.
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Combine environment reward with keyboard reward.
        reward += keyboard_reward
        
        # Store the transition.
        agent.store_transition(state, action, reward, next_state, done)
        
        # When a reward event occurs, update the agent using all transitions,
        # then clear the replay buffer.
        if reward != 0:
            agent.update()
            agent.replay_buffer.clear()
            print("Reward event: reward =", reward)
        
        state = next_state
        
        # Render the updated board.
        env.render(screen)
        clock.tick(60)  # Maintain ~60 FPS
    
    pygame.quit()

if __name__ == '__main__':
    main()