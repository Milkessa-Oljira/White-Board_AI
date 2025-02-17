import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request
from dataclasses import dataclass
import pygame
import math

# Constants
BOARD_WIDTH = 400  # 40cm in pixels
BOARD_HEIGHT = 250  # 25cm in pixels
FOCUS_WINDOW_SIZE = 50  # 5cm in pixels
MAX_FORCE = 10.0
FRICTION_COEF = 0.3
PEN_MASS = 1.0
EYE_MASS = 1.0
DT = 0.016  # 60fps simulation

@dataclass
class Point:
    x: float
    y: float

class PhysicsEngine:
    def __init__(self):
        self.pen_pos = Point(BOARD_WIDTH/2, BOARD_HEIGHT/2)
        self.vision_pos = Point(BOARD_WIDTH/2, BOARD_HEIGHT/2)
        self.pen_velocity = Point(0, 0)
        self.vision_velocity = Point(0, 0)
        
    def apply_force(self, joint_type, fx, fy):
        if abs(fx) > MAX_FORCE:
            fx = MAX_FORCE * np.sign(fx)
        if abs(fy) > MAX_FORCE:
            fy = MAX_FORCE * np.sign(fy)
            
        if joint_type == "pen":
            pos = self.pen_pos
            vel = self.pen_velocity
            mass = PEN_MASS
        else:  # vision
            pos = self.vision_pos
            vel = self.vision_velocity
            mass = EYE_MASS
            
        # Update velocity with force and friction
        ax = (fx - FRICTION_COEF * vel.x) / mass
        ay = (fy - FRICTION_COEF * vel.y) / mass
        
        vel.x += ax * DT
        vel.y += ay * DT
        
        # Update position
        pos.x += vel.x * DT
        pos.y += vel.y * DT
        
        # Enforce boundaries
        pos.x = np.clip(pos.x, 0, BOARD_WIDTH)
        pos.y = np.clip(pos.y, 0, BOARD_HEIGHT)
        
        # Stop at boundaries
        if pos.x == 0 or pos.x == BOARD_WIDTH:
            vel.x = 0
        if pos.y == 0 or pos.y == BOARD_HEIGHT:
            vel.y = 0

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output forces and joint selection probability
        forces = torch.tanh(self.fc3(x[:, :2])) * MAX_FORCE
        joint_prob = torch.sigmoid(self.fc3(x[:, 2:]))
        return forces, joint_prob

class AIAgent:
    def __init__(self):
        # State: pen_pos, vision_pos, pen_vel, vision_vel, focus_window_content
        input_size = 8 + FOCUS_WINDOW_SIZE * FOCUS_WINDOW_SIZE
        hidden_size = 128
        # Output: force_x, force_y, joint_selection
        output_size = 3
        
        self.policy = PolicyNetwork(input_size, hidden_size, output_size)
        self.current_joint = "pen"  # Start with pen control
        
    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            forces, joint_prob = self.policy(state_tensor)
            
            # Deterministic joint selection
            self.current_joint = "pen" if joint_prob.item() > 0.5 else "vision"
            
            return {
                "joint": self.current_joint,
                "force_x": forces[0][0].item(),
                "force_y": forces[0][1].item()
            }

class WhiteboardSimulation:
    def __init__(self):
        self.physics = PhysicsEngine()
        self.ai = AIAgent()
        self.writing_surface = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.is_writing = False
        self.is_scrolling = False
        
    def update(self, human_input, reward=0):
        # Process human input
        if human_input:
            self.physics.apply_force("pen", human_input["force_x"], human_input["force_y"])
            self.is_writing = human_input.get("is_writing", False)
            self.is_scrolling = human_input.get("is_scrolling", False)
        
        # Get AI state
        state = self._get_state()
        
        # Get AI action
        ai_action = self.ai.get_action(state)
        
        # Apply AI action
        self.physics.apply_force(
            ai_action["joint"],
            ai_action["force_x"],
            ai_action["force_y"]
        )
        
        # Update writing surface if writing
        if self.is_writing:
            x, y = int(self.physics.pen_pos.x), int(self.physics.pen_pos.y)
            self.writing_surface[max(0, y-1):min(BOARD_HEIGHT, y+2),
                               max(0, x-1):min(BOARD_WIDTH, x+2)] = 255
            
        return {
            "pen_pos": (self.physics.pen_pos.x, self.physics.pen_pos.y),
            "vision_pos": (self.physics.vision_pos.x, self.physics.vision_pos.y),
            "writing_surface": self.writing_surface.tolist(),
            "ai_action": ai_action
        }
    
    def _get_state(self):
        # Combine all state information
        state = [
            self.physics.pen_pos.x / BOARD_WIDTH,
            self.physics.pen_pos.y / BOARD_HEIGHT,
            self.physics.vision_pos.x / BOARD_WIDTH,
            self.physics.vision_pos.y / BOARD_HEIGHT,
            self.physics.pen_velocity.x / MAX_FORCE,
            self.physics.pen_velocity.y / MAX_FORCE,
            self.physics.vision_velocity.x / MAX_FORCE,
            self.physics.vision_velocity.y / MAX_FORCE
        ]
        
        # Add focus window content
        x, y = int(self.physics.vision_pos.x), int(self.physics.vision_pos.y)
        half_size = FOCUS_WINDOW_SIZE // 2
        
        focus_window = self.writing_surface[
            max(0, y-half_size):min(BOARD_HEIGHT, y+half_size+1),
            max(0, x-half_size):min(BOARD_WIDTH, x+half_size+1)
        ]
        
        # Pad if necessary
        if focus_window.shape != (FOCUS_WINDOW_SIZE, FOCUS_WINDOW_SIZE):
            padded = np.zeros((FOCUS_WINDOW_SIZE, FOCUS_WINDOW_SIZE))
            padded[:focus_window.shape[0], :focus_window.shape[1]] = focus_window
            focus_window = padded
            
        state.extend(focus_window.flatten() / 255.0)
        return state

app = Flask(__name__)
simulation = WhiteboardSimulation()

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/update', methods=['POST'])
def update():
    human_input = request.json
    reward = human_input.pop('reward', 0)
    state = simulation.update(human_input, reward)
    return jsonify(state)

if __name__ == '__main__':
    app.run(debug=True)