from flask import Flask, render_template, jsonify, request
import threading, time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# -------------------------------
# Simulation Constants & Settings
# -------------------------------
BOARD_WIDTH = 25.0   # in centimeters
BOARD_HEIGHT = 40.0  # in centimeters
TIME_STEP = 0.1      # seconds per simulation update
MAX_FORCE = 5.0      # muscle-like maximum force (arbitrary units)
PEN_MASS = 1.0       # mass of each pen
EYE_MASS = 1.0       # weight for the AI vision "eye" (affects inertia)

# -------------------------------
# Global Simulation State
# -------------------------------
# Positions: represented as numpy arrays [x, y]
# Initialize both pens and the AI's vision focus at the center of the board.
state = {
    "human_pen": np.array([BOARD_WIDTH/2, BOARD_HEIGHT/2]),
    "ai_pen": np.array([BOARD_WIDTH/2, BOARD_HEIGHT/2]),
    "ai_vision": np.array([BOARD_WIDTH/2, BOARD_HEIGHT/2])
}

# Velocities for each joint (for physics updates)
state_vel = {
    "human_pen": np.array([0.0, 0.0]),
    "ai_pen": np.array([0.0, 0.0]),
    "ai_vision": np.array([0.0, 0.0])
}

# A lock to protect state access from multiple threads.
state_lock = threading.Lock()

# ------------------------------------------
# Deterministic Policy Neural Network (AI)
# ------------------------------------------
class DeterministicPolicy(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super(DeterministicPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Two heads: one for discrete mode selection (3 modes) and one for continuous forces.
        self.mode_head = nn.Linear(hidden_dim, 3)   # modes: 0 (move without writing), 1 (write), 2 (scroll)
        self.force_head = nn.Linear(hidden_dim, 2)    # force dx, dy

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mode_logits = self.mode_head(x)
        # Using tanh to bound output between -1 and 1, then scale by MAX_FORCE.
        forces = torch.tanh(self.force_head(x)) * MAX_FORCE
        return mode_logits, forces

# Create an instance of the deterministic policy network.
policy_net = DeterministicPolicy()

def ai_decide_action():
    """Get the AI action given the current state. The observation vector consists of:
       [human_pen_x, human_pen_y, ai_pen_x, ai_pen_y, ai_vision_x, ai_vision_y]."""
    with state_lock:
        obs = np.concatenate([
            state["human_pen"],
            state["ai_pen"],
            state["ai_vision"]
        ])
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    mode_logits, forces = policy_net(obs_tensor)
    mode = int(torch.argmax(mode_logits).item())  # deterministic selection
    forces = forces.detach().numpy()
    return mode, forces  # returns a mode (0,1,2) and a 2-element force vector

# --------------------------------------------------
# Simulation Update Loop (Physics & Agent Control)
# --------------------------------------------------
def update_simulation():
    """This loop updates the simulation state using simple physics.
       The AI alternates control of its pen and vision joints each update."""
    global state, state_vel
    while True:
        with state_lock:
            # Update positions with simple Euler integration.
            for key in ["human_pen", "ai_pen", "ai_vision"]:
                state[key] += state_vel[key] * TIME_STEP
                # Apply simple friction (velocity decay).
                state_vel[key] *= 0.9
                # Enforce board boundaries.
                state[key][0] = np.clip(state[key][0], 0, BOARD_WIDTH)
                state[key][1] = np.clip(state[key][1], 0, BOARD_HEIGHT)

            # AI Control: alternate between updating the pen joint and vision joint.
            # (The AI is only allowed to move one joint at a time.)
            current_step = int(time.time() / TIME_STEP)
            mode, forces = ai_decide_action()
            if current_step % 2 == 0:
                # Control the AI pen joint.
                # For scrolling mode (mode 2), assume extra friction reducing the effective force.
                if mode == 2:
                    forces = forces * 0.5
                state_vel["ai_pen"] += forces * (TIME_STEP / PEN_MASS)
            else:
                # Control the AI vision joint. For vision, the reaction is scaled by the eye's weight.
                state_vel["ai_vision"] += forces * (TIME_STEP / EYE_MASS)

        time.sleep(TIME_STEP)

# Start the simulation update loop in a background thread.
sim_thread = threading.Thread(target=update_simulation, daemon=True)
sim_thread.start()

# -------------------------------
# Flask Routes / API Endpoints
# -------------------------------
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/state')
def get_state():
    with state_lock:
        # Convert numpy arrays to lists for JSON serialization.
        current_state = {k: state[k].tolist() for k in state}
    return jsonify(current_state)

@app.route('/api/human_action', methods=['POST'])
def human_action():
    """
    Expects a JSON payload with the following structure:
    {
       "mode": <int>,          // 0: move without writing, 1: writing, 2: scrolling
       "force": {"dx": <float>, "dy": <float>}
    }
    Applies the force (after scaling for scrolling if needed) to the human pen joint.
    """
    data = request.get_json()
    mode = data.get("mode", 0)
    dx = data.get("force", {}).get("dx", 0.0)
    dy = data.get("force", {}).get("dy", 0.0)
    with state_lock:
        # For scrolling mode, extra friction reduces the effective force.
        if mode == 2:
            dx *= 0.5
            dy *= 0.5
        state_vel["human_pen"] += np.array([dx, dy]) * (TIME_STEP / PEN_MASS)
    return jsonify({"status": "ok"})

@app.route('/api/reward', methods=['POST'])
def reward():
    """
    Endpoint to receive a reward signal.
    In a more complete implementation, this reward would be integrated into the AI's learning process.
    For this demo, we simply print the reward.
    """
    data = request.get_json()
    reward_value = data.get("reward", 0)
    print("Received reward signal:", reward_value)
    return jsonify({"status": "ok"})

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
