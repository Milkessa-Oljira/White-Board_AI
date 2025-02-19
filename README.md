# White-Board_AI

White-Board_AI is a custom Gymnasium environment for multi-agent interaction on a shared whiteboard. In this environment, a human and an AI agent work together on a digital whiteboard—where the AI learns solely from sparse human rewards (only +1 or -1 provided via keyboard) and raw sensory inputs. The environment uses realistic physics to simulate pen and vision joint dynamics, offering a unique testbed for studying behavior, collaboration, and sparse-reward learning.

---

## Features

- **Multi-Agent Interaction:**  
  A shared canvas where human and AI agents interact in real time.

- **Sparse Reward Signals:**  
  The AI receives only human-provided rewards (+1 or -1) via keyboard, forcing it to learn from sparse, delayed feedback.

- **Realistic Physics Simulation:**  
  - **Pen Joint:** Modeled with a mass of 0.015 kg and viscous friction (0.35), with position clamped by a safety margin to prevent it from disappearing.
  - **Vision Joint:** Modeled with a mass of 0.0075 kg and lower friction (0.2) to ensure the focus window always remains within the board.
  - Semi-implicit Euler integration ensures stable and smooth motion.
  
- **Multi-Modal Observations:**  
  - A composite image of the board is provided (a blurred full board with a clear focus window).
  - Reaction forces (from the pen and vision joints) are given as a numeric vector.

- **Pygame Rendering:**  
  The environment is rendered in real time using Pygame, providing a smooth, interactive display. Only human scrolling is enabled.

- **Gymnasium Compatibility:**  
  Fully compliant with the Gymnasium API for easy integration with modern RL frameworks.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Milkessa-Oljira/White-Board_AI.git
   cd White-Board_AI
