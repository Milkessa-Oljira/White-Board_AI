import os
from gymnasium.envs.registration import register
from agent import HGABLAgent

# Register the custom environment
register(
    id='WhiteboardEnv-v0',
    entry_point='whiteboard_env:WhiteboardEnv',  # Assumes whiteboard_env.py exists
)

def main():
    # Initialize the agent
    agent = HGABLAgent()
    state_file = "agent_state.pth"  # Fixed file path for simplicity

    # Load state if it exists
    if os.path.exists(state_file):
        agent.load_state(state_file)
        print(f"Loaded agent state from {state_file}")

    # Run the agent
    agent.run()

    # Save state when the program exits
    agent.save_state(state_file)
    print(f"Saved agent state to {state_file}")

if __name__ == "__main__":
    main()