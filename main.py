#!/usr/bin/env python
"""
Main entry point for the multi-agent RL scenario with whiteboard environment.
This file handles environment creation, agent initialization, and the main game loop.
"""

import gymnasium as gym
import pygame
import torch
from gymnasium.envs.registration import register
from agent import SACAgent

# Register the custom environment (do this before creating the environment)
register(
    id='WhiteboardEnv-v0',
    entry_point='whiteboard_env:WhiteboardEnv',
)

def main():
    pygame.init()
    
    # Create the environment using gymnasium
    env = gym.make('WhiteboardEnv-v0')
    state, _ = env.reset()
    
    # Create a display window (e.g., 800x600) so scrolling is visible
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Whiteboard")
    
    clock = pygame.time.Clock()
    env_dims = {
        'image_height': env.observation_space['image_array'].shape[0],
        'image_width': env.observation_space['image_array'].shape[1],
        'numeric_dim': env.observation_space['numeric'].shape[0]
    }
    agent = SACAgent(env_dims, device=torch.device("cpu"))
    running = True
    keyboard_reward = 0.0  # default reward
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                # Convert viewport coordinates to board coordinates
                board_y = y + env.unwrapped.board_offset_y
                if 0 <= x < env.unwrapped.board_width_pix and 0 <= board_y < env.unwrapped.board_height_pix:
                    env.unwrapped.human_pen_active = True
                    env.unwrapped.human_cursor_pos = (x / env.unwrapped.pixels_per_cm, board_y / env.unwrapped.pixels_per_cm)
                else:
                    env.unwrapped.human_pen_active = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    env.unwrapped.human_drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    env.unwrapped.human_drawing = False

            elif event.type == pygame.MOUSEWHEEL:
                # Scroll the board: event.y is +1 (up) or -1 (down)
                env.unwrapped.board_offset_y -= event.y * 20  # adjust scroll speed as needed

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

        # Agent selects an action (still a numpy array)
        action_array = agent.select_action(state)

        # *KEY CHANGE*: Convert the numpy array action to a dictionary
        action = {
            'joint_select': int(action_array[0]),  # Convert to int for discrete action
            'pen_mode': int(action_array[1]),       # Convert to int for discrete action
            'pen_force': action_array[2:4],
            'vision_force': action_array[4:6],
        }

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Combine environment reward with keyboard reward
        reward += keyboard_reward

        # Store the transition
        agent.store_transition(state, action, reward, next_state, done)

        # When a reward event occurs, update the agent using all transitions,
        # then clear the replay buffer
        if reward != 0:
            agent.update()
            agent.replay_buffer.clear()
            print("Reward event: reward =", reward)

        state = next_state

        # Render the updated board
        env.render()
        clock.tick(120)  # Maintain 120 FPS

    env.close()
    pygame.quit()

if __name__ == '__main__':
    main()