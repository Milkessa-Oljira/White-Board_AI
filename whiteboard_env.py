import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# =====================
# Environment Constants
# =====================

# Board dimensions in cm:
BOARD_WIDTH_CM = 200
BOARD_HEIGHT_CM = 175

# Resolution: 
PIXELS_PER_CM = 5
BOARD_WIDTH_PIX = BOARD_WIDTH_CM * PIXELS_PER_CM  # 1000 pixels
BOARD_HEIGHT_PIX = BOARD_HEIGHT_CM * PIXELS_PER_CM  # 875 pixels

# Focus window dimensions (clear region): 
FOCUS_SIZE_CM = 75
FOCUS_SIZE_PIX = FOCUS_SIZE_CM * PIXELS_PER_CM

# Pen joint parameters
PEN_MASS = 0.015  # in kg
PEN_FRICTION_COEFF = 0.35  # friction coefficient for pen movement
MAX_FORCE = 15.0  # maximum muscle-like force (N)

# Vision joint parameters
EYE_MASS = 0.0075  # in kg
EYE_FRICTION_COEFF = 0.2  # friction for vision joint (without extra friction)

# Boundary limits in cm
MIN_X, MAX_X = 0, BOARD_WIDTH_CM
MIN_Y, MAX_Y = 0, BOARD_HEIGHT_CM

# Pen modes (only valid if moving pen)
PEN_MODE_MOVE = 0  # moving without writing
PEN_MODE_WRITE = 1  # writing mode

# =====================
# Environment Class
# =====================

class WhiteboardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, render_mode="human"):
        super(WhiteboardEnv, self).__init__()
        
        # Initialize pygame if not already initialized.
        if not pygame.get_init():
            pygame.init()
        
        # Board dimensions in cm and pixels (using global constants)
        self.board_width_cm = BOARD_WIDTH_CM 
        self.board_height_cm = BOARD_HEIGHT_CM
        self.pixels_per_cm = PIXELS_PER_CM  
        self.board_width_pix = self.board_width_cm * self.pixels_per_cm
        self.board_height_pix = self.board_height_cm * self.pixels_per_cm 

        self.focus_size_cm = FOCUS_SIZE_CM
        self.focus_size_pix = self.focus_size_cm * self.pixels_per_cm

        self.dt = 1.0 / self.metadata["render_fps"]
        self.render_mode = render_mode
        self.screen = None

        # Define action space: a Dict containing:
        # - 'joint_select': Discrete(2) (0 for pen, 1 for vision)
        # - 'pen_mode': Discrete(2) (0: move, 1: write)
        # - 'pen_force': Box(2,) in [-MAX_FORCE, MAX_FORCE]
        # - 'vision_force': Box(2,) in [-MAX_FORCE, MAX_FORCE]
        self.action_space = spaces.Dict({
            'joint_select': spaces.Discrete(2),
            'pen_mode': spaces.Discrete(2),
            'pen_force': spaces.Box(low=-MAX_FORCE, high=MAX_FORCE, shape=(2,), dtype=np.float32),
            'vision_force': spaces.Box(low=-MAX_FORCE, high=MAX_FORCE, shape=(2,), dtype=np.float32),
        })
        
        # Observation space: a Dict containing:
        # - 'image_array': the corresponding NumPy array (for processing)
        # - 'numeric': a 4-dimensional vector [ai_pen_force_x, ai_pen_force_y, vision_force_x, vision_force_y] normalized by MAX_FORCE.
        self.observation_space = spaces.Dict({
            'image_array': spaces.Box(low=0, high=255, shape=(self.board_height_pix, self.board_width_pix, 3), dtype=np.uint8),
            'numeric': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        })

        # Create a blank white board as a Pygame Surface.
        self.board_image = pygame.Surface((self.board_width_pix, self.board_height_pix))
        self.board_image.fill((255, 255, 255))
        
        # Board vertical scroll offset (in pixels)
        self.board_offset_y = 0

        # ----- AI Joint States -----
        # AI pen: position (cm) and velocity (cm/s)
        self.ai_pen_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.ai_pen_vel = np.zeros(2, dtype=np.float32)
        # Vision joint for AI (controls focus)
        self.vision_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.vision_vel = np.zeros(2, dtype=np.float32)
        # Last applied forces (for reaction feedback)
        self.last_ai_pen_force = np.zeros(2, dtype=np.float32)
        self.last_vision_force = np.zeros(2, dtype=np.float32)

        self.last_mode = None

        # ----- Human Joint States -----
        # Human pen: position (cm) and velocity (cm/s)
        self.human_pen_pos = np.array([self.board_width_cm / 2, self.board_height_cm / 2], dtype=np.float32)
        self.human_pen_vel = np.zeros(2, dtype=np.float32)
        # Flags for human input
        self.human_pen_active = False
        self.human_drawing = False
        # self.human_scrolling = False
        self.human_cursor_pos = None

        # Continuous environment
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset board and joint states.
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

        observation = self.get_state()
        return observation, {}

    def get_state(self):
        """Returns state with consistent reaction forces."""
        composite_img = self._create_composite_image()
        composite_img_array = pygame.surfarray.array3d(composite_img)
        composite_img_array = np.transpose(composite_img_array, (1, 0, 2))
        composite_img_array = np.asarray(composite_img_array, dtype=np.uint8)

        # Calculate reaction forces *consistently* with _update_joint()
        ai_pen_force = -self._calculate_actual_force('ai_pen', self.ai_pen_vel, self.last_ai_pen_force)  # Use helper function
        vision_force = -self._calculate_actual_force('vision', self.vision_vel, self.last_vision_force) # Use helper function

        numeric_state = np.array(
            np.concatenate([ai_pen_force, vision_force]) / MAX_FORCE,
            dtype=np.float32
        )
        return {'image_array': composite_img_array, 'numeric': numeric_state}

    def _calculate_actual_force(self, joint_type, velocity, last_applied_force):
        """Calculates the actual force considering friction and mode."""
        if joint_type == 'ai_pen':
            mass = PEN_MASS
            damping = PEN_FRICTION_COEFF
            mode = self.last_mode # Keep track of the last mode
        elif joint_type == 'vision':
            mass = EYE_MASS
            damping = EYE_FRICTION_COEFF
            mode = None  # Vision doesn't have modes

        gravity = 9.8
        static_friction_coeff = 0.4
        dynamic_friction_coeff = 0.2
        small_threshold = 0.01

        if np.linalg.norm(velocity) < small_threshold:
            if np.linalg.norm(last_applied_force) < static_friction_coeff * mass * gravity:
                force = -damping * velocity # Only viscous damping
            else:
                force = -damping * velocity - dynamic_friction_coeff * mass * gravity * velocity / (np.linalg.norm(velocity) + 1e-6) # Add dynamic friction
        else:
            force = -damping * velocity - dynamic_friction_coeff * mass * gravity * velocity / (np.linalg.norm(velocity) + 1e-6) # Add dynamic friction

        return force

    def _create_composite_image(self):
        """
        Creates a composite image by blurring the entire board (via downscaling/upscaling)
        except for a clear focus window (centered on the AI vision position).
        """
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
        
        # Copy the clear focus region from the board.
        focus_surface = self.board_image.subsurface(focus_rect).copy()
        blurred.blit(focus_surface, (x1, y1))
        return blurred

    def step(self, action):
        """Processes an action and updates joint states and the board."""
        # Update human input.
        self._update_human_pen()
        self._update_human_scrolling()

        # Process action (a dict).
        joint_select = action['joint_select']
        mode = action['pen_mode']  # Valid only if joint_select==0
        
        pen_force = np.zeros(2, dtype=np.float32)
        vision_force = np.zeros(2, dtype=np.float32)
        
        if joint_select == 0:  # Pen control.
            # For all pen modes (move, write, scroll), apply the pen force.
            pen_force = np.clip(action['pen_force'], -MAX_FORCE, MAX_FORCE)
            self.last_mode = mode
            self.last_ai_pen_force = pen_force
            self.last_vision_force = np.zeros(2, dtype=np.float32)
        else:  # Vision control.
            vision_force = np.clip(action['vision_force'], -MAX_FORCE, MAX_FORCE)
            self.last_ai_pen_force = np.zeros(2, dtype=np.float32)
            self.last_vision_force = vision_force
            mode = None  # Mode not used for vision.
            self.last_mode = mode
        
        # Update AI joints.
        self._update_joint('ai_pen', pen_force, mode)
        self._update_joint('vision', vision_force)
        
        # Compute reward (placeholder).
        reward = 0
        
        self.done = False
        next_state = self.get_state()
        info = {'mode': mode}
        return next_state, reward, self.done, False, info

    def _update_joint(self, joint, applied_force, mode=None):
        """
        Updates the state of the specified joint using semi-implicit Euler integration.
        We model friction as a viscous damping force. When in scrolling mode, extra damping
        is applied, and only the vertical component of the applied force is used for scrolling.
        """
        # Select parameters based on joint type.
        if joint == 'ai_pen':
            mass = PEN_MASS
            damping = PEN_FRICTION_COEFF
            v = self.ai_pen_vel
            p = self.ai_pen_pos
        elif joint == 'vision':
            mass = EYE_MASS
            damping = EYE_FRICTION_COEFF
            v = self.vision_vel
            p = self.vision_pos
        else:
            raise ValueError("Invalid joint name.")
        
        a = (applied_force - damping * v) / mass
        v_new = v + a * self.dt
        p_new = p + v_new * self.dt

        # --- Boundary Enforcement with Margins ---
        if joint == 'vision':
            # For vision, ensure the focus window stays within the board.
            margin_x = self.focus_size_cm / 2.0
            margin_y = self.focus_size_cm / 2.0
            p_new[0] = np.clip(p_new[0], margin_x, self.board_width_cm - margin_x)
            p_new[1] = np.clip(p_new[1], margin_y, self.board_height_cm - margin_y)
            if p_new[0] <= margin_x or p_new[0] >= self.board_width_cm - margin_x:
                v_new[0] = 0
            if p_new[1] <= margin_y or p_new[1] >= self.board_height_cm - margin_y:
                v_new[1] = 0
        elif joint == 'ai_pen':
            # For the pen, define a margin so it never reaches the very edge.
            pen_margin = 2.0  # in cm
            p_new[0] = np.clip(p_new[0], pen_margin, self.board_width_cm - pen_margin)
            p_new[1] = np.clip(p_new[1], pen_margin, self.board_height_cm - pen_margin)
            if p_new[0] <= pen_margin or p_new[0] >= self.board_width_cm - pen_margin:
                v_new[0] = 0
            if p_new[1] <= pen_margin or p_new[1] >= self.board_height_cm - pen_margin:
                v_new[1] = 0
        # --- End Boundary Enforcement ---

        # Update state.
        if joint == 'ai_pen':
            self.ai_pen_vel = v_new
            self.ai_pen_pos = p_new
            if mode == PEN_MODE_WRITE:
                self._draw_at_ai_pen()
        else:
            self.vision_vel = v_new
            self.vision_pos = p_new

    def _draw_at_ai_pen(self):
        """Draws dots representing the AI pen on the board with density based on velocity."""
        x = int(self.ai_pen_pos[0] * self.pixels_per_cm)
        y = int(self.ai_pen_pos[1] * self.pixels_per_cm)
        
        # Base dot size
        dot_size = 3
        
        # Draw the main dot
        pygame.draw.circle(self.board_image, (0, 0, 0), (x, y), dot_size)
        
        # Calculate velocity magnitude
        vel_magnitude = np.linalg.norm(self.ai_pen_vel)
        
        # If moving at moderate speed, add intermediate dots to fill gaps
        if vel_magnitude > 2.0:
            # Add additional dots based on velocity
            num_extra_dots = min(int(vel_magnitude / 2), 5)  # Limit maximum extra dots
            
            # Calculate previous position based on velocity vector
            vel_direction = self.ai_pen_vel / (vel_magnitude + 1e-6)  # Avoid division by zero
            
            for i in range(1, num_extra_dots + 1):
                # Place dots between current and previous position
                factor = i / (num_extra_dots + 1)
                prev_x = int((self.ai_pen_pos[0] - vel_direction[0] * factor * self.dt * vel_magnitude) * self.pixels_per_cm)
                prev_y = int((self.ai_pen_pos[1] - vel_direction[1] * factor * self.dt * vel_magnitude) * self.pixels_per_cm)
                
                # Ensure coordinates are within bounds
                if 0 <= prev_x < self.board_width_pix and 0 <= prev_y < self.board_height_pix:
                    # Draw smaller dots for the intermediate positions
                    smaller_dot_size = max(dot_size - 1, 2)
                    pygame.draw.circle(self.board_image, (0, 0, 0), (prev_x, prev_y), smaller_dot_size)

    def _update_human_pen(self):
        """Updates human pen position based on mouse cursor."""
        if self.human_pen_active and self.human_cursor_pos is not None:
            target = np.array(self.human_cursor_pos, dtype=np.float32)
            self.human_pen_pos = target.copy()
            if self.human_drawing:
                self._draw_at_human_pen()

    def _draw_at_human_pen(self):
        """Draws dots representing the human pen on the board with density based on velocity."""
        x = int(self.human_pen_pos[0] * self.pixels_per_cm)
        y = int(self.human_pen_pos[1] * self.pixels_per_cm)
        
        # Base dot size
        dot_size = 3
        
        # Draw the main dot (blue for human)
        pygame.draw.circle(self.board_image, (0, 0, 255), (x, y), dot_size)
        
        # Calculate velocity magnitude
        vel_magnitude = np.linalg.norm(self.human_pen_vel)
        
        # If moving at moderate speed, add intermediate dots to fill gaps
        if vel_magnitude > 2.0:
            # Add additional dots based on velocity
            num_extra_dots = min(int(vel_magnitude / 2), 5)  # Limit maximum extra dots
            
            # Calculate previous position based on velocity vector
            vel_direction = self.human_pen_vel / (vel_magnitude + 1e-6)  # Avoid division by zero
            
            for i in range(1, num_extra_dots + 1):
                # Place dots between current and previous position
                factor = i / (num_extra_dots + 1)
                prev_x = int((self.human_pen_pos[0] - vel_direction[0] * factor * self.dt * vel_magnitude) * self.pixels_per_cm)
                prev_y = int((self.human_pen_pos[1] - vel_direction[1] * factor * self.dt * vel_magnitude) * self.pixels_per_cm)
                
                # Ensure coordinates are within bounds
                if 0 <= prev_x < self.board_width_pix and 0 <= prev_y < self.board_height_pix:
                    # Draw smaller dots for the intermediate positions
                    smaller_dot_size = max(dot_size - 1, 2)
                    pygame.draw.circle(self.board_image, (0, 0, 255), (prev_x, prev_y), smaller_dot_size)

    def _update_human_scrolling(self):
        """Updates board_offset_y if human scrolling is active."""
        if self.human_scrolling:
            self.board_offset_y += 5

    def render(self):
        """Renders the board using Pygame. Extracts a viewport based on board_offset_y."""
        if self.render_mode != "human":
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Whiteboard")
        
        # Ensure board_offset_y is within valid range
        screen_width, screen_height = self.screen.get_size()
        max_offset = max(0, self.board_height_pix - screen_height)
        self.board_offset_y = np.clip(self.board_offset_y, 0, max_offset)
        
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
        tracker_size = self.focus_size_pix
        tracker_x = int(self.vision_pos[0] * self.pixels_per_cm) - tracker_size // 2
        tracker_y = int(self.vision_pos[1] * self.pixels_per_cm) - tracker_size // 2
        tracker_surface = pygame.Surface((tracker_size, tracker_size), pygame.SRCALPHA)
        tracker_surface.fill((255, 0, 0, 77))
        board_to_show.blit(tracker_surface, (tracker_x, tracker_y))
        pygame.draw.rect(board_to_show, (255, 0, 0), (tracker_x, tracker_y, tracker_size, tracker_size), 2)
        
        # Extract viewport based on board_offset_y
        viewport_rect = pygame.Rect(0, self.board_offset_y, screen_width, min(screen_height, self.board_height_pix - self.board_offset_y))
        viewport = board_to_show.subsurface(viewport_rect)
        
        self.screen.fill((200, 200, 200))  # Fill with gray background
        self.screen.blit(viewport, (0, 0))
        
        # Draw scroll indicators if not at top/bottom
        if self.board_offset_y > 0:
            pygame.draw.polygon(self.screen, (80, 80, 80), [(screen_width//2 - 10, 15), (screen_width//2 + 10, 15), (screen_width//2, 5)])
        if self.board_offset_y < max_offset:
            pygame.draw.polygon(self.screen, (80, 80, 80), [(screen_width//2 - 10, screen_height-15), (screen_width//2 + 10, screen_height-15), (screen_width//2, screen_height-5)])
        
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
