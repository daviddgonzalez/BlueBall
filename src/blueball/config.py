"""Tunable constants for Blue Ball. Edit values here to retune feel."""

# Display
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
BACKGROUND_COLOR = (126, 199, 255)

# Physics loop
PHYS_HZ = 120
PHYS_DT = 1.0 / PHYS_HZ
MAX_ACCUMULATED_STEPS = 10  # avoid spiral-of-death if a frame is very slow

# Gravity (Pymunk uses y-down by default if we configure it that way; we use y-down here)
GRAVITY = (0, 800)

# Ball physics
BALL_RADIUS = 16
BALL_MASS = 1.0
BALL_FRICTION = 0.9
BALL_ELASTICITY = 0.05
MOVE_TORQUE = 3850.0
MAX_ANGULAR_VEL = 60.5
AIR_CONTROL = 0.8
# Direct horizontal force applied while airborne so the ball can change
# direction midair. Torque alone only spins it.
AIR_MOVE_FORCE = 263.0

# Jump
JUMP_IMPULSE = 300.0
JUMP_CUT_FACTOR = 0.4

# Player dies if they fall this far below the screen
FALL_DEATH_Y = 1200

# Input feel (seconds)
JUMP_BUFFER_TIME = 0.10
COYOTE_TIME = 0.08
GROUNDED_NORMAL_TOLERANCE_DEG = 30.0

# Camera
CAMERA_DEAD_ZONE_W = 200
CAMERA_DEAD_ZONE_H = 120
CAMERA_LERP = 0.15

# Patroller defaults
PATROLLER_SPEED = 60.0

# Collectible visuals
COLLECTIBLE_RADIUS = 10
COLLECTIBLE_PULSE_HZ = 2.0

# Determinism
DEFAULT_SEED = 1
