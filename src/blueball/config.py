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
GRAVITY = (0, 480)

# Ball physics
BALL_RADIUS = 16
BALL_MASS = 1.0
BALL_FRICTION = 0.9
BALL_ELASTICITY = 0.05
MOVE_TORQUE = 6784.0
MAX_ANGULAR_VEL = 28.125
# Hard cap on the ball's linear-velocity magnitude. Matched to the ground-roll
# top speed (MAX_ANGULAR_VEL * BALL_RADIUS = 450 px/s) so the ball doesn't
# spin faster than it can translate (which would look like slipping).
MAX_LINEAR_SPEED = 450.0
# Torque multiplier while airborne. 0 = no torque in air, so the ball's spin
# is frozen during a jump and there's no slip-induced "kick" on landing.
AIR_CONTROL = 0.0
# Direct horizontal force applied while grounded - bypasses the friction
# acceleration ceiling so reversals don't feel mushy. Torque is still applied
# in parallel so the ball visibly spins as it rolls.
GROUND_MOVE_FORCE = 420.0
# Horizontal force in midair. Asymmetric: BRAKE is applied when the input
# direction is opposite the current horizontal velocity (correcting a wrong
# jump arc); ACCEL when the input matches velocity (or velocity is ~0).
# Higher brake than accel keeps air control responsive for corrections
# without letting the player accelerate freely in midair.
AIR_MOVE_FORCE_BRAKE = 300.0
AIR_MOVE_FORCE_ACCEL = 60.0

# Jump
JUMP_IMPULSE = 315.0
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

# Abilities
ABILITY_PICKUP_DEFAULT_HEIGHT = 64    # px above ground where pickups float
