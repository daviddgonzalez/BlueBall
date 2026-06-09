"""Tunable constants for Blue Ball. Edit values here to retune feel."""

# Display
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
BACKGROUND_COLOR = (126, 199, 255)

# Pixel-art render pipeline: world is drawn to a (WINDOW / PIXEL_SCALE) virtual
# surface and nearest-neighbor-upscaled to the window. Must divide WINDOW evenly.
PIXEL_SCALE = 2

# Screen-shake magnitude decays exponentially toward 0 at this rate (per second).
SHAKE_DECAY = 8.0

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
MAX_ANGULAR_VEL = 19.6875
# Hard cap on the ball's linear-velocity magnitude. Matched to the ground-roll
# top speed (MAX_ANGULAR_VEL * BALL_RADIUS = 315 px/s) so the ball doesn't
# spin faster than it can translate (which would look like slipping).
MAX_LINEAR_SPEED = 315.0
# Torque multiplier while airborne. 0 = no torque in air, so the ball's spin
# is frozen during a jump and there's no slip-induced "kick" on landing.
AIR_CONTROL = 0.0
# Direct horizontal force applied while grounded - bypasses the friction
# acceleration ceiling so reversals don't feel mushy. Torque is still applied
# in parallel so the ball visibly spins as it rolls. Asymmetric like the air
# forces: BRAKE when the input opposes current velocity (reversal), ACCEL when
# it matches.
GROUND_MOVE_FORCE = 420.0
GROUND_MOVE_FORCE_BRAKE = 750.0
# Horizontal force in midair. Asymmetric: BRAKE is applied when the input
# direction is opposite the current horizontal velocity (correcting a wrong
# jump arc); ACCEL when the input matches velocity (or velocity is ~0).
# Higher brake than accel keeps air control responsive for corrections
# without letting the player accelerate freely in midair.
AIR_MOVE_FORCE_BRAKE = 750.0
AIR_MOVE_FORCE_ACCEL = 218.4

# Jump
JUMP_IMPULSE = 315.0
JUMP_CUT_FACTOR = 0.4

# Player dies if they fall this far below the screen
FALL_DEATH_Y = 1200

# Input feel (seconds)
JUMP_BUFFER_TIME = 0.10
COYOTE_TIME = 0.08
# Surfaces within this angle of flat count as "ground" for jumping/coyote.
GROUNDED_NORMAL_TOLERANCE_DEG = 30.0
# A more lenient angle for "I touched down" — used only to refresh the air
# jump. Touching any slope up to this steepness (e.g. a bump side) refreshes
# the double jump, even though it's too steep to push off as primary ground.
AIR_JUMP_REFRESH_TOLERANCE_DEG = 70.0

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

# Boost pads
BOOST_PAD_THICKNESS = 16  # px — how thick the floor strip is in world units
# Extra sensor height ABOVE the pad, so a ball passing just over the pad
# (airborne after a bump/launch) still triggers the boost instead of missing it.
BOOST_PAD_CATCH_HEIGHT = 28
BOOST_PAD_DEFAULT_MULTIPLIER = 2.0
# Fraction of the (directional cap - current) velocity gap closed by the
# instant kick at pickup time. The kick is applied along the pad's arrow
# direction. 1.0 = snap to the new cap; lower = gentler launch.
BOOST_PAD_KICK_FACTOR = 0.3

# Phase 3 chunks
ICE_FLOOR_FRICTION = 0.05
SPRING_DEFAULT_IMPULSE = 600.0
CRUMBLE_DEFAULT_DELAY_S = 0.5
MOVING_PLATFORM_DEFAULT_SPEED = 80.0
CHARGER_DEFAULT_SIGHT_RANGE = 200.0
CHARGER_DEFAULT_SIGHT_ARC_DEG = 60.0
CHARGER_DEFAULT_CHARGE_SPEED = 180.0
CHARGER_DEFAULT_PATROL_SPEED = 40.0
PROJECTILE_DEFAULT_SPEED = 220.0
PROJECTILE_DEFAULT_PULSE_PERIOD_S = 0.6
PROJECTILE_DEFAULT_RADIUS = 10

# Observation
MAX_RAY_LEN = 300.0
NUM_RAYS = 8

# Active render theme. "pixel" ships now; "neon" is a future slot.
ACTIVE_THEME = "pixel"

# AI / GA training
TRAIN_POP_SIZE      = 80      # spec default for real training
TRAIN_GENERATIONS   = 200     # spec default for real training
MAX_STEPS           = 3000    # per-evaluation timeout (~25s at PHYS_HZ=120)
GA_MUTATION_RATE    = 0.1
GA_MUTATION_SIGMA   = 0.1
GA_TOURNAMENT_K     = 4
GA_ELITISM          = 1
GA_FITNESS_STD_PENALTY = 1.0  # lambda: per-episode std penalty (mean - lam*std)
# Goal-completion bonus, in units of the level's full width: reaching the goal
# adds GOAL_MULT * level_width to fitness, so completion dominates a no-goal
# traversal (which maxes near level_width) and auto-scales with level length.
GOAL_MULT              = 2.0

# --- Completion Gym ---
# Flat reward banked per cleared gym segment. ~ GOAL_MULT (2.0) * a typical
# segment width (~600 px), so reward-per-completion is in the same range as the
# campaign goal bonus (aids transfer). Tunable.
GYM_SEGMENT_BONUS = 1200.0
GYM_RAMP_PER_SEGMENT = 0.15  # target tier climbs by this per segment of depth
GYM_SIGMA = 1.0              # Gaussian spread mixing adjacent tiers

# Reference seeds for reproducible training runs. Pinning these makes a run
# fully deterministic: GA_SEED fixes evolution (population init, mutation,
# crossover, tournament); INFINITE_RUN_SEED fixes the Infinite Run chunk
# layout (the sampler_seed); world_seed (DEFAULT_SEED) fixes physics. Same
# triple -> byte-identical best genome. Change these to train on a different
# reference course.
GA_SEED             = 0
INFINITE_RUN_SEED   = 1234
