import math

POPULATION_SIZE = 350
# percentage = (81.65 * math.sqrt(N + 0.375) + 50) / N
SELECTION_THRESHOLD_DIVISOR = (
    100 * POPULATION_SIZE / (81.65 * math.sqrt(POPULATION_SIZE + 0.375) + 50)
)
THRESHOLD = 8
WHITE = (200, 200, 200)
BLACK = (0, 0, 0)
GREY = (70, 70, 70)

MOCAP_WIDTH_PROJECTION = 3.30
MOCAP_HEIGHT_PROJECTION = 1.65
MOCAP_HEIGHT_Z_PROJECTION = 2.30

WINDOW_HEIGHT_APP = 800

WINDOW_Y = (
    WINDOW_HEIGHT_APP
    / (MOCAP_HEIGHT_PROJECTION + MOCAP_HEIGHT_Z_PROJECTION)
    * MOCAP_HEIGHT_PROJECTION
)
WINDOW_X = WINDOW_Y / MOCAP_HEIGHT_PROJECTION * MOCAP_WIDTH_PROJECTION
WINDOW_Z = WINDOW_Y / MOCAP_HEIGHT_PROJECTION * MOCAP_HEIGHT_Z_PROJECTION

WINDOW_WIDTH_APP = WINDOW_X + WINDOW_Y


USER_DOT_SIZE = 6  # Set the size of the grid block
DOT_SIZE = 4
SURR_DOT_SIZE = 5
SURR_DOT_SIZE_2 = 6
MOVE = 5
TARGET_WIDTH_PROJECTION = WINDOW_X  # / 100  # temp
TARGET_HEIGHT_PROJECTION = WINDOW_Y  # / 100  # temp
TARGET_HEIGHT_Z_PROJECTION = WINDOW_Z  # / 100  # temp
TARGET_IP = "localhost"
TARGET_PORT = 8001
NETSEND_DELAY = 0.05
SELECT_DISTANCE_THRESHOLD = 1
K = 5
NUMBER_OF_IDX_TO_APPLY_SCORE = 1
N_LARGEST_TO_SELECT = 10
MUTATION_SCALE = 0.2

USE_SIMULATION = True
FULLSCREEN = True

# additive synth patch
# DATA_FIELDS = ["carrier", "ratio", "metrodev", "att", "sus", "score", "pop"]

# vocal mocap
DATA_FIELDS = ["p1", "p2", "p3", "p4", "p5", "p6", "score", "pop"]
