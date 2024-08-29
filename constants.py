import math

#Environment constants
WIDTH = 1000
HEIGHT = 800
FPS = 60

BACKGROUND_COLOR = 'black'
PLATFROM_COLOR = 'white'
BALL_COLOR = 'red'

PLATFORM_THICKNES = 5
PLATFROM_LENGTH = 600
BALL_RADIUS = 30

MAX_ANGLE_CHANGE = math.radians(90)
MAX_ANGLE = math.radians(85)

MAX_Y_SPEED = 1100
MAX_X_SPEED = 900

G = 1000

X_V = WIDTH/2
Y_V = HEIGHT/2

MAX_EPISODE_DURATION = 10
FALL_PUNISHMENT = -1
TIME_REWARD = 0.01
DISTANCE_REWARD = 0.09
MAX_REWARD = 100

#Neural Network constants
INPUT_DIMENSION = 4
OUTPUT_DIMENSION = 3

#Training
BATCH_SIZE = 30
GAMMA = 0.99
EPSILON = 0.2
LEARNING_RATE = 1e-3
NEURONS = 256
NORMALIZED = True

VERSION = 1