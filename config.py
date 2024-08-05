# File paths
TRAIN_IMAGES_PATH = 'data/train-images-idx3-ubyte'
TRAIN_LABELS_PATH = 'data/train-labels-idx1-ubyte'
TEST_IMAGES_PATH = 'data/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = 'data/t10k-labels-idx1-ubyte'

# Model parameters
K = 5  # Number of neighbors for k-NN

# Data subset sizes (for faster testing)
N_TRAIN = 1000
N_TEST = 100

# Output settings
PRINT_INTERVAL = 10  # Print progress every 10 images
NUM_PREDICTIONS_TO_SHOW = 10  # Number of individual predictions to display
