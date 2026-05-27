import pandas as pd
import numpy as np

class neural_network:
    def __init__(self, W, b):
        self.W = W
        self.b = b

# read dataset
train_df = pd.read_csv("src/data/mnist_train.csv")
test_df = pd.read_csv("src/data/mnist_test.csv")

# split train dataframe to X, y
X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values

# split test dataframe to X, y
X_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

# normalize dataset
X_train = X_train.T / 255
X_test = X_test.T / 255

# weights and biases initialization
np.random.seed(42)

# first hidden layer
neuron1 = 128
Weight1 = np.random.randn(neuron1, X_train.shape[0]) * 0.01
bias1 = np.zeros((neuron1, 1))

# output layer
neuron2 = 10
Weight2 = np.random.randn(neuron2, neuron1) * 0.01
bias2 = np.zeros((neuron2, 1))