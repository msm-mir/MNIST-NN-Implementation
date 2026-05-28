import pandas as pd
import numpy as np

class neural_network:
    def __init__(self, X, y_train, neurons):
        # weight matrix
        self.W = {}
        # bias matrix (with one column)
        self.b = {}
        # output of linear equation
        self.Z = {}
        # output of activation function
        self.A = {0: X}
        # number of neurons for each layer
        self.n = neurons
        # one hot y matrix
        self.y_oh = self.init_y_one_hot(y_train)
        # derivatives of matrices
        self.dW = {}
        self.db = {}
        self.dZ = {}
    
    # weights and biases initialization
    def init_W_b(self):
        np.random.seed(42)

        # start from 1st layer to the last layer that we've set
        for i in range(1, len(self.n)):
            self.W[i] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            self.b[i] = np.zeros((self.n[i], 1))
    
    # ReLU activation function for hidden layers
    def relu(self, Z):
        return np.maximum(0, Z)
    
    # Sigmoid activation function for output layer
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    # forward propagation step
    def forward_propagation(self):
        # for each layer
        for i in range(1, len(self.n)):
            # calculate linear equation output (Z = W . A + b)
            self.Z[i] = np.dot(self.W[i], self.A[i - 1]) + self.b[i]
            
            # ReLU activation function for all layers except output layer
            if i != len(self.n) - 1:
                self.A[i] = self.relu(self.Z[i])
            # Sigmoid activation function only for output layer
            else:
                self.A[i] = self.sigmoid(self.Z[i])

    # broadcast y_train to A's dimensions
    def init_y_one_hot(self, y_train):
        m = y_train.shape[0]
        self.y_oh = np.zeros((10, m))
        self.y_oh[y_train, np.arange(m)] = 1

    # calculate loss function
    def cross_entropy(self):
        m = y_train.shape[0]

        # clip output layer values to avoid exact 0 and 1
        A_clipped = np.clip(self.A[-1], 1e-15, 1 - 1e-15)

        # calculate the formula
        loss_elements = (self.y_oh * np.log(A_clipped)) + ((1 - self.y_oh) * np.log(1 - A_clipped))
        cost = (-1 / m) * np.sum(loss_elements)

        return np.squeeze(cost)


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

# number of neurons for each layer (input, hidden, output)
neurons = {0: X_train.shape[0], 1: 128, 2: 10}