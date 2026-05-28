import pandas as pd
import numpy as np

class neural_network:
    def __init__(self, X, y, neurons, learning_rate, epochs):
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
        self.y_oh = self.init_y_one_hot(y)
        # derivatives of matrices
        self.dW = {}
        self.db = {}
        self.dZ = {}

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.init_W_b()
    
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

        return self.A[len(self.n) - 1]

    # broadcast y matrix to A's dimensions
    def init_y_one_hot(self, y):
        m = y.shape[0]
        new_y = np.zeros((10, m))
        new_y[y, np.arange(m)] = 1
        return new_y

    # calculate loss function
    def cross_entropy(self):
        m = self.y_oh.shape[1]
        n = len(self.n) - 1

        # clip output layer values to avoid exact 0 and 1
        A_clipped = np.clip(self.A[n], 1e-15, 1 - 1e-15)

        # calculate the formula of cross entropy
        loss_elements = (self.y_oh * np.log(A_clipped)) + ((1 - self.y_oh) * np.log(1 - A_clipped))
        cost = (-1 / m) * np.sum(loss_elements)

        return np.squeeze(cost)
    
    # backward propagation
    def back_propagation(self):
        m = self.y_oh.shape[1]

        # for each hidden layer
        for i in range(len(self.n) - 1, 0, -1):
            if i == len(self.n) - 1:
                self.dZ[i] = self.A[i] - self.y_oh
            else:
                self.dZ[i] = np.dot(self.W[i + 1].T, self.dZ[i + 1]) * (self.Z[i] > 0)
            
            self.dW[i] = (1 / m) * np.dot(self.dZ[i], self.A[i - 1].T)
            self.db[i] = (1 / m) * np.sum(self.dZ[i], axis=1, keepdims=True)

    # update weights and biases using gradient descent
    def update_params(self):
        # for each layer
        for i in range(1, len(self.n)):
            self.W[i] = self.W[i] - (self.learning_rate * self.dW[i])
            self.b[i] = self.b[i] - (self.learning_rate * self.db[i])

    def fit(self):        
        for epoch in range(self.epochs + 1):
            # prediction
            output = self.forward_propagation()

            # loss function
            cost = self.cross_entropy()

            # back propagation
            self.back_propagation()

            # update weights and biases
            self.update_params()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Cost = {cost:.5f}')

    def predict(self, X_test):
        # set the input data
        self.A[0] = X_test

        # calculate probabilities for each test data
        probabilities = self.forward_propagation()

        # prediction
        predictions = np.argmax(probabilities, axis=0)

        return predictions

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

# init model params
learning_rate = 0.76
epochs = 500

# create the model
nn = neural_network(X_train, y_train, neurons, learning_rate, epochs)
print(f'Learning Rate: {learning_rate}, Epochs: {epochs}')
nn.fit()
print()