import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

class neural_network:
    def __init__(self, neurons, learning_rate, epochs, keep_n_prob):
        # weight matrix
        self.W = {}
        # bias matrix (with one column)
        self.b = {}
        # output of linear equation
        self.Z = {}
        # output of activation function
        self.A = {}
        # number of neurons for each layer
        self.n = neurons
        # derivatives of matrices
        self.dW = {}
        self.db = {}
        self.dZ = {}
        # store kept neurons 
        self.keep_n = {}

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.keep_n_prob = keep_n_prob

        self.accuracy_history = []
        self.loss_history = []

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
    def forward_propagation(self, X, is_training):
        self.A[0] = X

        # for each layer
        for i in range(1, len(self.n)):
            # calculate linear equation output (Z = W . A + b)
            self.Z[i] = np.dot(self.W[i], self.A[i - 1]) + self.b[i]
            
            # ReLU activation function for all layers except output layer
            if i != len(self.n) - 1:
                self.A[i] = self.relu(self.Z[i])

                # apply dropout only during training
                if is_training and self.keep_n_prob < 1.0:
                    # generate a dropout mask of the same size as the current layer
                    self.keep_n[i] = (np.random.rand(*self.A[i].shape) < self.keep_n_prob).astype(int)
                    # set the output of dropped neurons to zero
                    self.A[i] = self.A[i] * self.keep_n[i]
                    # inverted dropout scaling
                    self.A[i] = self.A[i] / self.keep_n_prob

            # Sigmoid activation function only for output layer
            else:
                self.A[i] = self.sigmoid(self.Z[i])

        return self.A[len(self.n) - 1]

    # one-hot encode the y matrix
    def init_y_one_hot(self, y):
        m = y.shape[0]
        new_y = np.zeros((10, m))
        new_y[y, np.arange(m)] = 1
        return new_y

    # calculate loss function
    def cross_entropy(self, y_oh):
        m = y_oh.shape[1]
        n = len(self.n) - 1

        # clip output layer values to avoid exact 0 and 1
        A_clipped = np.clip(self.A[n], 1e-15, 1 - 1e-15)

        # calculate the formula of cross entropy
        loss_elements = (y_oh * np.log(A_clipped)) + ((1 - y_oh) * np.log(1 - A_clipped))
        cost = (-1 / m) * np.sum(loss_elements)

        return np.squeeze(cost)
    
    # backward propagation
    def back_propagation(self, y_oh):
        m = y_oh.shape[1]

        # for each hidden layer
        for i in range(len(self.n) - 1, 0, -1):
            if i == len(self.n) - 1:
                self.dZ[i] = self.A[i] - y_oh
            else:
                self.dZ[i] = np.dot(self.W[i + 1].T, self.dZ[i + 1]) * (self.Z[i] > 0)

            # apply dropout mask to the gradients in hidden layers
            if self.keep_n_prob < 1.0:
                self.dZ[i] = self.dZ * self.keep_n[i]
                self.dZ[i] = self.dZ / self.keep_n_prob
            
            self.dW[i] = (1 / m) * np.dot(self.dZ[i], self.A[i - 1].T)
            self.db[i] = (1 / m) * np.sum(self.dZ[i], axis=1, keepdims=True)

    # update weights and biases using gradient descent
    def update_params(self):
        # for each layer
        for i in range(1, len(self.n)):
            self.W[i] = self.W[i] - (self.learning_rate * self.dW[i])
            self.b[i] = self.b[i] - (self.learning_rate * self.db[i])

    def predict(self, X, is_training):
        # calculate probabilities for each instance
        output = self.forward_propagation(X, is_training)

        # prediction
        predictions = np.argmax(output, axis=0)

        return predictions

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        m = X.shape[1]

        # shuffle data at each epoch to vary the batches
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_y = y[permutation]

        # split data to complete batches
        num_complete_batches = m // batch_size
        for k in range(num_complete_batches):
            mini_batch_X = shuffled_X[:, k * batch_size : (k + 1) * batch_size]
            mini_batch_y = shuffled_y[k * batch_size : (k + 1) * batch_size]
            mini_batches.append(mini_batch_X, mini_batch_y)

        # for remaining data
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_batches * batch_size : ]
            mini_batch_y = shuffled_y[num_complete_batches * batch_size : ]
            mini_batches.append((mini_batch_X, mini_batch_y))

        return mini_batches

    def fit(self, X, y, batch_size, is_training):
        for epoch in range(1, self.epochs + 1):
            # create mini batches for this epoch
            mini_batches = self.create_mini_batches(X, y, batch_size)
            epoch_cost = 0

            for mini_batch in mini_batches:
                # split mini batch to X, y
                mini_batch_X, mini_batch_y = mini_batch

                # one-hot encode the labels of y batch
                y_oh = self.init_y_one_hot(mini_batch_y)

                # prediction by forward propagation
                predictions = self.predict(mini_batch_X, is_training)

                # loss function
                cost = self.cross_entropy(y_oh)
                epoch_cost += cost

                # back propagation
                self.back_propagation(y_oh)

                # update weights and biases
                self.update_params()

            # store accuracy in each epoch for plotting
            predictions = self.predict(X, is_training)
            current_accuracy = self.accuracy(y, predictions)
            self.accuracy_history.append(current_accuracy)

            # store loss in each epoch for plotting
            avg_epoch_cost = epoch_cost / len(mini_batches)
            self.loss_history.append(avg_epoch_cost)

            if epoch == self.epochs:
                print(f"Last Epoch ({epoch})'s Cost: {avg_epoch_cost:.5f}\n")
    
    def accuracy(self, y, predictions):
        # real outputs vs prediction
        correct_predictions = (predictions == y)

        # calculate accuracy evaluation
        accuracy = np.mean(correct_predictions) * 100

        return accuracy

    def loss(self, y):
        # broadcast y matrix to A's dimensions
        y_oh = self.init_y_one_hot(y)

        # calculate loss evaluation
        loss = self.cross_entropy(y_oh)

        return loss

    def confusion_matrix(self, y, predictions):
        # new (10, 10) matrix with zero values
        confusion_m = np.zeros((10, 10), dtype=int)
    
        # populating the confusion matrix
        for true_label, pred_label in zip(y, predictions):
            confusion_m[true_label, pred_label] += 1
            
        return confusion_m

    # save trained weights and biases in a file
    def save_W_and_b(self):
        model_params = {'W': self.W, 'b': self.b, 'n': self.n}
        with open("weights_and_biases.pkl", 'wb') as f:
            pickle.dump(model_params, f)

    # load trained weights and biases in a file
    def load_W_and_b(self):
        with open("weights_and_biases.pkl", 'rb') as f:
            model_params = pickle.load(f)
        
        self.W = model_params['W']
        self.b = model_params['b']
        self.n = model_params['n']

def accuracy_plot(model, y, predictions, set_name):
    accuracy = model.accuracy(y, predictions)
    print(f'Accuracy on {set_name} Set: {accuracy:.2f}%')

    plt.figure(figsize=(7, 5))
    plt.plot(model.accuracy_history, label=f'{set_name} Accuracy')
    plt.title(f'Model Accuracy over Epochs on {set_name} Set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()

def loss_plot(model, y, set_name):
    loss = model.loss(y)
    print(f'Loss on {set_name} Set: {loss:.2f}')

    plt.figure(figsize=(7, 5))
    plt.plot(model.loss_history, label=f'{set_name} Loss')
    plt.title(f'Model Loss over Epochs on {set_name} Set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def confusion_matrix_plot(y, predictions, set_name):
    cm = confusion_matrix(y, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix on {set_name} Set')
    plt.show()

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
learning_rate = 0.8
epochs = 120
batch_size = 128
keep_neuron_prob = 0.8

# create the model
nn = neural_network(neurons, learning_rate, epochs, keep_neuron_prob)

# training the model
print('Starting training...\n')
print(f'Learning Rate: {learning_rate}')
nn.fit(X_train, y_train, batch_size)
print('Training completed!\n')

# save model parameters to a file
nn.save_W_and_b()

# evaluation plotting for training set
predictions = nn.predict(X_train, True)
accuracy_plot(nn, y_train, predictions, 'Training')
loss_plot(nn, y_train, 'Training')
confusion_matrix_plot(y_train, predictions, 'Training')

print()

# create new model and load previous model parameters from a file
fresh_nn = neural_network(neurons, learning_rate, epochs, keep_neuron_prob)
fresh_nn.load_W_and_b()

# evaluation plotting for test set
predictions = fresh_nn.predict(X_test, False)
accuracy_plot(fresh_nn, y_test, predictions, 'Test')
loss_plot(fresh_nn, y_test, 'Test')
confusion_matrix_plot(y_test, predictions, 'Test')

print()