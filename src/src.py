import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

class neural_network:
    def __init__(self, neurons, learning_rate, epochs, keep_n_prob, optimizer):
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
        # for adam algorithm
        self.V_dW = {}
        self.V_db = {}
        self.S_dW = {}
        self.S_db = {}

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.keep_n_prob = keep_n_prob
        self.optimizer = optimizer
        self.update_steps_cnt = 0

        self.accuracy_history = []
        self.loss_history = []

        if optimizer == 'gd': self.init_gd_W_b()
        else: self.init_adam_W_b()
    
    # weights and biases initialization for gd algorithm
    def init_gd_W_b(self):
        np.random.seed(42)

        # start from 1st layer to the last layer that we've set
        for i in range(1, len(self.n)):
            self.W[i] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            self.b[i] = np.zeros((self.n[i], 1))
    
    # weights and biases initialization for adam algorithm
    def init_adam_W_b(self):
        for i in range(1, len(self.n)):
            self.V_dW[i] = np.zeros_like(self.W[i])
            self.V_db[i] = np.zeros_like(self.b[i])
            self.S_dW[i] = np.zeros_like(self.W[i])
            self.S_db[i] = np.zeros_like(self.b[i])

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

    # update weights and biases using gd algorithm
    def update_gd_params(self):
        # for each layer
        for i in range(1, len(self.n)):
            self.W[i] -= (self.learning_rate * self.dW[i])
            self.b[i] -= (self.learning_rate * self.db[i])

    # update weights and biases using adam algorithm
    def update_adam_params(self):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        self.update_steps_cnt += 1
            
        for i in range(1, len(self.n)):
            # calculate momentum
            self.V_dW[i] = (beta1 * self.V_dW[i]) + ((1 - beta1) * self.dW[i])
            self.V_db[i] = (beta1 * self.V_db[i]) + ((1 - beta1) * self.db[i])
            
            # calculate RMSProp
            self.S_dW[i] = (beta2 * self.S_dW[i]) + ((1 - beta2) * (self.dW[i] ** 2))
            self.S_db[i] = (beta2 * self.S_db[i]) + ((1 - beta2) * (self.db[i] ** 2))
            
            # bias correction
            V_dW_corrected = self.V_dW[i] / (1 - (beta1 ** self.t))
            V_db_corrected = self.V_db[i] / (1 - (beta1 ** self.t))
            S_dW_corrected = self.S_dW[i] / (1 - (beta2 ** self.t))
            S_db_corrected = self.S_db[i] / (1 - (beta2 ** self.t))
            
            # final update
            self.W[i] -= (self.learning_rate * V_dW_corrected) / (np.sqrt(S_dW_corrected) + epsilon)
            self.b[i] -= (self.learning_rate * V_db_corrected) / (np.sqrt(S_db_corrected) + epsilon)

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

    def train_step(self, mini_batch, is_training):
        # split mini batch to X, y
        mini_batch_X, mini_batch_y = mini_batch

        # one-hot encode the labels of y batch
        y_oh = self.init_y_one_hot(mini_batch_y)

        # prediction by forward propagation
        self.forward_propagation(mini_batch_X, is_training)

        # loss function
        cost = self.cross_entropy(y_oh)
        epoch_cost += cost

        # back propagation
        self.back_propagation(y_oh)

        # update weights and biases
        if self.optimizer == 'adam': self.update_adam_params()
        else: self.update_gd_params()

        return epoch_cost, 

    def fit(self, X, y, batch_size, is_training, patience):
        # initialization for early stopping
        best_cost = float('inf1')
        patience_cnt = 0

        for epoch in range(1, self.epochs + 1):
            # create mini batches for this epoch
            mini_batches = self.create_mini_batches(X, y, batch_size)
            epoch_cost = 0

            for mini_batch in mini_batches:
                epoch_cost,  = self.train_step(mini_batch, is_training)

            # store accuracy in each epoch for plotting
            predictions = self.predict(X, is_training)
            current_accuracy = self.accuracy(y, predictions)
            self.accuracy_history.append(current_accuracy)

            # store loss in each epoch for plotting
            avg_epoch_cost = epoch_cost / len(mini_batches)
            self.loss_history.append(avg_epoch_cost)

            if epoch == self.epochs:
                print(f"Last Epoch ({epoch})'s Cost: {avg_epoch_cost:.5f}\n")

            if avg_epoch_cost < best_cost:
                best_cost = avg_epoch_cost
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                print(f'\nEarly stopping triggered at epoch {epoch} with best cost: {best_cost:.5f}\n')
                break
    
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
    
    # advanced analyzing
    errors_matrix = cm.copy()
    np.fill_diagonal(errors_matrix, 0)
    print("\nAdvanced Confusion Matrix Analysis")
    top_error_indices = np.argsort(errors_matrix.flatten())[-3:][::-1]
    for idx in top_error_indices:
        true_digit = idx // 10
        pred_digit = idx % 10
        error_count = errors_matrix[true_digit, pred_digit]
        
        print(f"Digit '{true_digit}' was mistakenly predicted as '{pred_digit}' ({error_count} times)")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix on {set_name} Set')
    plt.show()

def evaluation_plotting(model, X, y, is_training, set_name):
    predictions = model.predict(X, is_training)
    accuracy_plot(model, y, predictions, set_name)
    loss_plot(model, y, set_name)
    confusion_matrix_plot(y, predictions, set_name)
    print()

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

# init model params
# number of neurons for each layer (input, hidden, output)
neurons = {0: 784, 1: 128, 2: 10}
learning_rate = 0.8, epochs = 120,  keep_neuron_prob = 0.8, optimizer = 'adam'

# create the model
nn = neural_network(neurons, learning_rate, epochs, keep_neuron_prob, optimizer)

# init fit params
batch_size = 128, patience = 5

# training the model
print('Starting training...\n')
print(f'Learning Rate: {learning_rate}')
nn.fit(X_train, y_train, batch_size, patience)
print('Training completed!\n')

# save model parameters to a file
nn.save_W_and_b()

# evaluation plotting for training set
evaluation_plotting(nn, X_train, y_train, True, 'Training')

# create new model and load previous model parameters from a file
fresh_nn = neural_network(neurons, learning_rate, epochs, keep_neuron_prob)
fresh_nn.load_W_and_b()

# evaluation plotting for test set
evaluation_plotting(fresh_nn, X_test, y_test, False, 'Test')