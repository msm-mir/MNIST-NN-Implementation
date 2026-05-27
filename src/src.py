import pandas as pd
import numpy as np

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

W1 = np.random.randn(128, X_train.shape[0]) * 0.01
b1 = np.zeros((128, 1))

W2 = np.random.randn(10, 128) * 0.01
b2 = np.zeros((10, 1))