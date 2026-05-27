import pandas as pd

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
X_train = X_train / 255
X_test = X_test / 255
