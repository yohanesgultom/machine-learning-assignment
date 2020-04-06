"""
# Logistic regression

Command: `python logreg.py`

Result will be the attributes and predictions at the end of each lines followed by Accuracy & Execution Time:

```
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: vgood (True)
Accuracy: 63.7426900585%
Execution Time: 0.186897993088s
```

@author yohanes.gultom@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linreg import gradient_descent

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def encode_cols(_df, encoding):
    df = _df.copy()
    for col, encoding_map in encoding.items():
        df.replace(encoding_map, inplace=True) 
    return df   

def encode_object_cols(_df, label_index=None):
    df = _df.copy()
    encoding = {}
    label = df.columns[label_index] if label_index else None
    for col in df.columns:
        if df.dtypes[col] == 'object':
            cats = df[col].astype('category').cat.categories.tolist()
            c = 1 if label != col else 0
            codes = list(range(c, len(cats)+c))
            encoding_map = {col : {k: v for k,v in zip(cats, codes)}}
            df.replace(encoding_map, inplace=True)
            encoding[col] = encoding_map
    return df, encoding

if __name__ == "__main__":
    # read training and testing data
    train_file = 'playtennis.data'
    test_file = 'playtennis.test'
    train_raw_df = pd.read_csv(train_file)
    test_raw_df = pd.read_csv(test_file)
    train_df, encoding = encode_object_cols(train_raw_df, label_index=-1)
    test_df = encode_cols(test_raw_df, encoding)
    
    # training
    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values

    # Add a column of ones to X (interception data)
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    m, n = X.shape

    # Initialize theta parameters
    theta = np.zeros(shape=(n, 1))

    # Some gradient descent settings
    iterations = 400
    alpha = 0.2

    # compute and display initial cost
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations, compute_cost=compute_cost)

    # plot MSE
    plt.figure(2)
    plt.plot(np.arange(J_history.shape[0]), J_history, c='r')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost over Iteration')

    # test
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)    
    y_pred = np.around(X_test.dot(theta).flatten())
    acc = np.sum(y_pred == y_test) / len(y_pred) * 100.0   

    for i, row in test_raw_df.iterrows():
        print(f"{row.values} = {'Yes' if y_pred[i] == 1 else 'No'}")
    print(f'\nAccuracy: {acc}%')

    plt.show()
