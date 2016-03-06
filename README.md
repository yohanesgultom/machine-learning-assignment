# Machine Learning Assignment
Collection of machine learning algorithms with open dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

### Prerequisites (Python)

1. [Python](https://www.python.org/downloads/) >= 2.7
2. [PIP](https://pip.pypa.io/en/stable/installing/) >= 1.5.6


## Decision Tree ID3 (Python)
Generative learning decision tree based on Entropy (H) and Information Gain (IG)

#### Installation

If you haven't done it before, run this commands inside the cloned directory:
```
sudo pip install numpy
sudo pip install Graphviz
sudo pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709
sudo pip install pydot
```

#### Run Prediction with Sample Data

Run this commands inside the cloned directory after successful installation:
```
python id3.py car.data car.test
```

Result will be the attributes and predictions at the end of each lines followed by **Accuracy** & **Execution Time**:
```
{....}
{'LugBoot': 'big', 'Maint': 'low', 'Persons': '4', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': '4', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: vgood (True)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (True)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: vgood (True)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: vgood (True)
Accuracy: 77.1929824561%
Execution Time: 0.871682882309s
```
Add option `--plot` to plot the decision tree in png format:
```
python id3.py car.data car.test --plot
```

The result is saved in the same directory with file name pattern `[data_file_name].png` eg. car.data.png:

![decision tree](https://github.com/yohanesgultom/machine-learning-assignment/blob/master/car.data.png)

## Naive Bayes Classifier (Python)
Classifier using Naive Density Estimator

#### Installation

If you haven't done it before, run this commands inside the cloned directory:
```
sudo pip install numpy
```

#### Run Prediction with Sample Data

Run this commands inside the cloned directory after successful installation:
```
python nbc.py car.data car.test
```

Result will be the attributes and predictions at the end of each lines followed by **Accuracy** & **Execution Time**:
```
{....}
{'LugBoot': 'big', 'Maint': 'low', 'Persons': '4', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': '4', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (True)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Buying': 'low'} Quality: acc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Buying': 'low'} Quality: vgood (True)
Accuracy: 63.7426900585%
Execution Time: 0.186897993088s
```

## Join-Distribution Bayes Classifier (Python)
Classifier using Join-Distribution Density Estimator

#### Installation

If you haven't done it before, run this commands inside the cloned directory:
```
sudo pip install numpy
```

#### Run Prediction with Sample Data

Run this commands inside the cloned directory after successful installation:
```
python jbc.py car.data car.test
```

Result will be the attributes and predictions at the end of each lines followed by **Accuracy** & **Execution Time**:
```
{....}
{'LugBoot': 'big', 'Maint': 'low', 'Persons': '4', 'Safety': 'med', 'Doors': '5more', 'Quality': 'good', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': '4', 'Safety': 'high', 'Doors': '5more', 'Quality': 'vgood', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Quality': 'unacc', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Quality': 'acc', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'small', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Quality': 'good', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Quality': 'unacc', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Quality': 'good', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'med', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Quality': 'vgood', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'low', 'Doors': '5more', 'Quality': 'unacc', 'Buying': 'low'} Quality: unacc (True)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'med', 'Doors': '5more', 'Quality': 'good', 'Buying': 'low'} Quality: unacc (False)
{'LugBoot': 'big', 'Maint': 'low', 'Persons': 'more', 'Safety': 'high', 'Doors': '5more', 'Quality': 'vgood', 'Buying': 'low'} Quality: unacc (False)
Accuracy: 55.5555555556%
Execution Time: 0.0280420780182s
```
