## Machine Learning Assignment
Collection of machine learning algorithms with open dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

### ID3 (Python)
Generative learning decision tree based on Entropy (H) and Information Gain (IG)

#### Prerequisites

1. [Python](https://www.python.org/downloads/) >= 2.7
2. [PIP](https://pip.pypa.io/en/stable/installing/) >= 1.5.6

#### Installation

Run this commands inside the cloned directory:
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

Result will be the attributes and predictions at the end of each lines
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
Execution Time: 0.0362010002136s
```
