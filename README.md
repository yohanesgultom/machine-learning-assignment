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
```

#### Run Prediction with Sample Data

Run this commands inside the cloned directory after successful installation:
```
python id3.py car.data car.test
```

Result will be the attributes and predictions at the end of each lines
```
{....}

{'maint': 'vhigh', 'persons': '2', 'lug_boot': 'small', 'safety': 'low', 'doors': '2', 'buying': 'vhigh'} quality: unacc
{'maint': 'vhigh', 'persons': '2', 'lug_boot': 'small', 'safety': 'med', 'doors': '2', 'buying': 'vhigh'} quality: unacc
{'maint': 'med', 'persons': '4', 'lug_boot': 'med', 'safety': 'high', 'doors': '2', 'buying': 'vhigh'} quality: acc
{'maint': 'med', 'persons': '4', 'lug_boot': 'big', 'safety': 'med', 'doors': '2', 'buying': 'vhigh'} quality: acc
{'maint': 'med', 'persons': '4', 'lug_boot': 'big', 'safety': 'high', 'doors': '2', 'buying': 'vhigh'} quality: acc
{'maint': 'low', 'persons': '4', 'lug_boot': 'med', 'safety': 'med', 'doors': '5more', 'buying': 'med'} quality: good
{'maint': 'low', 'persons': 'more', 'lug_boot': 'small', 'safety': 'high', 'doors': '5more', 'buying': 'med'} quality: good
{'maint': 'low', 'persons': 'more', 'lug_boot': 'big', 'safety': 'med', 'doors': '5more', 'buying': 'med'} quality: good
{'maint': 'low', 'persons': '4', 'lug_boot': 'med', 'safety': 'high', 'doors': '5more', 'buying': 'med'} quality: vgood
{'maint': 'low', 'persons': '4', 'lug_boot': 'big', 'safety': 'high', 'doors': '5more', 'buying': 'med'} quality: vgood
{'maint': 'low', 'persons': 'more', 'lug_boot': 'med', 'safety': 'high', 'doors': '5more', 'buying': 'med'} quality: vgood
{'maint': 'low', 'persons': 'more', 'lug_boot': 'big', 'safety': 'high', 'doors': '5more', 'buying': 'med'} quality: vgood
```

