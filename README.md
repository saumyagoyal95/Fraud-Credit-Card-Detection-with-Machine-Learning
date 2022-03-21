# Fraud-Credit-Card-Detection-with-Machine-Learning

<div align="center">

# Fraud-Credit-Card-Detection-with-Machine-Learning

[About](#about) •
[Configuration Requirements](#configuration-requirements) •
[Findings](#installation) •
  
</div>

## 📒 About <a name="about"></a>

Created a project on the Kaggle dataset for detecting the Credit card fraud with the help of Machine learning algorithms and techniques. I have tried using various algorithms and finally came up with one best model. It is an extensive Jupyter Notebook which includes all the background work I did.

## 👨‍💻 Configuration Requirements <a name="configuration-requirements"></a>

What is the required configuration for running this code
1. Jupyter Notebook
2. If you have less GPU support, try running the notebook on Google colab
3. Import statements includes

```python
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
StdScaler = StandardScaler()
Pt = PowerTransformer()

from sklearn import metrics
from sklearn import preprocessing
```
## 🖥️ Findings <a name="installation"></a>

Best Model that is selected for Detecting the Credit card fraud (after handling the imbalance):
- Model : XGBOOST model with Random Oversampling with StratifiedKFold CV
- XGboost roc_value: 0.9866310433921559
- XGBoost threshold: 0.00020692142425104976

Summary on best model on Imbalanced data :
- Model: Logistic Regression with L2 Regularisation with Repeated KFold Cross Validation
- Accuracy : 0.9987
- ROC: 0.9921

