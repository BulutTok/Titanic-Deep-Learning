# Titanic Survival Prediction 




This repository provides a complete example of downloading the Titanic dataset, preprocessing the data with scikit-learn pipelines, training a simple neural network in TensorFlow/Keras, and evaluating its performance.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Directory Structure](#directory-structure)    
3. [Usage](#usage)  
4. [Preprocessing Pipeline](#preprocessing-pipeline)  
5. [Model Architecture](#model-architecture)  
6. [Results](#results)  
7. [License & Contact](#license--contact)  

---

## Project Overview

We fetch the official Titanic `train.csv` and `test.csv` files, build a preprocessing pipeline that imputes missing values and one-hot encodes categorical features, train a small feed-forward neural network to predict survival, and report accuracy, confusion matrix, and classification metrics.

---

## Directory Structure

```

.
├── README.md
├── requirements.txt
├── datasets/
│   └── titanic/
│       ├── train.csv
│       └── test.csv
└── titanic\_pipeline.py         # Main script/notebook with all code

````

---



## Usage

1. **Fetch the data**
   The script will automatically download `train.csv` and `test.csv` into `datasets/titanic/` if they are not already present.

   ```bash
   python titanic_pipeline.py
   ```

2. **Run the notebook or script**
   This will:

   * Load and preprocess the data
   * Split off a validation set (20%)
   * Train a simple neural network for 40 epochs
   * Print validation accuracy, confusion matrix, and a classification report

---

## Preprocessing Pipeline

* **Numerical features** (`Age`, `SibSp`, `Parch`, `Fare`):
  – Impute missing values with the median
  – Standardize to zero mean & unit variance

* **Categorical features** (`Pclass`, `Sex`, `Embarked`):
  – Impute missing values with the most frequent category
  – One-hot encode (no sparse output)

Combined via `sklearn.compose.ColumnTransformer` into a single feature matrix.

---

## Model Architecture

A simple TensorFlow/Keras `Sequential` model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential([
    Input(shape=(n_features,)),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

Trained for 40 epochs with batch size 32, tracking validation performance each epoch.

---

## Results

* **Validation accuracy:** \~0.7933
* **Confusion matrix:**

  ```
  [[102   8]
   [ 29  40]]
  ```
* **Classification report:**

  ```
                precision    recall  f1-score   support

           0      0.779     0.927     0.846       110
           1      0.833     0.580     0.684        69

     accuracy                          0.793       179
    macro avg      0.806     0.753     0.765       179
  ```

weighted avg      0.800     0.793     0.784       179

```

---



##  License

---
Distributed under the **MIT License**.
See `LICENSE` file for more information.

---

### ✨ Acknowledgements

* Dataset originated from the **[Kaggle Titanic Challenge](https://www.kaggle.com/competitions/titanic)**.
* Download script points to the mirror hosted in the companion repo of *Hands‑On Machine Learning with Scikit‑Learn, Keras & TensorFlow* by **Aurélien Géron**.


## Contact

For questions, suggestions, or contributions, please contact:

**Bulut Tok**  
Email: [buluttok2013@gmail.com]

