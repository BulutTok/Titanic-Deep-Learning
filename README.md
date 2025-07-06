
# Titanic Survival Prediction

A complete walkthrough of downloading the Titanic dataset, preprocessing it with scikit-learn pipelines, training a simple TensorFlow/Keras neural network, and interpreting its results.

---

## Table of Contents

1. Project Overview
2. Directory Structure
3. Installation & Dependencies
4. Data Download & Schema
5. Preprocessing Pipeline
6. Model Architecture
7. Training & Validation
8. Final Evaluation
9. Results Interpretation
10. Licence
11. Acknowledgements
12.  Contact

---

## 1. Project Overview

This project demonstrates end-to-end machine learning on the Titanic passenger dataset:

* **Download**: Fetch `train.csv` and `test.csv` if missing
* **Preprocess**:

  * Impute and scale numerical features
  * Impute and one-hot encode categorical features
* **Model**: A two-layer feed-forward neural network
* **Train**: 40 epochs with an 80/20 train/validation split
* **Evaluate**: Accuracy, confusion matrix, and classification report

---

## 2. Directory Structure

```
titanic-survival-prediction/
├── README.md
├── requirements.txt
│── train.csv
│── test.csv
└── Titanic_Survival_Prediction_.ipynb    # Main script containing all code
```

---

## 3. Installation & Dependencies

1. Clone the repo and enter its folder

   ```bash
   git clone https://github.com/<username>/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. (Optional) Create a virtual environment

   ```bash
   python3 -m venv venv  
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install required packages

   ```bash
   pip install -r requirements.txt
   ```

The main dependencies are:

* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow`

---

## 4. Data Download & Schema

When you run the pipeline, it invokes:

```python
# download function
fetch_titanic_data()
```

This checks for and downloads `train.csv` and `test.csv` into `datasets/titanic/`.
Output:

```
Downloading train.csv
Downloading test.csv
```

indicates both files were fetched successfully.

**Column descriptions**

* `PassengerId`: unique passenger identifier
* `Survived`: target (0 = did not survive, 1 = survived)
* `Pclass`: passenger class (1, 2, 3)
* `Name`, `Sex`, `Age`
* `SibSp`, `Parch`: family aboard
* `Ticket`, `Fare`, `Cabin`, `Embarked`

---

## 5. Preprocessing Pipeline

We build two pipelines and combine them with `ColumnTransformer`:

```python
# numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False)),
])

# full preprocessing
preprocess_pipeline = ColumnTransformer([
    ("num", num_pipeline, ["Age", "SibSp", "Parch", "Fare"]),
    ("cat", cat_pipeline, ["Pclass", "Sex", "Embarked"]),
])
```

Applying `fit_transform` to the training data:

```python
X_train = preprocess_pipeline.fit_transform(train_data)
```

Produced array shape `(891, 12)`, meaning:

* **891** rows (passengers)
* **12** features: 4 scaled numeric + 8 one-hot columns

This confirms that missing ages/fares were imputed and categorical variables were converted to dummy variables.

---

## 6. Model Architecture

We use a simple Keras `Sequential` model:

```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),   # 12 input features
    Dense(100, activation="relu"),      # hidden layer
    Dense(1, activation="sigmoid")      # binary output
])
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

* **100 ReLU units** give the model capacity to learn non-linear interactions
* **Sigmoid output** is suited for binary survival prediction

---

## 7. Training & Validation

We split off 20% of the data for validation (stratified to preserve survival ratio), then train:

```python
history = model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=40,
    batch_size=32,
    verbose=2
)
```

**Key training observations**

* **Epoch 1**: Training accuracy \~ 64%, validation \~ 71%
* **Epoch 5**: Training \~ 83%, validation \~ 79%
* **Epoch 10–20**: Training stabilizes around 83–84%, validation around 82–81%
* **Epoch 40**: Training \~ 84.7%, validation \~ 79.3%

This pattern shows:

* Rapid learning in the first 5 epochs
* Validation accuracy peaks around epoch 9 (≈82.7%)
* Slight gap (≈4–5%) between train/validation indicates mild overfitting

---

## 8. Final Evaluation

After training, we evaluate on the hold-out validation set:

```python
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")
```

Result:

```
Validation accuracy: 0.7933
```

We then compute the confusion matrix and classification report:

```python
y_pred = (model.predict(X_val) >= 0.5).astype(int)
confusion_matrix(y_val, y_pred)
classification_report(y_val, y_pred, digits=3)
```

Outputs:

* **Confusion matrix**

  ```
  [[102   8]
   [ 29  40]]
  ```

  * 102 true negatives (correctly predicted “did not survive”)
  * 40 true positives (correctly predicted “survived”)
  * 8 false positives (predicted survived but did not)
  * 29 false negatives (predicted did not survive but did)

* **Classification report**

  |        Class | Precision | Recall | F1-score | Support |
  | -----------: | --------: | -----: | -------: | ------: |
  |            0 |     0.779 |  0.927 |    0.846 |     110 |
  |            1 |     0.833 |  0.580 |    0.684 |      69 |
  |     Accuracy |           |        |    0.793 |     179 |
  |    Macro avg |     0.806 |  0.753 |    0.765 |     179 |
  | Weighted avg |     0.800 |  0.793 |    0.784 |     179 |

---

## 9. Results Interpretation

1. **Overall Accuracy (79.33%)**
   The model correctly classifies nearly 8 out of 10 passengers.

2. **Class Imbalance**

   * 0 (“did not survive”) makes up \~62% of the validation set (110/179).
   * 1 (“survived”) is \~38% (69/179).

3. **Recall vs. Precision**

   * For **class 0**, recall is very high (0.927), meaning most non-survivors are caught.
   * For **class 1**, recall drops to 0.580, so the model misses about 42% of survivors.
   * Precision for both classes (\~0.78–0.83) shows balanced false positive rates.

4. **Overfitting**
   A small train/validation accuracy gap (\~5%) suggests mild overfitting. Strategies to improve might include:

   * Adding dropout
   * Early stopping on validation accuracy
   * Collecting or engineering more features

5. **Next Steps**

   * Experiment with deeper networks or different architectures
   * Tune hyperparameters (learning rate, batch size, number of units)
   * Incorporate additional features (e.g., title extracted from name)

---



## 10. License

Distributed under the MIT License.
See LICENSE file for more information.

## 11. Acknowledgements

• Dataset originated from the Kaggle Titanic Challenge ([https://www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)).
• Download script points to the mirror hosted in the companion repo of Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron.

## 12. Contact

For questions, suggestions, or contributions, please contact:
Bulut Tok
Email: [buluttok2013@gmail.com](mailto:buluttok2013@gmail.com)

