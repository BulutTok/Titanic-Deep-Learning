# Titanic Deep Learning

This repository contains a Python project that leverages deep learning techniques to predict the survival of Titanic passengers. Using TensorFlow and Keras, the project preprocesses the data, builds a simple neural network model, and generates predictions formatted for Kaggle submission.

## Overview

The objective is to predict whether a Titanic passenger survived based on features such as age, sex, passenger class, number of siblings/spouses aboard, fare paid, and more. The dataset is split into a training set (with survival labels) and a test set (without labels). The predictions from the deep learning model are written to a CSV file for submission to Kaggle.

## Data

The project automatically downloads the Titanic dataset from:
```
https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/
```
The following CSV files are used:
- **train.csv**: Contains training data with labels (Survived column).
- **test.csv**: Contains test data without labels.

**Key Attributes:**
- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival outcome (0 = did not survive, 1 = survived).
- **Pclass**: Passenger class.
- **Name**, **Sex**, **Age**: Passenger details.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Ticket fare (in pounds).
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation.

## Project Workflow

1. **Data Fetching:**
   - The script downloads the dataset if it’s not already available in the `datasets/titanic` directory.

2. **Data Loading:**
   - The CSV files are loaded into Pandas DataFrames.
   - The `PassengerId` column is set as the index for both training and test datasets.

3. **Preprocessing:**
   - **Numerical Pipeline:**
     - Attributes: `Age`, `SibSp`, `Parch`, `Fare`
     - Steps: Impute missing values using the median and scale the features.
   - **Categorical Pipeline:**
     - Attributes: `Pclass`, `Sex`, `Embarked`
     - Steps: Impute missing values using the most frequent value and apply one-hot encoding.
   - The numerical and categorical pipelines are combined using a `ColumnTransformer`.

4. **Deep Learning Model:**
   - A simple neural network is built using TensorFlow’s Keras API:
     - **Layer 1:** Dense layer with 100 neurons and ReLU activation.
     - **Output Layer:** Dense layer with 1 neuron and sigmoid activation.
   - The model uses binary crossentropy as the loss function, the Adam optimizer, and tracks accuracy.
   - The model is trained for 40 epochs on the preprocessed training data.

5. **Prediction and Submission:**
   - The model predicts survival probabilities for the test set.
   - Predictions are converted to binary outcomes (1 if probability ≥ 0.5, else 0).
   - The results are saved in a CSV file named `submission_dl.csv`, containing `PassengerId` and `Survived` columns.

## Requirements

- **Python 3.x**
- **Pandas**
- **Scikit-learn**
- **TensorFlow** (includes Keras)
- **Urllib** (standard library)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/titanic-deep-learning.git
   cd titanic-deep-learning
   ```

2. **Install the Required Packages:**
   ```bash
   pip install pandas scikit-learn tensorflow
   ```

3. **Run the Script:**
   - You can run the script directly from the command line:
     ```bash
     python titanic_deep_learning.py
     ```
   - Alternatively, you can execute the code in a Jupyter Notebook or Google Colab environment.

## Usage

- **Data Download:**  
  The script automatically downloads `train.csv` and `test.csv` into the `datasets/titanic` folder if they are not already present.

- **Preprocessing and Training:**  
  The preprocessing pipelines transform numerical and categorical attributes. The deep learning model is then trained using the processed training data.

- **Prediction:**  
  After training, the model predicts survival outcomes on the test set. These predictions are converted into binary values and saved in `submission_dl.csv`.

## Results

- The model’s architecture is summarized after training.
- A CSV file (`submission_dl.csv`) is generated, which contains the final predictions for submission to Kaggle. This file includes:
  - `PassengerId`
  - `Survived` (predicted outcome)

## Contact

For questions, suggestions, or contributions, please contact:

**Bulut Tok**  
Email: [buluttok2013@gmail.com](mailto:buluttok2013@gmail.com)


###Acknowledgments
Thanks to Ageron for providing the initial datasets.
Kaggle for the Titanic machine learning competition platform.
