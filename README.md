# Titanic Survival Prediction 

*A hands‑on demonstration of an end‑to‑end deep‑learning pipeline with the classic Titanic dataset.*

---

## Repository Contents

| Path                                 | Description                                                                                                                                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Titanic_Survival_Prediction_.ipynb` | Jupyter notebook that downloads the raw data, performs feature engineering & preprocessing with **scikit‑learn** pipelines, trains a small **TensorFlow/Keras** neural‑network classifier, and evaluates it with accuracy, confusion‑matrix & classification‑report. |
| `datasets/`                          | Created automatically at first run; stores the CSV files fetched from the original dataset source.                                                                                                                                                                   |
| `environment.yml` *(optional)*       | Conda environment file you can create with `conda env export > environment.yml` once the project runs on your machine.                                                                                                                                               |

---

##  Quick Start

```bash
# 1️⃣  Clone the repo
git clone https://github.com/<your-user>/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2️⃣  (Recommended) create an isolated environment
conda create -n titanic python=3.11
conda activate titanic

# 3️⃣  Install requirements
pip install -r requirements.txt        # or see the list below

# 4️⃣  Launch the notebook
jupyter lab Titanic_Survival_Prediction_.ipynb
```

The notebook is **fully self‑contained**: the first code cell downloads the Kaggle‑style CSVs directly from Aurélien Géron’s public repository, so no manual data download is needed.

---

##  Requirements

| Package        | Tested Version |
| -------------- | -------------- |
| `python`       | ≥ 3.10         |
| `pandas`       | ≥ 2.2          |
| `numpy`        | ≥ 1.26         |
| `scikit‑learn` | ≥ 1.4          |
| `tensorflow`   | ≥ 2.16         |
| `jupyterlab`   | ≥ 4.2          |

*Tip:* if you are new to Deep Learning and want a smaller install, swap `tensorflow` for `tensorflow‑cpu`.

---

## Workflow Highlights

1. **Data loading** – CSVs are read into Pandas DataFrames; `PassengerId` is set as index.
2. **Pre‑processing pipeline**

   * Numerical columns ⟶ `SimpleImputer(strategy="median")`
   * Categorical columns ⟶ `SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")`
   * `ColumnTransformer` stitches everything together.
3. **Model** – Sequential Keras NN

   * `Input` layer → dense‑ReLU → dense‑ReLU → sigmoid output.
4. **Training & validation** – 20 % hold‑out split.
5. **Evaluation** – Confusion matrix & `classification_report` (precision / recall / F1).

Feel free to tweak the architecture or replace it with gradient‑boosting, random forests, etc.—the preprocessing pipeline stays valid.

---

## 📊 Results

### Training run snapshot (40 epochs)

```
accuracy:      0.8469    loss: 0.3748
val_accuracy: 0.7933    val_loss: 0.4359
```

### Validation‑set performance (179 passengers)

|              | **Predicted 0** | **Predicted 1** |
| ------------ | --------------: | --------------: |
| **Actual 0** |         **102** |               8 |
| **Actual 1** |              29 |          **40** |

* **Overall accuracy:** **79.3 %**
* **Class metrics**

  * *Did not survive (0)* – precision 0.779, recall 0.927, F1 0.846
  * *Survived (1)* – precision 0.833, recall 0.580, F1 0.684
  * Macro‑average F1 0.765

### Most informative features

`Sex_female`, `Pclass_3`, `Fare`, `Age`, `Embarked_S`
*(see the notebook’s final section for permutation‑importance & coefficient plots)*

---



##  Contributing

1. Fork the repo & create your feature branch (`git checkout -b feat/amazing-idea`)
2. Commit your changes (`git commit -m 'Add amazing idea'`)
3. Push to the branch (`git push origin feat/amazing-idea`)
4. Open a pull request

Please follow [**PEP 8**](https://peps.python.org/pep-0008/) and include unit tests where reasonable.

---

##  License

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

