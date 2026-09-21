# Module 2 — Analytics Pipeline

## Overview

This module implements an end-to-end Titanic analytics and predictive modeling workflow.

The pipeline covers:

* Dataset loading using Seaborn
* Data quality inspection
* Missing-value handling
* Univariate and multivariate exploratory data analysis
* Outlier detection using the IQR method
* Descriptive statistics and skewness analysis
* Correlation analysis
* Classification using Logistic Regression, Decision Tree, and Random Forest
* Class-imbalance handling using class weighting and SMOTE
* Hyperparameter tuning using GridSearchCV
* Random Forest out-of-bag evaluation
* Regression using Linear Regression
* Regression error analysis using residual plots
* Model serialization using Joblib

The dataset is loaded once using Seaborn, saved locally as `titanic.csv`, cleaned, and reused for EDA and modeling.

---

## Project Structure

```text
analytics/
│
├── eda.py
├── modeling.py
├── titanic.csv
├── titanic_cleaned.csv
│
├── artifacts/
│   ├── plots/
│   │   ├── age_histogram.png
│   │   ├── age_boxplot.png
│   │   ├── fare_histogram.png
│   │   ├── fare_boxplot.png
│   │   ├── correlation_heatmap.png
│   │   ├── survival_sex_pclass.png
│   │   ├── fare_class_survival.png
│   │   ├── age_fare_survival.png
│   │   ├── class_sex_survival.png
│   │   ├── decision_tree.png
│   │   ├── roc_logistic_regression.png
│   │   ├── roc_decision_tree.png
│   │   ├── roc_random_forest.png
│   │   └── regression_residuals.png
│   │
│   ├── classification_results.csv
│   ├── imbalance_results.csv
│   ├── regression_results.csv
│   └── best_pipeline.joblib
│
└── README.md
```

---

# 1. Dataset Loading

The Titanic dataset is loaded once using:

```python
sns.load_dataset("titanic")
```

The raw dataset is saved locally as:

```text
titanic.csv
```

The cleaned dataset is saved as:

```text
titanic_cleaned.csv
```

---

# 2. Data Inspection and Cleaning

The following dataset inspection methods are used:

```python
df.info()
df.describe()
```

Missing-value percentages are calculated using:

```python
df.isnull().mean() * 100
```

The following threshold-based strategy is implemented:

| Missing Percentage | Treatment             |
| ------------------ | --------------------- |
| `< 5%`             | Drop affected rows    |
| `5% – 30%`         | Impute missing values |
| `> 30%`            | Drop the column       |

For numeric columns, median imputation is used.

For categorical columns, mode imputation is used.

---

# 3. Univariate Analysis

Histograms and boxplots are generated for:

* Age
* Fare

The plots are saved under:

```text
artifacts/plots/
```

Example methods used:

```python
sns.histplot()
sns.boxplot()
```

---

# 4. Outlier Detection

Outliers are detected using the Interquartile Range (IQR) method.

The implementation calculates:

```text
Q1
Q3
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

The method is applied to:

* Age
* Fare

The number of detected outliers is reported in the console.

---

# 5. Descriptive Statistics and Skewness

For the `fare` variable, the following statistics are calculated:

* Mean
* Median
* Mode

The relationship between mean, median, and mode is used to describe the distribution as:

* Right-skewed
* Left-skewed
* Approximately symmetric

---

# 6. Survival Analysis

Survival rates are calculated by:

* Sex
* Passenger class
* Sex and passenger class

Example:

```python
df_clean.groupby("sex")["survived"].mean()
```

Boolean masking is also used to compare specific groups such as first-class male and female passengers.

---

# 7. Correlation Analysis

A correlation matrix is calculated using:

* Survived
* Pclass
* Age
* SibSp
* Parch
* Fare

A correlation heatmap is generated using Seaborn.

The two strongest absolute correlations are also identified programmatically.

Output:

```text
artifacts/plots/correlation_heatmap.png
```

---

# 8. Multivariate Analysis

Four multivariate visualizations are generated.

## 8.1 Survival Rate by Sex and Passenger Class

This chart compares survival rates across passenger sex and passenger class.

The visualization shows differences in survival rates between male and female passengers across the three passenger classes.

It helps examine the combined relationship between sex, passenger class, and survival.

---

## 8.2 Fare by Passenger Class and Survival

This boxplot compares fare distributions across passenger classes and survival status.

Fare distributions differ across passenger classes, with higher passenger classes generally associated with higher fares.

The plot also allows comparison of fare distributions between passengers who survived and those who did not.

---

## 8.3 Age vs Fare by Survival

This scatter plot examines the relationship between passenger age and fare while distinguishing survival status.

The observations are distributed across different age and fare ranges.

The visualization provides a multivariate view of age, fare, and survival rather than considering each variable independently.

---

## 8.4 Class, Sex and Survival

This visualization compares survival rates across passenger classes and sex.

The chart shows that survival outcomes vary across combinations of passenger class and sex.

This provides a combined view of the relationship between class, sex, and survival.

---

# 9. Classification

The target variable is:

```text
survived
```

The classification features are:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

The dataset is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Stratification is used to preserve the class distribution between training and testing sets.

---

# 10. Preprocessing Pipeline

A preprocessing pipeline is implemented using:

* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* `Pipeline`

### Numeric features

Median imputation followed by standardization:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

### Categorical features

Most-frequent imputation followed by one-hot encoding:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

The numeric and categorical transformations are combined using `ColumnTransformer`.

---

# 11. Classification Models

Three classification algorithms are trained:

### Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

### Decision Tree

```python
DecisionTreeClassifier(
    max_depth=5
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200
)
```

---

# 12. Classification Metrics

Each model is evaluated using:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

ROC curves are also generated for all three models.

The generated results are stored in:

```text
artifacts/classification_results.csv
```

---

# 13. Model Comparison

The three models are evaluated using the same test set and metrics.

The implementation records:

```text
Accuracy
Precision
Recall
F1
AUC
```

The final classifier selected by the implementation is the model having the highest F1-score:

```python
classification_df.loc[
    classification_df["F1"].idxmax(),
    "Model"
]
```

The selected pipeline is saved as:

```text
artifacts/best_pipeline.joblib
```

---

# 14. Class Imbalance

Three approaches are compared:

1. Baseline Logistic Regression
2. Class-weighted Logistic Regression
3. Logistic Regression with SMOTE

### Baseline

The normal Logistic Regression model is used without class balancing.

### Class Weighting

The following setting is used:

```python
class_weight="balanced"
```

### SMOTE

Synthetic Minority Oversampling Technique is applied only inside the training pipeline:

```python
SMOTE(random_state=42)
```

The three strategies are compared using:

* Precision
* Recall
* F1-score

Results are saved to:

```text
artifacts/imbalance_results.csv
```

The comparison is intended to show the precision-recall trade-off produced by different imbalance-handling strategies.

---

# 15. Hyperparameter Tuning

Random Forest hyperparameters are optimized using `GridSearchCV`.

The search includes:

```text
n_estimators:
100, 200

max_depth:
None, 5, 10

max_features:
sqrt, log2
```

Five-fold cross-validation is used:

```python
GridSearchCV(
    ...,
    cv=5,
    scoring="f1"
)
```

The best parameters and best cross-validation score are reported.

---

# 16. Random Forest OOB Evaluation

The tuned Random Forest is evaluated using the out-of-bag score.

The model is configured with:

```python
oob_score=True
```

The resulting OOB score is printed after hyperparameter tuning.

---

# 17. Regression

A separate regression task is implemented to predict:

```text
fare
```

Features include:

```text
survived
pclass
age
sibsp
parch
sex
embarked
```

Categorical variables are converted using one-hot encoding.

The regression pipeline uses:

```text
SimpleImputer
      ↓
StandardScaler
      ↓
LinearRegression
```

---

# 18. Regression Metrics

The regression model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R²
* Adjusted R²

Results are saved to:

```text
artifacts/regression_results.csv
```

---

# 19. Residual Analysis

Residuals are calculated as:

```text
Residual = Actual Fare - Predicted Fare
```

A residual plot is generated with:

```text
X-axis → Predicted Fare
Y-axis → Residual
```

The plot is saved as:

```text
artifacts/plots/regression_residuals.png
```

The residual plot should be inspected to assess whether the residual spread appears approximately constant or whether there is evidence of changing variance across predicted values.

No fixed heteroscedasticity conclusion is assumed without inspecting the generated residual plot.

---

# 20. Model Serialization

The selected classification pipeline is saved using Joblib:

```python
joblib.dump(
    best_classifier,
    "artifacts/best_pipeline.joblib"
)
```

The saved pipeline is then reloaded using:

```python
joblib.load(
    "artifacts/best_pipeline.joblib"
)
```

Sample predictions are generated after reloading to verify that the saved pipeline can be reused.

---

# 21. Output Files

The main generated outputs are:

```text
titanic.csv
titanic_cleaned.csv

artifacts/classification_results.csv
artifacts/imbalance_results.csv
artifacts/regression_results.csv

artifacts/best_pipeline.joblib
```

Visual outputs are stored under:

```text
artifacts/plots/
```

---

# 22. How to Run

From the `analytics` directory:

```bash
python eda.py
```

This performs:

```text
Dataset Loading
      ↓
Data Inspection
      ↓
Missing Value Handling
      ↓
EDA
      ↓
Outlier Detection
      ↓
Correlation Analysis
      ↓
Multivariate Analysis
      ↓
titanic_cleaned.csv
```

Then run:

```bash
python modeling.py
```

This performs:

```text
Load Cleaned Dataset
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Classification
      ↓
Metrics
      ↓
Imbalance Comparison
      ↓
GridSearchCV
      ↓
Random Forest OOB
      ↓
Regression
      ↓
Residual Analysis
      ↓
Model Serialization
```

---

## Summary

The analytics module provides a complete workflow from exploratory analysis to predictive modeling, including preprocessing pipelines, multiple classification algorithms, class-imbalance strategies, hyperparameter tuning, Random Forest OOB evaluation, regression analysis, residual analysis, and model serialization.
