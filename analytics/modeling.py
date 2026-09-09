import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
df = pd.read_csv("titanic_cleaned.csv")

print("Dataset Shape:", df.shape)

print("\nClass Balance:")
print(df["survived"].value_counts())

print("\nClass Balance Percentage:")
print(df["survived"].value_counts(normalize=True) * 100)

target = "survived"

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[classification_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("AUC:", auc)

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"artifacts/plots/roc_{name.lower().replace(' ', '_')}.png"
    )
    plt.close()

decision_tree = trained_models["Decision Tree"]

feature_names = decision_tree.named_steps[
    "preprocessor"
].get_feature_names_out()

tree_model = decision_tree.named_steps["model"]

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree")
plt.tight_layout()
plt.savefig("artifacts/plots/decision_tree.png")
plt.close()

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced_model.fit(X_train, y_train)

balanced_predictions = balanced_model.predict(X_test)

smote_model = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

smote_model.fit(X_train, y_train)

smote_predictions = smote_model.predict(X_test)

imbalance_results = []

for name, predictions in [
    ("Baseline", baseline_predictions),
    ("Class Weight Balanced", balanced_predictions),
    ("SMOTE", smote_predictions)
]:

    imbalance_results.append({
        "Strategy": name,
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions)
    })

imbalance_df = pd.DataFrame(imbalance_results)

print("\nImbalance Comparison:")
print(imbalance_df)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Score:")
print(grid_search.best_score_)

print("\nOOB Score:")
print(best_rf.named_steps["model"].oob_score_)

regression_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

regression_df = df[
    regression_features + ["fare"]
].copy()

regression_df = regression_df.dropna()

X_reg = regression_df[regression_features]
y_reg = regression_df["fare"]

X_reg = pd.get_dummies(
    X_reg,
    columns=["sex", "embarked"],
    drop_first=True
)

X_reg = X_reg.astype(float)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

regression_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)

regression_model.fit(
    X_reg_train,
    y_reg_train
)

regression_predictions = regression_model.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    regression_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        regression_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    regression_predictions
)

n = len(y_reg_test)
p = X_reg_test.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\nRegression Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

residuals = y_reg_test - regression_predictions

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=regression_predictions,
    y=residuals
)

plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Regression Residual Plot")
plt.tight_layout()
plt.savefig("artifacts/plots/regression_residuals.png")
plt.close()

classification_df = pd.DataFrame(results)

print("\nClassification Model Comparison:")
print(classification_df)

best_classifier_name = classification_df.loc[
    classification_df["F1"].idxmax(),
    "Model"
]

best_classifier = trained_models[
    best_classifier_name
]

best_classifier.fit(X_train, y_train)

joblib.dump(
    best_classifier,
    "artifacts/best_pipeline.joblib"
)

print("\nBest Classifier:", best_classifier_name)

loaded_pipeline = joblib.load(
    "artifacts/best_pipeline.joblib"
)

sample_prediction = loaded_pipeline.predict(
    X_test.iloc[:5]
)

print("\nReloaded Pipeline Predictions:")
print(sample_prediction)

print("\nFinal Classification Metrics:")
print(
    classification_df.to_string(index=False)
)

print("\nRegression Metrics:")
print(
    pd.DataFrame([{
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adjusted_r2
    }]).to_string(index=False)
)

classification_df.to_csv(
    "artifacts/classification_results.csv",
    index=False
)

imbalance_df.to_csv(
    "artifacts/imbalance_results.csv",
    index=False
)

pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}]).to_csv(
    "artifacts/regression_results.csv",
    index=False
)

print("\nModeling completed successfully.")