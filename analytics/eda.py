import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("artifacts/plots", exist_ok=True)

df = sns.load_dataset("titanic")

df.to_csv("titanic.csv", index=False)

print("Shape:", df.shape)

print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe())

missing = df.isnull().mean() * 100
missing = missing[missing > 0]

print("\nMissing Value Percentage:")
print(missing)

df_clean = df.copy()

for column in missing.index:
    percentage = missing[column]

    if percentage < 5:
        df_clean = df_clean.dropna(subset=[column])
        print(f"{column}: {percentage:.2f}% -> Drop rows")
    elif percentage <= 30:
        if pd.api.types.is_numeric_dtype(df_clean[column]):
            df_clean[column] = df_clean[column].fillna(
                df_clean[column].median()
            )
        else:
            df_clean[column] = df_clean[column].fillna(
                df_clean[column].mode()[0]
            )
        print(f"{column}: {percentage:.2f}% -> Impute")
    else:
        df_clean = df_clean.drop(columns=[column])
        print(f"{column}: {percentage:.2f}% -> Drop column")

print("\nCleaned Shape:", df_clean.shape)

print("\nRemaining Missing Values:")
print(df_clean.isnull().sum())

for column in ["age", "fare"]:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_clean[column], kde=True)
    plt.title(f"{column.capitalize()} Distribution")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{column}_histogram.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df_clean[column])
    plt.title(f"{column.capitalize()} Box Plot")
    plt.xlabel(column)
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{column}_boxplot.png")
    plt.close()

def outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

age_outliers = outlier_count(df_clean["age"])
fare_outliers = outlier_count(df_clean["fare"])

print("\nIQR Outliers:")
print("Age:", age_outliers)
print("Fare:", fare_outliers)

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]

print("\nFare Statistics:")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

if fare_mean > fare_median > fare_mode:
    print("Fare distribution: Right-skewed")
elif fare_mean < fare_median < fare_mode:
    print("Fare distribution: Left-skewed")
else:
    print("Fare distribution: Approximately symmetric")

sex_survival = df_clean.groupby("sex")["survived"].mean()

pclass_survival = df_clean.groupby("pclass")["survived"].mean()

sex_pclass_survival = df_clean.groupby(
    ["sex", "pclass"]
)["survived"].mean()

print("\nSurvival Rate by Sex:")
print(sex_survival)

print("\nSurvival Rate by Pclass:")
print(pclass_survival)

print("\nSurvival Rate by Sex and Pclass:")
print(sex_pclass_survival)

male_first = df_clean[
    (df_clean["sex"] == "male") &
    (df_clean["pclass"] == 1)
]

female_first = df_clean[
    (df_clean["sex"] == "female") &
    (df_clean["pclass"] == 1)
]

print("\nBoolean Masking:")
print(
    "Male first class survival:",
    male_first["survived"].mean()
)

print(
    "Female first class survival:",
    female_first["survived"].mean()
)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = df_clean[corr_columns].corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()
plt.savefig("artifacts/plots/correlation_heatmap.png")
plt.close()

pairs = []

for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        pairs.append(
            (
                corr_columns[i],
                corr_columns[j],
                abs(corr.iloc[i, j]),
                corr.iloc[i, j]
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: x[2],
    reverse=True
)

print("\nTwo Strongest Correlations:")

for pair in pairs[:2]:
    print(
        f"{pair[0]} - {pair[1]}: "
        f"{pair[3]:.4f}"
    )

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="sex",
    y="survived",
    hue="pclass"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()
plt.savefig("artifacts/plots/survival_sex_pclass.png")
plt.close()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_clean,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Fare by Class and Survival")
plt.tight_layout()
plt.savefig("artifacts/plots/fare_class_survival.png")
plt.close()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived"
)

plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig("artifacts/plots/age_fare_survival.png")
plt.close()

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df_clean,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Class, Sex and Survival")
plt.tight_layout()
plt.savefig("artifacts/plots/class_sex_survival.png")
plt.close()

before_age_mean = df_clean["age"].mean()
before_age_std = df_clean["age"].std()

before_fare_mean = df_clean["fare"].mean()
before_fare_std = df_clean["fare"].std()

df_clean["age_z"] = (
    df_clean["age"] - before_age_mean
) / before_age_std

df_clean["fare_z"] = (
    df_clean["fare"] - before_fare_mean
) / before_fare_std

print("\nStandardization Before:")
print("Age mean:", before_age_mean)
print("Age std:", before_age_std)
print("Fare mean:", before_fare_mean)
print("Fare std:", before_fare_std)

print("\nStandardization After:")
print("Age z mean:", df_clean["age_z"].mean())
print("Age z std:", df_clean["age_z"].std())
print("Fare z mean:", df_clean["fare_z"].mean())
print("Fare z std:", df_clean["fare_z"].std())

df_clean = df_clean.drop(
    columns=["age_z", "fare_z"]
)

df_clean.to_csv("titanic_cleaned.csv", index=False)

print("\nEDA completed successfully.")
print("Raw dataset saved to: titanic.csv")
print("Cleaned dataset saved to: titanic_cleaned.csv")
print("Charts saved to: artifacts/plots/")