# Module 2 — Analytics Pipeline

## Overview

This module implements an end-to-end Titanic analytics and predictive modeling workflow. The dataset is loaded once using Seaborn, cleaned according to the required missing-value thresholds, saved as an offline CSV, analyzed through EDA, and then reused for classification and regression modeling.

## Dataset Loading

The Titanic dataset was loaded once using:

```python
sns.load_dataset("titanic")
## Multivariate Analysis — Chart Interpretations

### 1. Survival Rate by Sex and Passenger Class
The chart shows that survival rates vary across both passenger sex and passenger class.
Female passengers generally have higher survival rates than male passengers across passenger classes.
First-class passengers also show higher survival rates compared with lower passenger classes.

### 2. Fare by Passenger Class and Survival
The distribution of fare differs substantially across passenger classes, with higher fares generally associated with higher passenger classes.
The chart also shows differences in fare distributions between passengers who survived and those who did not.
This indicates that passenger class and fare are related to survival outcomes.

### 3. Age vs Fare by Survival
The scatter plot shows the relationship between passenger age and fare while distinguishing passengers by survival status.
Survival outcomes are distributed across different age and fare ranges rather than being determined by a single variable.
Higher fares are concentrated among some passengers from higher passenger classes.

### 4. Class, Sex and Survival
The chart demonstrates that survival rates differ across passenger classes and between male and female passengers.
Female passengers generally have higher survival rates, while male survival rates vary considerably across passenger classes.
This highlights the combined effect of passenger class and sex on survival outcomes.
