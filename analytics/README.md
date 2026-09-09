# Module 2 — Analytics Pipeline

## Overview

This module implements an end-to-end Titanic analytics and predictive modeling workflow. The dataset is loaded once using Seaborn, cleaned according to the required missing-value thresholds, saved as an offline CSV, analyzed through EDA, and then reused for classification and regression modeling.

## Dataset Loading

The Titanic dataset was loaded once using:

```python
sns.load_dataset("titanic")