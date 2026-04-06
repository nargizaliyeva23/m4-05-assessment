![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Assessment | End-to-End Supervised Learning Pipeline

## Overview

This assessment evaluates your ability to build a complete supervised learning pipeline from scratch. You'll take a real-world dataset through the full lifecycle: exploration, cleaning, feature engineering, model training, comparison, and final analysis. This is the kind of workflow you'll repeat in every data science project — the goal is to demonstrate that you can execute it independently and make informed decisions at each stage.

You'll work with the Adult Income (Census) dataset, a widely-used benchmark for binary classification. The task is to predict whether an individual earns more than $50K per year based on demographic and employment features. The dataset has a realistic mix of challenges: missing values, categorical features, class imbalance, and features that require thoughtful engineering.

## Learning Goals

This assessment evaluates your ability to:

- Conduct thorough exploratory data analysis and handle real-world data quality issues.
- Build a preprocessing pipeline using scikit-learn's `ColumnTransformer` and `Pipeline`.
- Train, evaluate, and compare at least four supervised learning models.
- Tune hyperparameters systematically and analyze feature importances.
- Communicate findings clearly through code, visualizations, and written analysis.

## Prerequisites

- Python 3.9+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Requirements

1. **Fork** this repository to your own GitHub account.
2. **Clone** the fork to your local machine.
3. Work in a single Jupyter Notebook called **`m4-05-assessment.ipynb`**.
4. **Commit regularly** — your commit history should show incremental progress, not a single final commit.

## Tasks

### Task 1 — Data Exploration & Cleaning

Load the Adult Income dataset and perform a thorough EDA.

```python
from sklearn.datasets import fetch_openml
adult = fetch_openml("adult", version=2, as_frame=True)
X, y = adult.data, adult.target
```

1. **Explore the dataset:** Report the shape, feature names, data types, and the first few rows. How many numerical vs. categorical features are there?
2. **Target variable:** What are the classes? What is the class distribution? Is the dataset balanced?
3. **Missing values:** Identify columns with missing values (some may be encoded as `"?"` or `NaN`). Report the count and percentage for each. Decide on a strategy for each column (drop rows or drop column) and justify your choices.
4. **Distributions:** Create visualizations for at least 3 numerical features (histograms or box plots) and at least 3 categorical features (bar charts). Highlight any interesting patterns or outliers.
5. **Bivariate analysis:** Explore relationships between key features and the target. For example: How does income vary by education level? By occupation? By hours worked per week?
6. **Document your findings** in markdown cells. Write a summary paragraph at the end of Task 1 describing the dataset's key characteristics and any concerns for modeling.

### Task 2 — Feature Engineering & Preprocessing

Build a robust preprocessing pipeline.

1. **Separate features by type:** Identify which columns are numerical and which are categorical.
2. **Numerical preprocessing:** Create a pipeline that scales features (`StandardScaler`).
3. **Categorical preprocessing:** Create a pipeline that encodes features (`OneHotEncoder` with `handle_unknown="ignore"`).
4. **Combine with ColumnTransformer:** Build a single `ColumnTransformer` that applies the appropriate pipeline to each feature type.
5. **Full pipeline:** Wrap the `ColumnTransformer` and a classifier into a scikit-learn `Pipeline`. Demonstrate that it works by fitting and predicting with a simple model (e.g., `LogisticRegression`).
6. **Train/test split:** Split the data (80/20, `stratify=y`, `random_state=42`). All preprocessing must be fit on training data only.

In a markdown cell, explain why using a `Pipeline` prevents data leakage and why this matters.

### Task 3 — Model Training & Comparison

Train at least four different models and compare their performance.

1. Using your preprocessing pipeline from Task 2, train the following models:
   - `LogisticRegression(max_iter=1000)`
   - `SVC(probability=True)`
   - `RandomForestClassifier(random_state=42)`
   - `GradientBoostingClassifier(random_state=42)`

2. For each model, evaluate on the test set using:
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1 score (weighted)

3. Create a **comparison table** (DataFrame) with all models and metrics. Sort by F1 score.
4. Plot **ROC curves** for all four models on a single figure with AUC values in the legend.
5. Plot the **confusion matrix** for each model.
6. In a markdown cell, discuss: Which model performs best? Are there meaningful differences between the models? Which metric is most important for this problem (predicting income) and why?

### Task 4 — Best Model Analysis

Deep-dive into your best model.

1. Select the best-performing model from Task 3.
2. **Hyperparameter tuning:** Use `GridSearchCV` or `RandomizedSearchCV` with at least 3 hyperparameters. Use 5-fold cross-validation with `scoring="f1_weighted"`.
3. Report the **best parameters** and the improvement in cross-validation score over the default model.
4. Evaluate the tuned model on the test set. Create a detailed `classification_report`.
5. **Feature importance analysis:**
   - If your best model supports `feature_importances_` (tree-based), plot the top 15 most important features.
   - If not (e.g., SVM), use permutation importance from scikit-learn.
   - Which features are most predictive of high income? Do the results align with your EDA findings?
6. **Executive summary:** Write a 200–300 word markdown cell summarizing your entire analysis. Include:
   - The problem and dataset
   - Key preprocessing decisions
   - Which model you recommend and why
   - The most important predictive features
   - Limitations and potential next steps

## Submission

### What to submit

- `m4-05-assessment.ipynb` — your completed notebook with all code, outputs, and written analysis.

### Definition of done (checklist)

- [ ] Dataset is loaded, explored, and cleaned with documented decisions.
- [ ] A scikit-learn Pipeline with ColumnTransformer handles all preprocessing.
- [ ] At least 4 models are trained and compared with a metrics table.
- [ ] ROC curves and confusion matrices are plotted for all models.
- [ ] Best model is tuned with GridSearchCV/RandomizedSearchCV.
- [ ] Feature importances are analyzed and visualized.
- [ ] An executive summary ties the analysis together.
- [ ] Commit history shows incremental progress.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "assessment: complete end-to-end supervised learning pipeline"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.

## Evaluation Criteria

| Criterion | Weight | Description |
|---|---|---|
| Data Exploration & Cleaning | 20% | Thoroughness of EDA, quality of visualizations, justified handling of missing values |
| Feature Engineering & Pipeline | 20% | Correct use of ColumnTransformer and Pipeline, no data leakage, clear code |
| Model Training & Comparison | 25% | At least 4 models trained, proper evaluation metrics, clear comparison table and plots |
| Best Model Analysis | 25% | Systematic hyperparameter tuning, feature importance analysis, quality of executive summary |
| Code Quality & Communication | 10% | Clean code, clear markdown explanations, notebook runs without errors, regular commits |
