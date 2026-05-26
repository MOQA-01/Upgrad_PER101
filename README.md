# Predicting Employee Retention
### Logistic Regression Analysis — Assignment `LogR/01` · Group `PER101`

> **Objective:** Develop a Logistic Regression model to predict binary outcomes (**Stayed / Left**) based on employee data and help HR proactively identify employees likely to stay, while understanding the factors driving their loyalty.

---

## Team

| Name | Role |
|------|------|
| Mohammed Qalandar Shah Quazi | Group Member |
| Ramyata Rohan Mendhe | Group Member |
| Teja P | Group Member |
| Pratiksha Hanumant Wagaj | Group Member |

---

## Project Overview

| Field | Details |
|-------|---------|
| **Assignment ID** | LogR/01 |
| **Group Name** | PER101 |
| **Language** | Python 3 |
| **Data Source** | Kaggle – Employee Retention Dataset |
| **Report Date** | May 2026 |

A mid-sized technology company wants to proactively identify employees likely to stay and understand the factors contributing to their loyalty. This project builds a **Logistic Regression** model using demographic details, job satisfaction scores, performance metrics, and tenure data.

---

## Repository Structure

```
PER101/
├── README.md                              # This file
├── Predicting_Employee_Retention_Mohammed Qalandar Shah Quazi_Ramyata Rohan Mendhe_Teja P_Pratiksha Hanumant Wagaj.ipynb  # Main analysis notebook
├── Employee_data.csv                      # Raw dataset (74,610 rows)
└── Employee_Retention_Report.docx         # Full written report
```

---

## Dataset

### Overview

| Attribute | Detail |
|-----------|--------|
| **Total Records** | 74,610 rows × 24 columns |
| **After Cleaning** | 70,635 rows *(94.67% retained)* |
| **Missing Values** | `Distance from Home`: 1,912 · `Company Tenure`: 2,413 |
| **Redundant Columns Dropped** | `Employee ID` *(non-predictive identifier)* |
| **Target Variable** | `Attrition` — `Stayed` / `Left` |
| **Train / Validation Split** | 70% training (49,444) · 30% validation (21,191) |

### Feature Columns

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `Employee ID` | Categorical | Unique employee identifier *(dropped)* |
| 2 | `Age` | Numerical | Employee age |
| 3 | `Gender` | Categorical | Male / Female |
| 4 | `Years at Company` | Numerical | Tenure in years |
| 5 | `Job Role` | Categorical | Department/function |
| 6 | `Monthly Income` | Numerical | Monthly salary (right-skewed) |
| 7 | `Work-Life Balance` | Categorical | Poor / Fair / Good / Excellent |
| 8 | `Job Satisfaction` | Categorical | Low / Medium / High / Very High |
| 9 | `Performance Rating` | Categorical | Low / Below Avg / Average / High |
| 10 | `Number of Promotions` | Numerical | Count of promotions received |
| 11 | `Overtime` | Categorical | Yes / No |
| 12 | `Distance from Home` | Numerical | Commute distance |
| 13 | `Education Level` | Categorical | High School → PhD |
| 14 | `Marital Status` | Categorical | Single / Married / Divorced |
| 15 | `Number of Dependents` | Numerical | Count of dependents |
| 16 | `Job Level` | Categorical | Entry / Mid / Senior |
| 17 | `Company Size` | Categorical | Small / Medium / Large |
| 18 | `Company Tenure (In Months)` | Numerical | Months with company |
| 19 | `Remote Work` | Categorical | Yes / No |
| 20 | `Leadership Opportunities` | Categorical | Yes / No |
| 21 | `Innovation Opportunities` | Categorical | Yes / No |
| 22 | `Company Reputation` | Categorical | Poor / Fair / Good / Excellent |
| 23 | `Employee Recognition` | Categorical | Low / Medium / High |
| 24 | `Attrition` | **Target** | **Stayed / Left** |

---

## Exploratory Data Analysis (EDA)

### 2.1 Class Balance
Training set is near-balanced — no oversampling required:

| Class | Proportion |
|-------|-----------|
| **Stayed** | 52.15% |
| **Left** | 47.85% |

### 2.2 Univariate Analysis – Numerical Features
- Distributions of all **7 numerical features** were examined.
- `Monthly Income` is **right-skewed**; most other features are approximately uniform or normally distributed.

### 2.3 Correlation Analysis
- Numerical features are **largely uncorrelated** with each other (all pairwise correlations **< 0.15**), confirming low multicollinearity among the continuous predictors.

### 2.4 Bivariate Analysis – Categorical Features vs Attrition
Key observations — employees with the following characteristics show **significantly higher departure rates**:
- ❌ Poor **Work-Life Balance**
- ❌ Low **Job Satisfaction**
- ❌ Low **Performance Rating**
- ❌ Low **Employee Recognition**
- ❌ Poor **Company Reputation**

---

## Feature Engineering

| Step | Description |
|------|-------------|
| **Dummy Encoding** | One-hot encoding applied to all **15 categorical columns** (`drop_first=True`). Feature matrix expanded from 23 → **52 columns**. |
| **Feature Scaling** | `StandardScaler` fitted on **training data only**, then applied to validation set to prevent data leakage. |
| **RFE Feature Selection** | Top **15 features** selected using Recursive Feature Elimination (RFE) with `LogisticRegression` as estimator. |

### Top 15 RFE-Selected Features

| Feature | Category | Coefficient | Direction |
|---------|----------|-------------|-----------|
| Job Level – Senior | Career | **+2.51** | ✅ Retention |
| Remote Work – Yes | Work Arrangement | **+1.70** | ✅ Retention |
| Marital Status – Single | Demographics | **−1.69** | ❌ Attrition |
| Education Level – PhD | Education | **+1.52** | ✅ Retention |
| Work-Life Balance – Poor | Well-being | **−1.26** | ❌ Attrition |
| Work-Life Balance – Fair | Well-being | **−1.09** | ❌ Attrition |
| Job Level – Mid | Career | **+0.97** | ✅ Retention |
| Company Reputation – Poor | Employer Brand | **−0.77** | ❌ Attrition |
| Performance Rating – Low | Performance | **−0.60** | ❌ Attrition |
| Gender – Male | Demographics | **+0.59** | ✅ Retention |
| Company Reputation – Fair | Employer Brand | **−0.53** | ❌ Attrition |
| Job Satisfaction – Low | Engagement | **−0.50** | ❌ Attrition |
| Job Satisfaction – Very High | Engagement | **−0.48** | ❌ Attrition |
| Overtime – Yes | Work Conditions | **−0.37** | ❌ Attrition |
| Performance Rating – Below Avg | Performance | **−0.31** | ❌ Attrition |

---

## Model Building

### Algorithm
**Logistic Regression** (via `statsmodels.discrete.discrete_model.Logit`) with statistical summaries (p-values, coefficients, VIF).

### ROC Curve – Training Set
- **Training AUC = 0.826** — strong separability between retained and departed employees.

### Optimal Threshold Selection
The optimal cutoff **(0.5)** is where **|Sensitivity − Specificity|** is minimised, balancing the ability to correctly identify both retained and departed employees.

| Cutoff | Accuracy | Sensitivity | Specificity |
|--------|----------|-------------|-------------|
| 0.3 | 70.45% | 90.65% | 48.44% |
| 0.4 | 73.13% | 84.08% | 61.19% |
| **0.5 ✦** | **73.92%** | **75.48%** | **72.22%** |
| 0.6 | 73.12% | 64.49% | 82.51% |
| 0.7 | 70.93% | 54.47% | 88.87% |

---

## Model Evaluation – Validation Set

### Final Performance Metrics

| Accuracy | AUC-ROC | Sensitivity | Specificity | Precision |
|----------|---------|-------------|-------------|-----------|
| **73.64%** | **0.8239** | **74.78%** | **72.41%** | **74.61%** |

### Confusion Matrix (Validation Set — 21,191 rows)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **True Positives (TP)** | 8,244 | Correctly predicted as *Stayed* |
| **True Negatives (TN)** | 7,361 | Correctly predicted as *Left* |
| **False Positives (FP)** | 2,805 | Predicted *Stayed*, actually *Left* |
| **False Negatives (FN)** | 2,781 | Predicted *Left*, actually *Stayed* |

> **Total correct**: 15,605 / 21,191 = **73.64% accuracy**

---

## 💡 Key Insights

1. **Job Level is the strongest retention predictor.** Senior (+2.51) and Mid-level (+0.97) employees are far more likely to stay; **entry-level staff show the highest churn risk**.

2. **Remote Work significantly boosts retention.** Remote-enabled employees are much more likely to stay (+1.70); **flexibility is a key differentiator**.

3. **Poor Work-Life Balance strongly drives attrition.** Poor (−1.26) and Fair (−1.09) balance scores are **top predictors of departure**.

4. **Single employees leave more often.** Greater geographic mobility and fewer local ties make single employees **higher flight risks** (−1.69).

5. **Company Reputation impacts retention.** Poor (−0.77) and Fair (−0.53) reputation perceptions **significantly increase departure likelihood**.

6. **Low Performance Ratings correlate with attrition.** Low (−0.60) and Below-Average (−0.31) rated employees are **more likely to leave**.

7. **Job Satisfaction extremes both predict departure.** Low satisfaction (−0.50) and, surprisingly, **Very High satisfaction (−0.48) are both departure signals** — possibly indicating top performers are being poached.

---

## Recommendations

| # | Recommendation | Expected Impact |
|---|----------------|-----------------|
| 1 | Build **structured career paths** from Entry → Senior with transparent promotion timelines. | 🔴 High |
| 2 | **Expand remote / hybrid work policy** and communicate it during onboarding and reviews. | 🔴 High |
| 3 | Run **quarterly Work-Life Balance surveys**; flag and address teams with Poor/Fair scores. | 🔴 High |
| 4 | Launch **engagement programmes** for single employees — mentorship, social events, relocation support. | 🟡 Medium |
| 5 | Conduct **bi-annual employer brand audits** and publish improvements transparently. | 🟠 Medium–High |
| 6 | Introduce a monthly **Employee Recognition Programme** (spot awards, peer shoutouts). | 🟡 Medium |
| 7 | **Use the model monthly** to generate a flight-risk score per employee for proactive HR interventions. | 🔴 High |

---

## Methodology Summary

```
Raw Data (74,610 rows)
        │
        ▼
Data Cleaning ──► Drop missing rows (5.33%) ──► Drop Employee ID
        │
        ▼
70/30 Train-Validation Split (stratified, random_state=42)
        │
        ▼
EDA on Training Set ──► Distributions · Correlations · Class Balance · Bivariate Analysis
        │
        ▼
Feature Engineering ──► One-Hot Encoding (15 cols) ──► StandardScaler ──► RFE (top 15)
        │
        ▼
Logistic Regression (statsmodels Logit)
        │
        ▼
Threshold Optimisation ──► Sensitivity/Specificity crossover at 0.5
        │
        ▼
Validation Evaluation ──► Accuracy 73.64% · AUC 0.8239 · Confusion Matrix
```

---

## Tech Stack

| Library | Version | Usage |
|---------|---------|-------|
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical computing |
| `matplotlib` | latest | Visualisation |
| `seaborn` | latest | Statistical plots |
| `scikit-learn` | latest | RFE, StandardScaler, metrics |
| `statsmodels` | latest | Logit model with p-values & VIF |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/PER101.git
cd PER101

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

# Launch Jupyter
jupyter notebook Predicting_Employee_Retention_Mohammed Qalandar Shah Quazi_Ramyata Rohan Mendhe_Teja P_Pratiksha Hanumant Wagaj.ipynb
```

> **Note:** The dataset file `Employee_data.csv` must be in the same directory as the notebook.

---

*Report Date: May 2026 · Assignment ID: LogR/01 · Group: PER101*
