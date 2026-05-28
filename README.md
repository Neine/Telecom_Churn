# 📡 Telecom Customer Churn Prediction

A machine learning case study using **Logistic Regression** to predict customer churn for a telecom provider. Given 21 customer-level predictor variables, the model identifies customers likely to switch to a competing provider — enabling proactive retention strategies.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Data Dictionary](#data-dictionary)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Getting Started](#getting-started)

---

## Problem Statement

Customer churn is one of the most critical business metrics in the telecom industry. This project builds a binary classification model to predict whether a customer will churn (`1`) or stay (`0`), using demographic, account, and service usage data.

> **Churn rate in this dataset: ~27%**

---

## Dataset

The dataset is split across three source files that are merged on `customerID`:

| File | Description |
|------|-------------|
| `churn_data.csv` | Account info — tenure, contract type, billing, monthly/total charges, and the target variable `Churn` |
| `customer_data.csv` | Demographics — gender, senior citizen status, partner, dependents |
| `internet_data.csv` | Service add-ons — internet service type, online security, backup, streaming, tech support |
| `Telecom_Churn_Data_Dictionary.csv` | Definitions for all 21 variables |

**Total records:** ~7,043 customers | **Features after merging:** 21 predictors + 1 target

---

## Data Dictionary

| # | Variable | Description |
|---|----------|-------------|
| 1 | `CustomerID` | Unique customer identifier |
| 2 | `Gender` | Customer gender |
| 3 | `SeniorCitizen` | Whether the customer is a senior citizen |
| 4 | `Partner` | Whether the customer has a partner |
| 5 | `Dependents` | Whether the customer has dependents |
| 6 | `Tenure` | Duration (months) the customer has used the service |
| 7 | `PhoneService` | Whether the customer has a landline phone service |
| 8 | `MultipleLines` | Whether the customer has multiple internet lines |
| 9 | `InternetService` | Type of internet service (DSL / Fiber / None) |
| 10 | `OnlineSecurity` | Whether the customer has online security add-on |
| 11 | `OnlineBackup` | Whether the customer has online backup add-on |
| 12 | `DeviceProtection` | Whether the customer has device protection |
| 13 | `TechSupport` | Whether the customer has tech support |
| 14 | `StreamingTV` | Whether the customer streams TV |
| 15 | `StreamingMovies` | Whether the customer streams movies |
| 16 | `Contract` | Contract type (Month-to-month / One year / Two year) |
| 17 | `PaperlessBilling` | Whether the customer uses paperless billing |
| 18 | `PaymentMethod` | Payment method used |
| 19 | `MonthlyCharges` | Monthly charge amount |
| 20 | `TotalCharges` | Total charges to date |
| 21 | `Churn` | **Target** — whether the customer churned (Yes/No) |

---

## Project Structure

```
telecom-churn-prediction/
│
├── churn_data.csv                                        # Account & billing data
├── customer_data.csv                                     # Customer demographics
├── internet_data.csv                                     # Internet service features
├── Telecom_Churn_Data_Dictionary.csv                     # Variable definitions
└── Logistic_Regression_-_Telecom_Churn_Case_Study.ipynb  # Main analysis notebook
```

---

## Methodology

The notebook follows a structured, end-to-end ML pipeline:

**1. Data Import & Merging**
- Load all three CSVs and merge them into a single master dataframe on `customerID`

**2. Exploratory Data Analysis**
- Inspect shape, dtypes, and summary statistics

**3. Data Preparation**
- Map binary Yes/No variables to 0/1
- One-hot encode multi-level categorical variables (Contract, PaymentMethod, InternetService, etc.)
- Drop redundant original columns after encoding
- Handle missing values (~0.15% of `TotalCharges` rows dropped)

**4. Train-Test Split**
- 70% train / 30% test split (`random_state=100`)

**5. Feature Scaling**
- StandardScaler applied to continuous features: `tenure`, `MonthlyCharges`, `TotalCharges`

**6. Correlation Analysis**
- Heatmap to identify multicollinearity
- Drop highly correlated dummy variables (e.g., `MultipleLines_No`, `OnlineSecurity_No`, etc.)

**7. Model Building**
- Logistic Regression using `statsmodels` for p-value-based feature selection
- Iterative removal of features with high p-values and high VIF (Variance Inflation Factor)

**8. Model Evaluation**
- Confusion matrix, accuracy, sensitivity, specificity
- ROC curve and AUC score

**9. Optimal Cutoff Selection**
- Plots accuracy, sensitivity, and specificity across probability thresholds
- Optimal cutoff identified at **0.3** (training) and **0.42** (test)

**10. Precision-Recall Analysis**
- Precision-recall tradeoff curve for final model assessment

**11. Test Set Predictions**
- Final model applied to the held-out test set with performance metrics

---

## Results

| Metric | Value |
|--------|-------|
| Model | Logistic Regression |
| Train/Test Split | 70% / 30% |
| Optimal Cutoff | 0.42 (test set) |
| Evaluation Metrics | Accuracy, Sensitivity, Specificity, AUC |

> Detailed metric values (confusion matrix, AUC score, sensitivity/specificity) are available in the notebook output cells.

---

## Requirements

```
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook "Logistic_Regression_-_Telecom_Churn_Case_Study.ipynb"
   ```

4. **Run all cells** in order — the notebook is fully self-contained and walks through each step with inline commentary.

---

## 📌 Notes

- The `TotalCharges` column is imported as a string and must be converted to numeric — this is handled in the data preparation step.
- Rows with null `TotalCharges` (~11 records) are dropped as they represent less than 0.2% of the data.
- VIF analysis is used alongside p-values to handle multicollinearity before finalizing the feature set.
