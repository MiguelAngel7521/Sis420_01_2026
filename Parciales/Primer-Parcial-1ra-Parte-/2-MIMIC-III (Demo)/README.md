# ICU Length of Stay Prediction and Clinical Event Analysis (MIMIC-III)

## Overview
This project leverages the **MIMIC-III clinical database** to:

1. Predict **ICU Length of Stay (LOS)** using machine learning.
2. Perform exploratory and temporal analysis of patient clinical events.

The analysis is split into two parts:

- **Part 1:** Regression modeling for ICU LOS prediction.
- **Part 2:** Exploratory data analysis using CHARTEVENTS and temporal patterns.

---

## Dataset

The following MIMIC-III tables are used:

- **PATIENTS**: Patient demographics and mortality flags
- **ADMISSIONS**: Admission details including type, insurance, ethnicity, and diagnosis
- **ICUSTAYS**: ICU stay information including LOS and care unit
- **DIAGNOSES_ICD**: ICD-9 diagnosis codes
- **D_ICD_DIAGNOSES**: Diagnosis descriptions
- **CHARTEVENTS**: Time-stamped clinical measurements and interventions

> Note: Access to MIMIC-III requires credentialing via PhysioNet.

---

# Part 1: ICU Length of Stay Prediction

## Data Preprocessing

### Data Cleaning
- Removed inconsistencies across patient admissions
- Dropped irrelevant columns
- Handled missing values

### Feature Engineering
- Encoded categorical variables (admission type, insurance, ethnicity, care units)
- Grouped ethnicity into broader categories
- Created diagnosis categories from ICD-9 codes
- Expanded multi-diagnosis records into separate rows

### Data Integration
- Merged all tables into a unified dataset
- Final dataset contains:
  - ~738,000 rows
  - 14 features
  - Target variable: LOS (days)

---

## Models

| Model              | MSE   | RMSE | MAE  | R²   |
|-------------------|------|------|------|------|
| Linear Regression | 91.78 | —    | —    | 0.07 |
| Random Forest     | 65.07 | —    | 4.03 | 0.34 |
| Gradient Boosting | 69.14 | 8.32 | 4.31 | 0.30 |
| XGBoost           | 62.69 | 7.92 | 4.09 | 0.36 |

**Best Model:** XGBoost (R² = 0.36, MAE ≈ 4.09 days)

---

## Key Insights

- Most important features: admission type, diagnosis category, sequence number
- Model tends to underestimate long ICU stays
- Circulatory system diagnoses are the most frequent

---

# Part 2: Clinical Events and Temporal Analysis

## Data Loading & Cleaning
- Loaded tables: `icustays_cleaned`, `admissions_cleaned`, `diagnoses_icd_cleaned`, `d_items`, `patients_cleaned`, `chartevents`
- Merged into a unified dataset (`df_final`) linking:
  - Clinical events (CHARTEVENTS)
  - Patient demographics
  - Diagnoses
  - ICU stays
  - Admission details

---

## Patient Selection
- Filtered patients with **circulatory system diagnoses (ICD9 recode = 6)**
- Retained only patients with CHARTEVENTS data

---

## Correlation Analysis
- Encoded categorical variables: LABEL, DIAGNOSIS, ETHNICITY, ADMISSION_TYPE
- Computed correlation matrix

**Observation:**  
- Weak correlations overall  
- Strongest: `DIAGNOSIS_ENCODED ↔ ADMISSION_TYPE_ENCODED = 0.17`  
- Suggests linear models are insufficient to capture complex clinical patterns

---

## Exploratory Analysis of Hospitalizations
- Most patients had a single admission
- Outliers: two patients with 17 admissions (likely complex or chronic cases)
- Visualized timelines for high-admission patients (IDs 5060 and 73713) showing distribution of ICU events

---

## Length of Stay Analysis
- Mean LOS: 3.56 days
- Standard deviation: 4.72 days
- Maximum LOS: 88.37 days
- Significant variability and presence of outliers

---

## First 24-Hour ICU Interventions
- Extracted first 24 hours of ICU data
- Plotted top 10 most frequent ITEMIDs and fraction of day recorded
- Observed that most interventions occur early in the ICU stay

---

## Key Observations
- **Data Quality:** Several `SettingWithCopyWarning` and dtype warnings indicate slices of DataFrames; using `.copy()` is recommended
- **Correlation:** Weak correlations suggest that advanced modeling techniques are needed
- **Admissions & LOS:** Extreme cases (17 admissions, LOS up to 88 days) should be carefully handled
- **Temporal Patterns:** Early ICU interventions dominate, which aligns with clinical expectations

---

## Recommendations & Future Work

### Predictive Modeling
- Predict outcomes such as mortality, readmission, or prolonged LOS
- Features could include demographics, admission type, diagnosis categories, and aggregated early-hour measurements

### Clustering Patients
- Apply clustering methods (k-means, hierarchical) to identify patient subgroups with similar clinical patterns

### Survival Analysis
- Use time-to-event models (e.g., Cox proportional hazards) to study in-hospital mortality

### Feature Engineering
- Derive meaningful features from CHARTEVENTS:
  - Trends over time
  - Frequency of interventions
  - Threshold-based events (e.g., hypotension episodes)
- Group ITEMIDs into interpretable categories (vitals, lab results, medications)

### Handling Missing Data
- Apply imputation or missingness indicator variables

### Optimization
- Reduce memory footprint by selecting necessary columns early and using efficient data types

### Clinical Validation
- Compare findings with clinical literature or ICU protocols to validate insights

---

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lime

```

## Technologies Used
- Python 3.10+
- Pandas, NumPy, Matplotlib
- Scikit-learn
- XGBoost
- LIME

## Project Structure

```
MIMIC-ICU-LOS-Project/
│
├── data/
│ ├── PATIENTS.csv
│ ├── ADMISSIONS.csv
│ ├── ICUSTAYS.csv
│ ├── DIAGNOSES_ICD.csv
│ ├── D_ICD_DIAGNOSES.csv
│ └── CHARTEVENTS.csv
│
├── notebooks/
│ ├── mimic_code_part1.ipynb
│ ├── mimic_code_part2.ipynb
│ ├── mimic_code_part3.ipynb
│ 
└── README.md

```


