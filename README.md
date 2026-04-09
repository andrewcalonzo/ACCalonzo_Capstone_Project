# ACCalonzo_Capstone_Project
This project focuses on building and evaluating a machine learning model to detect fraudulent transactions in banking data. Given the inherent imbalance in fraud datasets, the primary goal is to develop a robust model capable of identifying a significant portion of fraudulent activities while managing false positives.
## 2. Problem Statement
Financial institutions face significant losses due to fraudulent transactions. Identifying these transactions accurately and efficiently is crucial to mitigate financial risks and maintain customer trust. The core challenge lies in the highly imbalanced nature of transactional data, where fraudulent instances are rare compared to legitimate ones, making their detection difficult for standard machine learning models.
## 3. Task Type
This is a **Binary Classification** task, specifically aiming to classify transactions as either 'Normal' (0) or 'Fraud' (1).
## 4. Success Metrics
Given the imbalanced nature of fraud detection, the following metrics for the minority class (Fraud) are critical for evaluating model performance:
**Technical Metrics**
*   **Recall (Sensitivity):** To minimize missed fraudulent transactions.
*   **Precision:** To reduce false alarms (legitimate transactions incorrectly flagged as fraud).
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure.
*   **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** To assess the model's ability to distinguish between fraud and normal transactions across various thresholds.
*   **Confusion Matrix:** For a visual breakdown of true positives, true negatives, false positives, and false negatives.
**Business Metrics**
*   **Fraud Detection Rate (FDR)** — Measures how effectively the system identifies actual fraudulent transactions.
*   **False Positive Rate (FPR)** — Shows how many legitimate transactions are incorrectly flagged as fraud.
*   **False Negative Rate (FNR)** — Indicates how many fraudulent transactions the system fails to detect.
*   **Financial Loss Prevented** — Represents the total monetary value saved by catching fraudulent transactions.
*   **Cost of Fraud** — Captures the financial impact of undetected fraud and related operational expenses.
*   **Fraud‑to‑Review Ratio** — Shows the proportion of true fraud cases among all transactions sent for review.
*   **ROI of Fraud Detection System** — Evaluates whether the fraud detection system generates more financial benefit than it costs.
## 5. Data Set
*   **Dataset Name:** `FraudShield_Banking_Data.csv`
*   **Source:** https://www.kaggle.com/datasets/algozee/financial-transaction-fraud-dataset
*   **Description:** This dataset contains banking transaction records with various features such as transaction amount, time, date, customer information, merchant details, and a `Fraud_Label` indicating whether a transaction is legitimate or fraudulent
For  the description of each field, please refer here: https://github.com/andrewcalonzo/ACCalonzo_Capstone_Project/blob/main/Data%20Dictionary
## 6. Repository Structure
```
fraud_detection_project/
├── data/
│   └── FraudShield_Banking_Data.csv
├── notebooks/
│   └── Capstone Project_Fraud Detection.ipynb  # This notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   └── best_fraud_model.pkl
├── reports/
│   ├── exploratory_data_analysis.pdf
│   └── model_performance_report.pdf
├── .gitignore
├── README.md  # This file
└── requirements.txt
```
## 7. Workflow
The project workflow follows standard machine learning development steps, with a particular focus on addressing class imbalance:

### 7.1. Data Loading
*   Loaded `FraudShield_Banking_Data.csv` into a pandas DataFrame.

### 7.2. Data Exploration
*   Performed initial checks for missing values across all columns.
*   Examined data types of each feature.
*   Analyzed the distribution of the target variable (`Fraud_Label`), confirming severe class imbalance.
*   Reviewed descriptive statistics for numerical features.

### 7.3. Data Preprocessing
*   Handled missing numerical values by imputing with the median.
*   Handled missing categorical values by imputing with the mode.
*   Combined `Transaction_Date` and `Transaction_Time` to create `Transaction_DateTime`.
*   Extracted new time-based features: `Transaction_Hour`, `Transaction_DayOfWeek`, `Transaction_Month`.
*   Encoded binary categorical features (e.g., `Is_International_Transaction`, `Is_New_Merchant`) to 0s and 1s.
*   Dropped irrelevant and high-cardinality columns (`Transaction_ID`, `Customer_ID`, `Merchant_ID`, `Device_ID`, `IP_Address`).
*   Encoded the target variable (`Fraud_Label`) to 0 for 'Normal' and 1 for 'Fraud'.
*   Applied One-Hot Encoding to remaining categorical features.
*   Scaled all numerical features using `StandardScaler`.

### 7.4. Model Training and Initial Evaluation
*   Separated features (X) and target (y).
*   Split the dataset into training and testing sets (70% train, 30% test).
*   **Initial Approach (RandomForest with SMOTE):**
    *   Addressed class imbalance on the training data using SMOTE (Synthetic Minority Over-sampling Technique).
    *   Trained a `RandomForestClassifier` on the resampled data.
    *   **Result:** Model completely failed to detect any fraud (Precision, Recall, F1-Score for 'Fraud' were 0).
*   **Alternative Approaches to Handle Imbalance:**
    *   **RandomForest with Class Weights:** Re-trained `RandomForestClassifier` with `class_weight='balanced'` on original training data.
    *   **Result:** Still failed to detect any fraud (Precision, Recall, F1-Score for 'Fraud' were 0).
    *   **RandomForest with ADASYN Oversampling:** Applied ADASYN oversampling to training data and trained `RandomForestClassifier`.
    *   **Result:** Also failed to detect any fraud (Precision, Recall, F1-Score for 'Fraud' were 0).
    *   **BalancedBaggingClassifier (Breakthrough):** Implemented `BalancedBaggingClassifier` with a `DecisionTreeClassifier` base estimator.
    *   **Result:** Achieved a Recall of ~0.2352 for the fraud class, making it the first model to successfully identify fraud.

### 7.5. Hyperparameter Tuning of Best Model
*   Defined a hyperparameter grid for the `BalancedBaggingClassifier` and its `DecisionTreeClassifier` base estimator.
*   Performed `RandomizedSearchCV` to optimize the `BalancedBaggingClassifier`, using `recall` as the scoring metric for cross-validation.
*   **Best Parameters Found:** `{'n_estimators': 100, 'max_features': 0.8, 'estimator__min_samples_leaf': 10, 'estimator__max_depth': 5}`.
*   **Cross-Validation Recall:** ~0.5662.

### 7.6. Final Model Evaluation (Tuned BalancedBaggingClassifier)
*   Evaluated the best-tuned `BalancedBaggingClassifier` on the test set.
*   **Key Metrics (for Fraud class):**
    *   **Recall:** ~0.5716 (Successfully identified over 57% of actual fraudulent transactions)
    *   **Precision:** ~0.0669
    *   **F1-Score:** ~0.1198
    *   **ROC-AUC:** ~0.6077
*   Presented a detailed classification report and confusion matrix.

### 7.7. Conclusion and Future Improvements
*   The tuned `BalancedBaggingClassifier` proved to be the most effective model, significantly improving fraud detection capabilities from 0% to over 57% recall.
*   **Next Steps:**
    *   Further hyperparameter tuning for precision/recall balance.
    *   Advanced feature engineering.
    *   Exploration of other anomaly detection algorithms.
    *   Cost-sensitive learning to better align with business objectives.
    *   Threshold adjustment based on specific business tolerance for false positives vs. false negatives.
```
