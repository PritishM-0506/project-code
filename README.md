# Machine Learning Assignment 2

## a. Problem statement

The objective of this project is to build and evaluate multiple machine learning classification models for predicting whether a patient is likely to experience a stroke using demographic, lifestyle, and clinical attributes.

The following five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

Each model was evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application was also developed to upload test data, select a model, display evaluation metrics, and show the confusion matrix and classification report.

## b. Dataset description

**Dataset Name:** Healthcare Stroke Prediction Dataset (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
**Problem Type:** Binary Classification  
**Target Column:** `stroke`  
**Target Values:** `0 = No Stroke`, `1 = Stroke`  
**Number of Records:** 5,110  
**Number of Input Features Used:** 10 after excluding the identifier column

The following input attributes were used:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `ever_married`
- `work_type`
- `Residence_type`
- `avg_glucose_level`
- `bmi`
- `smoking_status`

The `id` column was excluded from model training because it is only a patient identifier and does not represent a predictive health characteristic.

The dataset is highly skewed, with approximately 95% No Stroke records and approximately 5% Stroke records. Because of this class imbalance, some models achieved high Accuracy by predicting mainly the majority No Stroke class, while producing very low Recall and F1 Score for Stroke. Therefore, the evaluation results may not appear strong across all metrics. Accuracy alone is not sufficient for this dataset, and AUC, Precision, Recall, F1 Score, and MCC must also be considered while interpreting model performance.

## c. Github Repository Link

**GitHub Repository:** `https://github.com/PritishM-0506/project-code`

The repository contains all required project files, including:

- `app.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- Model implementation files
- Saved trained model files
- Preprocessing artifacts
- Evaluation results

## d. Models used

The following comparison table contains the evaluation metrics calculated for all five models implemented on the selected dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9521 | 0.8420 | 1.0000 | 0.0200 | 0.0392 | 0.1380 |
| Decision Tree | 0.9481 | 0.8226 | 0.0000 | 0.0000 | 0.0000 | -0.0123 |
| kNN | 0.9432 | 0.6023 | 0.0000 | 0.0000 | 0.0000 | -0.0201 |
| Naive Bayes | 0.4413 | 0.7991 | 0.0778 | 0.9600 | 0.1439 | 0.1652 |
| Random Forest (Ensemble) | 0.9481 | 0.8140 | 0.0000 | 0.0000 | 0.0000 | -0.0123 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved the highest Accuracy of 0.9521 and the highest AUC of 0.8420. Its Precision was 1.0000 because the very small number of positive predictions made by the model were correct. However, Recall was only 0.0200, which means the model detected only a very small proportion of actual Stroke cases. The high Accuracy was strongly influenced by the majority No Stroke class. |
| Decision Tree | Decision Tree achieved high Accuracy of 0.9481 and a reasonably good AUC of 0.8226. However, Precision, Recall, and F1 Score were all zero because the model did not correctly identify any Stroke cases at the default prediction threshold. The result shows that the model was biased toward the majority No Stroke class. |
| kNN | kNN achieved Accuracy of 0.9432, but its AUC of 0.6023 was the lowest among the evaluated models. Precision, Recall, and F1 Score were zero because the model failed to correctly detect Stroke cases. Even after scaling the continuous features, the local neighborhoods were dominated by the majority No Stroke class. |
| Naive Bayes | Naive Bayes achieved the highest Recall of 0.9600, the highest F1 Score of 0.1439, and the highest MCC of 0.1652. It identified most actual Stroke cases, but its low Precision of 0.0778 indicates that it also generated many false-positive Stroke predictions. Consequently, its overall Accuracy was lower at 0.4413. |
| Random Forest (Ensemble) | Random Forest achieved high Accuracy of 0.9481 and AUC of 0.8140. However, Precision, Recall, and F1 Score were zero because it did not correctly identify any Stroke cases at the default prediction threshold. Similar to the Decision Tree, the ensemble model was biased toward the majority No Stroke class. |
| Overall Winner for your dataset? | **Naive Bayes** was selected as the overall winner for this skewed dataset because it achieved the highest Recall, F1 Score, and MCC and detected the largest proportion of actual Stroke cases. Logistic Regression achieved the best Accuracy and AUC, but its Stroke Recall was only 0.0200. Since minority-class detection is important in this dataset, Naive Bayes provided the most useful baseline performance despite its high number of false positives. |

## Additional Information

### Steps followed

1. Loaded the Healthcare Stroke Prediction Dataset.
2. Verified the dataset columns, data types, and target distribution.
3. Converted `bmi` to numeric format and replaced missing BMI values using the median.
4. Removed the `id` column from the model input features.
5. Separated the input features and the `stroke` target variable.
6. Applied one-hot encoding to the categorical columns.
7. Divided the data into 80% training data and 20% testing data using a stratified split.
8. Applied `StandardScaler` only to the continuous columns `age`, `avg_glucose_level`, and `bmi` for models configured to use scaled data.
9. Trained Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, and Random Forest models.
10. Calculated Accuracy, AUC, Precision, Recall, F1 Score, and MCC for every model.
11. Saved all trained models and preprocessing artifacts for deployment.
12. Generated `test_data.csv` for testing through the Streamlit application.
13. Developed the Streamlit application with CSV upload, model selection, evaluation metrics, confusion matrix, classification report, and downloadable predictions.
14. Deployed the application using Streamlit Community Cloud.

### Skewed nature of the dataset

The target distribution is highly imbalanced:

- **No Stroke:** approximately 95%
- **Stroke:** approximately 5%

This skewed distribution has a significant effect on the model results:

- A model can obtain high Accuracy by predicting most records as No Stroke.
- High Accuracy does not necessarily mean that Stroke cases are being detected.
- Decision Tree, kNN, and Random Forest produced high Accuracy but zero Recall for Stroke.
- Logistic Regression achieved the highest Accuracy and AUC but detected very few Stroke cases.
- Naive Bayes produced lower Accuracy but detected most Stroke cases, resulting in the highest Recall, F1 Score, and MCC.

For this reason, the model comparison is based on all six evaluation metrics rather than Accuracy alone.

