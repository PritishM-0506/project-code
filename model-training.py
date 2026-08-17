import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from pathlib import Path

PROJECT_DIR = Path(".")
MODEL_DIR = PROJECT_DIR / "model"


print("Project directory:", PROJECT_DIR.resolve())
print("Model directory:", MODEL_DIR.resolve())

DATA_FILE = "healthcare-dataset-stroke-data.csv"

df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

#Fill Missing Values
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
bmi_median = float(df["bmi"].median())
df["bmi"] = df["bmi"].fillna(bmi_median)

#Identify data types of columns used in prepossing and differnt type of models.
target_column = "stroke"
identifier_column = "id"
continuous_columns = ["age","avg_glucose_level","bmi"]
binary_columns = ["hypertension","heart_disease"]
categorical_columns = ["gender","ever_married","work_type","Residence_type","smoking_status"]

X_raw = df.drop(columns=[identifier_column, target_column]).copy()
y = df[target_column].astype(int).copy()

#Encoding categorical columns
X_encoded = pd.get_dummies(X_raw,columns=categorical_columns,drop_first=True,dtype=int)

#Split - via indices

all_indices = np.arange(len(df))
train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(train_indices))
print("Testing rows :", len(test_indices))

X_train = X_encoded.iloc[train_indices].copy()
X_test = X_encoded.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()


#Scaling

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_columns] = scaler.fit_transform(X_train[continuous_columns])
X_test_scaled[continuous_columns] = scaler.transform(X_test[continuous_columns])
print("Selective scaling completed.")

from model.logistic_regression import build_model as build_logistic_regression
from model.decision_tree import build_model as build_decision_tree
from model.knn import build_model as build_knn
from model.naive_bayes import build_model as build_naive_bayes
from model.random_forest import build_model as build_random_forest

print("All model definitions imported successfully.")

# This dictiionary of model configitarion is necessary because each model recived gets differnt types of inputs scaled / non scaled.
# 
model_configurations = {
    "Logistic Regression": {
        "model": build_logistic_regression(),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "file_name": "logistic_regression.pkl",
        "uses_scaled_data": True
    },

    "Decision Tree": {
        "model": build_decision_tree(),
        "X_train": X_train,
        "X_test": X_test,
        "file_name": "decision_tree.pkl",
        "uses_scaled_data": False
    },

    "KNN": {
        "model": build_knn(),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "file_name": "knn.pkl",
        "uses_scaled_data": True
    },

    "Naive Bayes": {
        "model": build_naive_bayes(),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "file_name": "naive_bayes.pkl",
        "uses_scaled_data": True
    },

    "Random Forest": {
        "model": build_random_forest(),
        "X_train": X_train,
        "X_test": X_test,
        "file_name": "random_forest.pkl",
        "uses_scaled_data": False
    }
}

def evaluate_classifier(model, X_evaluation, y_actual):
    """
    Evaluate a trained binary classification model.

    Returns:
        metrics: Dictionary containing assignment metrics.
        y_pred: Predicted class labels.
        y_probability: Probability of the positive class.
    """

    y_pred = model.predict(X_evaluation)

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(
            X_evaluation
        )[:, 1]

    elif hasattr(model, "decision_function"):
        y_probability = model.decision_function(
            X_evaluation
        )

    else:
        y_probability = y_pred

    metrics = {
        "Accuracy": accuracy_score(
            y_actual,
            y_pred
        ),

        "AUC": roc_auc_score(
            y_actual,
            y_probability
        ),

        "Precision": precision_score(
            y_actual,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_actual,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_actual,
            y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_actual,
            y_pred
        )
    }

    return metrics, y_pred, y_probability

model_results = []
trained_models = {}
model_predictions = {}
model_probabilities = {}

for model_name, configuration in model_configurations.items():

    print(f"\nTraining {model_name}...")

    model = configuration["model"]
    model_X_train = configuration["X_train"]
    model_X_test = configuration["X_test"]

    start_time = time.perf_counter()

    model.fit(
        model_X_train,
        y_train
    )

    training_time = time.perf_counter() - start_time

    metrics, y_pred, y_probability = evaluate_classifier(
        model,
        model_X_test,
        y_test
    )

    model_file_path = MODEL_DIR / configuration["file_name"]

    with open(model_file_path, "wb") as file:
        pickle.dump(model, file)

    trained_models[model_name] = model
    model_predictions[model_name] = y_pred
    model_probabilities[model_name] = y_probability

    result_row = {
        "ML Model Name": model_name,
        "Accuracy": metrics["Accuracy"],
        "AUC": metrics["AUC"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "MCC": metrics["MCC"],
        "Training Time Seconds": training_time,
        "Scaled Continuous Features":
            configuration["uses_scaled_data"]
    }

    model_results.append(result_row)

    print(f"{model_name} completed.")
    print(f"Saved to: {model_file_path}")

    # Results Table

results_df = pd.DataFrame(model_results)

metric_columns = [
    "ML Model Name",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

assignment_results_df = results_df[metric_columns].copy()

for column in ["Accuracy","AUC","Precision","Recall","F1","MCC"]:
    assignment_results_df[column] = (assignment_results_df[column].round(4))


results_file = MODEL_DIR / "evaluation_results.csv"

assignment_results_df.to_csv(
    results_file,
    index=False
)

print("Evaluation results saved to:", results_file)

print(
    assignment_results_df.to_string(
        index=False
    )
)

# Best Model

for metric in ["Accuracy","AUC","Precision","Recall","F1","MCC"]:
    
    best_index = results_df[metric].idxmax()
    best_model = results_df.loc[best_index,"ML Model Name"]
    best_value = results_df.loc[best_index,metric]

    print(f"Best {metric}: "f"{best_model} ({best_value:.4f})")

# Confusion Matrix

fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(16, 9))

axes = axes.flatten()

for index, model_name in enumerate(model_configurations.keys()):
    cm = confusion_matrix(y_test,model_predictions[model_name])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[index]
    )

    axes[index].set_title(model_name)
    axes[index].set_xlabel("Predicted Class")
    axes[index].set_ylabel("Actual Class")

axes[-1].axis("off")

plt.suptitle("Confusion Matrices for Classification Models",fontsize=16)
plt.tight_layout()
plt.show()

#Classification report

for model_name in model_configurations.keys():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        classification_report(
            y_test,
            model_predictions[model_name],
            target_names=[
                "No Stroke",
                "Stroke"
            ],
            zero_division=0
        )
    )

scaler_file = MODEL_DIR / "scaler.pkl"

with open(scaler_file, "wb") as file:
    pickle.dump(scaler, file)

print("Scaler saved to:", scaler_file)

#Preprocessing Metadata

preprocessing_metadata = {
    "identifier_column": identifier_column,
    "target_column": target_column,
    "continuous_columns": continuous_columns,
    "binary_columns": binary_columns,
    "categorical_columns": categorical_columns,
    "encoded_feature_columns": X_train.columns.tolist(),
    "bmi_median": bmi_median,
    "scaled_models": [
        model_name
        for model_name, configuration
        in model_configurations.items()
        if configuration["uses_scaled_data"]
    ],
    "unscaled_models": [
        model_name
        for model_name, configuration
        in model_configurations.items()
        if not configuration["uses_scaled_data"]
    ]
}

metadata_file = MODEL_DIR / "preprocessing_metadata.pkl"

with open(metadata_file, "wb") as file:
    pickle.dump(
        preprocessing_metadata,
        file
    )

print("Preprocessing metadata saved to:", metadata_file)

#Create Test.csv
test_data = df.iloc[test_indices].drop(columns=[target_column]).copy()

test_data.to_csv(PROJECT_DIR / "test_data.csv",index=False)

print("test_data.csv shape:", test_data.shape)
print("test_data.csv created successfully.")

#Save Test Lables
test_labels = pd.DataFrame({"id": df.iloc[test_indices]["id"].values,"stroke": y_test.values})

test_labels.to_csv(MODEL_DIR / "test_labels.csv",index=False)

print("Test labels saved successfully.")

#Mandatory Check

required_files = [
    "model/logistic_regression.py",
    "model/decision_tree.py",
    "model/knn.py",
    "model/naive_bayes.py",
    "model/random_forest.py",

    "model/logistic_regression.pkl",
    "model/decision_tree.pkl",
    "model/knn.pkl",
    "model/naive_bayes.pkl",
    "model/random_forest.pkl",

    "model/scaler.pkl",
    "model/preprocessing_metadata.pkl",
    "model/evaluation_results.csv",
    "model/test_labels.csv",

    "test_data.csv"
]

all_files_exist = True

for file_name in required_files:
    file_exists = Path(file_name).exists()

    print(
        f"{file_name}: "
        f"{'FOUND' if file_exists else 'MISSING'}"
    )

    if not file_exists:
        all_files_exist = False

print("\nFinal artifact validation:", all_files_exist)

#Reload Models

for model_name, configuration in model_configurations.items():

    saved_path = MODEL_DIR / configuration["file_name"]

    with open(saved_path, "rb") as file:
        reloaded_model = pickle.load(file)

    model_X_test = configuration["X_test"]

    original_predictions = model_predictions[model_name]

    reloaded_predictions = reloaded_model.predict(
        model_X_test
    )

    predictions_match = np.array_equal(
        original_predictions,
        reloaded_predictions
    )

    print(
        f"{model_name}: "
        f"{'PASS' if predictions_match else 'FAIL'}"
    )