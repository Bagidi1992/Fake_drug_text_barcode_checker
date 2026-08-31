import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

dataset_path = Path(
    "/storage/emulated/0/NAFDAC_Registered_Products_Database.csv"
)

df = pd.read_csv(dataset_path)
df = df.rename(columns={
    "Product Name": "product_name",
    "Active Ingredients": "active_ingredients",
    "Product Category": "product_category",
    "NRN (NAFDAC Reg No)": "nafdac_reg_no",
    "Form": "form",
    "ROA (Route of Admin)": "route_of_admin",
    "Strengths": "strengths",
    "Status": "status"
})

def clean_text(value):
    if pd.isna(value):
        return ""
    
    return str(value).strip().upper()
text_columns = [
    "product_name",
    "active_ingredients",
    "product_category",
    "nafdac_reg_no",
    "form",
    "route_of_admin",
    "strengths",
    "status"
]

for column in text_columns:
    df[column] = df[column].apply(clean_text)

drug_df = df[
    df["product_category"] == "DRUGS"
].copy()

drug_df = drug_df.reset_index(drop=True)

drug_df = drug_df.drop_duplicates(
    subset=[
        "product_name",
        "nafdac_reg_no"
    ]
)

drug_df = drug_df.reset_index(drop=True)

cleaned_path = "/storage/emulated/0/download/cleaned_drug_data.csv"

drug_df.to_csv(
    cleaned_path,
    index=False
)

print("Cleaned dataset saved successfully.")
    
    
labelled_records = []

for index, row in drug_df.iterrows():

    base = row.to_dict()

    # Genuine example

    genuine = base.copy()

    genuine["source_id"] = index
    genuine["label"] = "Genuine"
    genuine["perturbation_type"] = "None"

    genuine["name_similarity"] = 1.0
    genuine["registration_match"] = 1
    genuine["ingredient_match"] = 1
    genuine["form_match"] = 1
    genuine["route_match"] = 1
    genuine["strength_match"] = 1
    genuine["category_match"] = 1
    genuine["status_valid"] = (
        1 if row["status"] == "ACTIVE" else 0
    )

    labelled_records.append(genuine)

    # Suspicious: altered NAFDAC no

    suspicious = base.copy()

    suspicious["source_id"] = index
    suspicious["nafdac_reg_no"] = (
        str(row["nafdac_reg_no"]) + "X"
    )

    suspicious["label"] = "Suspicious"
    suspicious["perturbation_type"] = (
        "Altered registration number"
    )

    suspicious["name_similarity"] = 1.0
    suspicious["registration_match"] = 0
    suspicious["ingredient_match"] = 1
    suspicious["form_match"] = 1
    suspicious["route_match"] = 1
    suspicious["strength_match"] = 1
    suspicious["category_match"] = 1
    suspicious["status_valid"] = (
        1 if row["status"] == "ACTIVE" else 0
    )

    labelled_records.append(suspicious)

    # Suspicious: altered name

    suspicious = base.copy()

    suspicious["source_id"] = index

    suspicious["product_name"] = (
        str(row["product_name"]) + " X"
    )

    suspicious["label"] = "Suspicious"
    suspicious["perturbation_type"] = (
        "Altered product name"
    )

    suspicious["name_similarity"] = 0.70
    suspicious["registration_match"] = 1
    suspicious["ingredient_match"] = 1
    suspicious["form_match"] = 1
    suspicious["route_match"] = 1
    suspicious["strength_match"] = 1
    suspicious["category_match"] = 1
    suspicious["status_valid"] = (
        1 if row["status"] == "ACTIVE" else 0
    )

    labelled_records.append(suspicious)


labelled_df = pd.DataFrame(
    labelled_records
)

print(labelled_df.head())

additional_records = []

for index, row in drug_df.iterrows():

    suspicious = row.to_dict()

    suspicious["source_id"] = index

    suspicious["active_ingredients"] = (
        str(row["active_ingredients"])
        + " UNKNOWN"
    )

    suspicious["label"] = "Suspicious"

    suspicious["perturbation_type"] = (
        "Altered active ingredient"
    )

    suspicious["name_similarity"] = 1.0
    suspicious["registration_match"] = 1
    suspicious["ingredient_match"] = 0
    suspicious["form_match"] = 1
    suspicious["route_match"] = 1
    suspicious["strength_match"] = 1
    suspicious["category_match"] = 1

    suspicious["status_valid"] = (
        1 if row["status"] == "ACTIVE" else 0
    )

    additional_records.append(suspicious)


additional_df = pd.DataFrame(
    additional_records
)

labelled_df = pd.concat(
    [labelled_df, additional_df],
    ignore_index=True
)

print("Total labelled records:", len(labelled_df))


labelled_df["label"].value_counts()
labelled_df["perturbation_type"].value_counts()
labelled_path = (
    "/storage/emulated/0/download/labelled_genuine_suspicious.csv"
)

labelled_df.to_csv(
    labelled_path,
    index=False
)

print("Labelled dataset saved.")


features = [
    "name_similarity",
    "registration_match",
    "ingredient_match",
    "form_match",
    "route_match",
    "strength_match",
    "category_match",
    "status_valid"
]

X = labelled_df[features]

y = labelled_df["label"]

print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")



y_pred = model.predict(
    X_test
)

print(y_pred[:20])


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Suspicious",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Suspicious",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Suspicious",
    zero_division=0
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "Genuine",
        "Suspicious"
    ]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Genuine",
        "Suspicious"
    ],
    yticklabels=[
        "Genuine",
        "Suspicious"
    ]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "/storage/emulated/0/download/confusion_matrix.png",
    dpi=300
)

plt.show()


importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(
    ascending=True
)

plt.figure(figsize=(8, 5))

importance.plot(
    kind="barh"
)

plt.xlabel(
    "Importance"
)

plt.title(
    "Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "/storage/emulated/0/download/feature_importance.png",
    dpi=300
)

plt.show()


evaluation_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

evaluation_results.to_csv(
    "/storage/emulated/0/download/evaluation_results.csv",
    index=False
)

print(evaluation_results)


report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

with open(
    "/storage/emulated/0/download/classification_report.txt",
    "w"
) as file:

    file.write(report)

print(report)


model_package = {
    "model": model,
    "features": features
}

joblib.dump(
    model_package,
    "/storage/emulated/0/download/drug_checker_model.pkl"
)

print("Trained model saved successfully.")