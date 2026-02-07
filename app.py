import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Heart Disease Prediction - Model Comparison")

uploaded_file = st.file_uploader("Upload heart.csv", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Derived feature
    df["Cholesterol_per_Age"] = df["Cholesterol"] / df["Age"]

    # Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("HeartDisease", axis=1)
    y = df_encoded["HeartDisease"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scaling (for KNN, LR, NB)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = []

    for name, model in models.items():

        # Choose scaled or unscaled
        if name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    comparison_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

    st.subheader("Model Comparison")
    st.dataframe(comparison_df)

    # Best Model
    best_model = comparison_df.iloc[0]["Model"]
    st.success(f"Best Model Based on Accuracy: {best_model}")
    # Retrain best model to show confusion matrix
    best_model_name = best_model
    best_model_obj = models[best_model_name]

    if best_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        best_model_obj.fit(X_train_scaled, y_train)
        y_pred_best = best_model_obj.predict(X_test_scaled)
    else:
        best_model_obj.fit(X_train, y_train)
        y_pred_best = best_model_obj.predict(X_test)

    st.subheader("Confusion Matrix - Best Model")

    cm = confusion_matrix(y_test, y_pred_best)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    st.pyplot(fig2)


    # Plot Accuracy
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=comparison_df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.markdown("---")
    st.caption("Machine Learning Assignment 2 - Heart Disease Prediction")

