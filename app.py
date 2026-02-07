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
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("❤️ Heart Disease Prediction - ML Model Comparison")
st.markdown("### Compare 6 Classification Models on Heart Disease Dataset")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 About This App")
    st.info("""
    This application compares 6 different machine learning classification models 
    for heart disease prediction:
    
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest
    - XGBoost
    """)
    
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload heart.csv file", 
        type=["csv"],
        help="Upload your heart disease dataset in CSV format"
    )
    
    st.markdown("---")
    st.caption("Machine Learning Assignment 2")
    st.caption("Heart Disease Prediction System")

if uploaded_file:
    
    # Load data with spinner
    with st.spinner('Loading dataset...'):
        df = pd.read_csv(uploaded_file)
    
    # Dataset Overview Section
    st.header("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Positive Cases", df['HeartDisease'].sum())
    with col4:
        st.metric("Negative Cases", len(df) - df['HeartDisease'].sum())
    
    # Show dataset preview
    with st.expander("🔍 View Dataset Sample"):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Feature Engineering
    st.header("⚙️ Feature Engineering & Preprocessing")
    with st.spinner('Processing features...'):
        df["Cholesterol_per_Age"] = df["Cholesterol"] / df["Age"]
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        X = df_encoded.drop("HeartDisease", axis=1)
        y = df_encoded["HeartDisease"]
        
        st.success(f"✓ Created derived feature: Cholesterol_per_Age")
        st.success(f"✓ Encoded categorical variables")
        st.success(f"✓ Final feature count: {X.shape[1]}")
    
    st.markdown("---")
    
    # Model Training Section
    st.header("🤖 Model Training & Evaluation")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    }
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    trained_models = {}
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        # Choose scaled or unscaled
        if name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        trained_models[name] = (model, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
        })
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("All models trained successfully! ✓")
    
    st.markdown("---")
    
    # Results Section
    st.header("📊 Model Comparison Results")
    
    comparison_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    
    # Display comparison table with styling
    st.subheader("Performance Metrics Table")
    st.dataframe(
        comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1', 'MCC'])
        .highlight_max(subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], color='lightgreen')
        .format("{:.4f}", subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
        use_container_width=True
    )
    
    # Best Model Highlight
    best_model_name = comparison_df.iloc[0]["Model"]
    best_accuracy = comparison_df.iloc[0]["Accuracy"]
    
    st.markdown(f"""
    <div class="success-box">
        <h3>🏆 Best Performing Model: {best_model_name}</h3>
        <p><strong>Accuracy:</strong> {best_accuracy:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization Section
    st.header("📈 Performance Visualizations")
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(
            x="Accuracy", 
            y="Model", 
            data=comparison_df, 
            palette="viridis",
            ax=ax1
        )
        ax1.set_xlabel("Accuracy Score", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Model", fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # Add value labels on bars
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Multi-Metric Comparison")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        metrics_df = comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']]
        metrics_df.plot(kind='bar', ax=ax2, width=0.8)
        
        ax2.set_ylabel("Score", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Model", fontsize=12, fontweight='bold')
        ax2.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
    
    st.markdown("---")
    
    # Model Selection for Detailed Analysis
    st.header("🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select a model to view detailed analysis:",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )
    
    model_obj, y_pred_selected = trained_models[selected_model]
    
    # Retrain for consistency (already done, just retrieve)
    if selected_model in ["Logistic Regression", "KNN", "Naive Bayes"]:
        y_pred_detail = model_obj.predict(X_test_scaled)
    else:
        y_pred_detail = model_obj.predict(X_test)
    
    # Create columns for confusion matrix and classification report
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Confusion Matrix - {selected_model}")
        cm = confusion_matrix(y_test, y_pred_detail)
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            ax=ax3,
            cbar_kws={'label': 'Count'},
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        ax3.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
        ax3.set_ylabel("True Label", fontsize=12, fontweight='bold')
        ax3.set_title(f"Confusion Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
    
    with col2:
        st.subheader(f"Classification Report - {selected_model}")
        report = classification_report(y_test, y_pred_detail, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(
            report_df.style.background_gradient(cmap='RdYlGn')
            .format("{:.2f}"),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Key Metrics Display
    st.header("📌 Key Performance Indicators")
    
    selected_metrics = comparison_df[comparison_df['Model'] == selected_model].iloc[0]
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Accuracy", f"{selected_metrics['Accuracy']:.3f}")
    with col2:
        st.metric("AUC", f"{selected_metrics['AUC']:.3f}")
    with col3:
        st.metric("Precision", f"{selected_metrics['Precision']:.3f}")
    with col4:
        st.metric("Recall", f"{selected_metrics['Recall']:.3f}")
    with col5:
        st.metric("F1 Score", f"{selected_metrics['F1']:.3f}")
    with col6:
        st.metric("MCC", f"{selected_metrics['MCC']:.3f}")

else:
    # Welcome message when no file is uploaded
    st.info("👈 Please upload the heart.csv dataset from the sidebar to begin model comparison")
    
    st.markdown("""
    ### How to use this application:
    
    1. **Upload Dataset**: Use the file uploader in the sidebar to upload your heart.csv file
    2. **View Results**: The app will automatically train all 6 models and display results
    3. **Compare Models**: Review the performance metrics table and visualizations
    4. **Detailed Analysis**: Select any model to view its confusion matrix and classification report
    
    ### Models Included:
    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbors (KNN)
    - Naive Bayes (Gaussian)
    - Random Forest (Ensemble)
    - XGBoost (Ensemble)
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Machine Learning Assignment 2 | Heart Disease Prediction System</p>
        <p>Built with Streamlit 🎈 | Powered by Scikit-learn & XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
