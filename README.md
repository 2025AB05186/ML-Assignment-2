# Machine Learning Assignment 2  
## Heart Disease Prediction – Model Comparison using Streamlit

---

## a) Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease based on patient medical attributes.

The application trains six classification algorithms on the same dataset and evaluates their performance using multiple evaluation metrics. The results are presented through an interactive Streamlit web application.

---

## b) Dataset Description

Dataset Used: Heart Disease Dataset  

Number of Instances: 918  
Number of Input Features: 12 (after feature engineering)  
Target Variable: `HeartDisease` (Binary: 0 = No Disease, 1 = Disease)

Feature Engineering:
An additional derived feature was created:
- `Cholesterol_per_Age = Cholesterol / Age`

Categorical features were encoded using one-hot encoding.

The dataset contains both numerical and categorical attributes related to patient health parameters such as:
- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- MaxHR
- ExerciseAngina
- ST_Slope
etc.

---

## c) Models Used and Evaluation Metrics

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.8826 | 0.9318 | 0.8731 | 0.9213 | 0.8966 | 0.7626 |
| Decision Tree | 0.8174 | 0.8145 | 0.8295 | 0.8425 | 0.8360 | 0.6302 |
| KNN | 0.8957 | 0.9427 | 0.8992 | 0.9134 | 0.9063 | 0.7887 |
| Naive Bayes | 0.9043 | 0.9427 | 0.9134 | 0.9134 | 0.9134 | 0.8066 |
| Random Forest | 0.8913 | 0.9429 | 0.8923 | 0.9134 | 0.9027 | 0.7799 |
| XGBoost | 0.8696 | 0.9298 | 0.8819 | 0.8819 | 0.8809 | 0.7369 |

---

## d) Observations and Model Performance Analysis

### Logistic Regression
Performed strongly with high AUC and balanced precision-recall. Suitable for linearly separable relationships.

### Decision Tree
Showed lower performance compared to other models. Likely affected by high variance and overfitting.

### K-Nearest Neighbors
Performed well after feature scaling. Demonstrated strong local learning capability.

### Naive Bayes
Achieved the highest accuracy and MCC score. The independence assumption worked effectively for this dataset.

### Random Forest
Improved over a single decision tree by reducing variance. Performance was competitive but slightly lower than Naive Bayes.

### XGBoost
Delivered strong AUC performance but slightly lower accuracy compared to Naive Bayes and KNN.

---

## Conclusion

Among all implemented models, Naive Bayes achieved the best overall performance based on Accuracy and MCC score. The results indicate that probabilistic modeling performed effectively on this dataset.

The Streamlit application provides an interactive interface to upload the dataset, train models, and visualize performance comparisons.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  
- Streamlit  

---
