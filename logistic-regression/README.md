# 🩺 Diabetes Prediction System

An end-to-end Machine Learning project to predict whether a patient has diabetes using Logistic Regression — implemented both with Scikit-Learn and from scratch.

---

## 🚀 Project Overview

This project builds a complete ML pipeline including:

- Data cleaning & preprocessing  
- Handling invalid medical values  
- Feature scaling  
- Model training (Scikit-Learn)  
- Custom Logistic Regression implementation  
- Model evaluation & comparison  

The goal was to combine practical ML workflow with strong mathematical understanding.

---

## 📊 Dataset

**Pima Indians Diabetes Dataset**

Features include:
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function

Target:
0 → Non-Diabetic
1 → Diabetic

---

## ⚙️ What I Implemented

### 1️⃣ Data Cleaning
- Replaced invalid zero values (Glucose, BMI, Insulin, etc.) with `NaN`
- Applied median imputation (robust to skewed medical data)

### 2️⃣ Train-Test Split
- 80/20 split  
- Stratified sampling  
- Reproducible results (`random_state=42`)

### 3️⃣ Feature Scaling
- Applied `StandardScaler`
- Prevented data leakage by fitting only on training data

---

## 🤖 Models Built

### ✅ Scikit-Learn Logistic Regression
Used as a benchmark model.

### 🛠 Custom Logistic Regression (From Scratch)
Implemented using:
- Sigmoid function  
- Binary Cross-Entropy Loss  
- Gradient Descent  
- Manual weight & bias updates  

This validates deep understanding of optimization and classification mechanics.

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 🛠 Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-Learn  

---


## 🧠 Key Takeaways

- Data preprocessing significantly impacts performance  
- Feature scaling is essential for gradient-based models  
- Evaluation metrics must align with real-world cost  
- Implementing algorithms from scratch strengthens ML intuition  

---

⭐ If you found this project interesting, feel free to star the repository!
