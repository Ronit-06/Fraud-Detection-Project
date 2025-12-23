# 💳 Credit Card Fraud Detection (ML + Streamlit)

A project that trains a machine learning model to detect potentially fraudulent credit card transactions and deploys it as an interactive **Streamlit** web app.

The app allows users to enter transaction details (distance, typical spending ratio, chip/PIN usage, online order, repeat retailer) and returns:
- A **fraud probability** score
- A **Fraudulent / Legit** prediction
- A simple **risk level** indicator

---

## ✨ Features

- End-to-end ML workflow in a **Jupyter Notebook**
  - Data loading and basic exploration
  - Train/test split with stratification
  - Preprocessing using `scikit-learn` Pipelines 
  - Logistic Regression model
  - Evaluation using classification report, confusion matrix, and PR-AUC
- **Streamlit** web app
  - User-friendly input labels 
  - Fraud probability + risk level display
  - Clear “How to interpret this” section for non-technical users

---

## 🧠 Dataset & Features

The dataset contains transaction-level features:

- `distance_from_home` – distance from home where the transaction happened  
- `distance_from_last_transaction` – distance from the last transaction  
- `ratio_to_median_purchase_price` – purchase amount compared to the user’s median purchase  
- `repeat_retailer` – whether the transaction is from the same retailer  
- `used_chip` – whether the card chip was used  
- `used_pin_number` – whether a PIN was used  
- `online_order` – whether the transaction was online  
- `fraud` – target label (1 = fraud, 0 = not fraud)

📌 **Dataset**  
- https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn (Pipeline + Logistic Regression)
- Streamlit
- Matplotlib
- joblib

---

## 📸 Application Screenshots

<img width="818" height="906" alt="image" src="https://github.com/user-attachments/assets/429bc588-b0da-492a-baea-408062bf4563" />


<img width="716" height="918" alt="image" src="https://github.com/user-attachments/assets/3ecf7aa2-3cb9-4464-90f3-ffa7141753b3" />

