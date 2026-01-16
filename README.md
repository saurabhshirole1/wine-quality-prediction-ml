# 🍷 Wine Quality Prediction using Machine Learning

🔗 **Live App:**  
👉 https://wine-quality-prediction-ml.streamlit.app/

---

## 📌 Project Overview

This project is an **end-to-end Machine Learning application** that predicts wine quality as **Average**, **Good**, or **Excellent** using physicochemical properties of wine.

The project covers the complete ML lifecycle:

- Data cleaning & exploration  
- Feature engineering  
- Model training & tuning  
- Model serialization  
- Interactive web app using **Streamlit**  
- Deployment on **Streamlit Cloud**

---

## 🎯 Problem Statement

Wine quality depends on multiple chemical properties such as acidity, alcohol, sulphates, and sulfur dioxide levels.  
Manually assessing quality is time-consuming and subjective.

**Goal:**  
Build a machine learning model that can accurately classify wine quality and provide an easy-to-use web interface for real-time predictions.

---

## 🧪 Dataset Description

The dataset contains physicochemical properties of red and white wines.

### 🔹 Features Used
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  
- Wine color (Red / White)

### 🔹 Target Variable
**Wine Quality Category**
- Average  
- Good  
- Excellent  

---

## 🧹 Data Preprocessing & EDA

- Removed duplicate records  
- Verified absence of missing values  
- Encoded categorical variable (`color`)  
- Performed distribution analysis  
- Analyzed feature correlations with wine quality  
- Created balanced quality categories from numeric scores  

---

## ⚖️ Class Imbalance Handling

The dataset was **imbalanced**, with fewer high-quality wines.

To address this:
- Used **RandomOverSampler** on the training data  
- Ensured fair learning across all classes  

---

## 🧠 Machine Learning Models Trained

Multiple models were trained and evaluated:

- K-Nearest Neighbors (KNN)  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- **Random Forest (Final Model)** ✅  

### 🔹 Final Model Selection
- **Random Forest Classifier**  
- Chosen based on overall accuracy and robustness  
- Hyperparameters tuned using **GridSearchCV**  

---

## 🌲 Final Model Configuration

- Algorithm: **Random Forest Classifier**  
- Class weighting applied  
- Optimized depth and number of estimators  
- Model serialized using **pickle**  

---

## 📊 Model Evaluation

- Accuracy comparison across models  
- Confusion matrix for multi-class classification  
- Classification report (precision, recall, F1-score)  

---

## 🖥️ Streamlit Web Application

An interactive **Streamlit dashboard** was built to allow users to:

- Input wine properties using sliders  
- Select wine color  
- Predict wine quality instantly  
- View input summary clearly  

### 🔹 UI Highlights
- Clean & minimal interface  
- Sidebar-based input controls  
- Instant prediction output  
- Deployed on **Streamlit Cloud**  

---

## ☁️ Deployment

- Source code hosted on **GitHub**  
- Application deployed using **Streamlit Cloud**  
- Publicly accessible live demo  

---

## 📂 Project Structure

```
wine-quality-prediction/
│
├── app
│   └── final_app.py       # Streamlit application
├── script         
│   └── final_model.py   # Model training & evaluation code
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── data/
│   └── winequality.csv
│   └── wine_quality_categorized.csv
│
├── model/
│   ├── rf_model.pkl       # Trained Random Forest model
│   └── scaler.pkl         # Feature scaler

```

⚙️ Installation & Usage

1️⃣ Clone the repository

```
git clone https://github.com/saurabhshirole1/wine-quality-prediction.git

cd wine-quality-prediction

```

2️⃣ Install dependencies

```
pip install -r requirements.txt

```

3️⃣ Run the Streamlit app

```
streamlit run app/final_app.py

```

🛠️ Technologies Used
* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib, Seaborn
* Streamlit
* Git & GitHub
* Streamlit Cloud


## 👤 Author

**Saurabh**  
Machine Learning & Data Science Enthusiast  

🔗 [LinkedIn](https://www.linkedin.com/in/saurabhshirole1/) | ✍️ [Medium Blogs](https://medium.com/@saurabhshirole1)





