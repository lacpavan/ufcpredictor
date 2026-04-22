# 🥊 UFC Fight Predictor

 Machine Learning system for analyzing and predicting UFC fight outcomes based on historical data.

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHBlMm1mcmI0dWR4cHJkaWk2a2poN2ZvNDN4aDM2eHI4Y2czZG03eCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/N6o559ZkDsozTnQDcE/giphy.gif" width="600"/>
</p>

<p align="center">
  Predicting UFC fight outcomes using Machine Learning
</p>

<p align="center">
  <a href="https://ufc-predictor.streamlit.app/">Live App</a> •
  <a href="https://github.com/lacpavan/ufcpredictor">GitHub Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg"/>
  <img src="https://img.shields.io/badge/App-Streamlit-red.svg"/>
  <img src="https://img.shields.io/badge/Status-Active-success.svg"/>
</p>

---

## 📌 Overview

As an MMA fan, I wanted to move beyond opinions and explore whether it's possible to predict fight outcomes using data.

This project uses historical UFC data to train a classification model that estimates the probability of a fighter winning a match.

---

## 🧠 Key ML Decisions

This project was designed with real-world Machine Learning challenges in mind:

- Temporal train-test split to avoid data leakage and simulate real-world predictions  
- Feature engineering based on fighter comparison (Red vs Blue corner)  
- Baseline model comparison before selecting the final model  
- Focus on model reliability, not just performance  
- Use of model interpretability with SHAP  

---

## ⚙️ Tech Stack

- Python  
- pandas, NumPy  
- scikit-learn  
- XGBoost / Gradient Boosting  
- SHAP (model explainability)  
- Streamlit (for interactive app)  

---

## 📊 Model Approach

The model predicts fight outcomes based on pre-fight features.

Key steps:

1. Data preprocessing  
2. Feature engineering (fighter vs opponent comparison)  
3. Temporal split (train on past fights, test on future fights)  
4. Model training and evaluation  
5. Model explainability using SHAP  
6. Prediction and visualization  

---

## 📈 Evaluation Metrics

- Accuracy  
- ROC AUC  
- Log Loss  

These metrics help evaluate both performance and prediction confidence.

---

## 🔍 Model Interpretability

To better understand model decisions, SHAP (SHapley Additive exPlanations) is used.

This allows:

- Identifying which features most influence predictions  
- Improving model transparency  
- Supporting more reliable decision-making  

Interpretability is essential to ensure trust in Machine Learning systems, especially in real-world applications.

---

## 🖥️ Application

The project includes an interactive Streamlit app where users can:

- Select two fighters  
- Compare their attributes  
- Get a predicted winner  
- Visualize prediction probabilities  
- Understand feature impact using SHAP explanations  

---

## 📂 Project Structure

ufcpredictor/

├── src/ufc_predictor/  
│   ├── data.py  
│   ├── features.py  
│   ├── modeling.py  
│   ├── train.py  
│   └── config.py  

├── app.py  
├── requirements.txt  
└── README.md  

---

## ▶️ How to run:

1. Install dependencies

pip install -r requirements.txt

2. Train the model

python -m src.ufc_predictor.train

3. Run the app

streamlit run app.py

---

## ⚠️ Limitations

- Predictions are based only on historical data  
- Does not include real-time factors such as injuries, weight cuts, or strategy  
- Model performance depends on data quality  

---

## 🚀 Future improvements

- Improve SHAP visualizations for better interpretability  
- Add API with FastAPI for real-time predictions  
- Deploy model to production environment  
- Add model monitoring and versioning  

---

## 💡 Why this project matters

This project goes beyond a simple ML model.

It focuses on:

- Avoiding common ML mistakes such as data leakage  
- Structuring a reproducible pipeline  
- Simulating real-world prediction scenarios  
- Making model decisions interpretable  

---

## 🤝 Let's Connect

If you're interested in Machine Learning, AI, or data-driven systems, feel free to connect!

---

Made by @lacpavan
