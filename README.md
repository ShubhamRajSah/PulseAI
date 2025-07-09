# 🧠 PulseAI – Diabetes Risk Predictor

*PulseAI* is a smart, ML-powered web application that predicts a user's risk of diabetes based on health parameters. Built with Streamlit and backed by a Random Forest model, it provides clear predictions alongside SHAP-based visual explanations — making results transparent and reliable.

---

## 🚀 Features

- 🧪 Predicts diabetes risk using a trained ML model
- 📊 Displays model confidence with intuitive color-coded feedback
- 🔍 Explains predictions with SHAP bar plots
- 🌐 Interactive UI built with Streamlit
- 🧰 Clean architecture and easy to run locally

---


## 🛠 Tech Stack

- Python 3.13.5
- Streamlit for UI & deployment
- scikit-learn (Random Forest Classifier)
- SHAP for model explainability
- pandas + matplotlib for data handling & plots

---

## 📷 Preview


![App UI](![alt text](image.png))  
User interface built with Streamlit.

![SHAP Plot](![alt text](image-1.png))  
SHAP-based feature importance for explainability.

---

📁 Project Structure

`
pulseai/
│
├── streamlit_app.py            # Main app script
├── model/
│   └── pulse_model.pkl         # Trained ML model
├── data/
│   └── diabetes.csv            # Dataset used for training
├── requirements.txt            # Python dependencies
├── packages.txt                # System-level dependencies (build-essential)
├── .streamlit/
│   └── config.toml             # App configuration
└── README.md                   # Project overview
`



## 📦 How to Run Locally

1. *Clone the repo*
   ```bash
   git clone https://github.com/ShubhamRajSah/pulseai.git
   cd pulseai

2.Install dependencies
pip install -r requirements.txt

3 Run the app
streamlit run streamlit_app.py


📊 Model & Dataset

- Algorithm: Random Forest Classifier
- Dataset: Pima Indians Diabetes Dataset (binary classification)

The model is trained on health metrics like glucose level, BMI, age, pregnancies, etc.

---

🧠 Model Explainability with SHAP

We use SHAP (SHapley Additive exPlanations) to give transparent insights into how each input feature affects the model's prediction. This increases trust and interpretability, especially for medical applications.

---

🔗 Live Demo

👉 Try PulseAI Live: https://pulseai-gwm8tcaydb8mmjnrxxqg9x.streamlit.app/
No login required — just input your details and get instant results.

---

🏁 Version

v1.0.0 – Initial public release

---

📬 Contact

Built with ❤ by Shubham Raj Sah  
Drop a ⭐ if you found this helpful!
`
