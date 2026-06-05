# Smart Diabetes Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?logo=streamlit)](https://diabetes-predictor-tumtwcikhbk7q6l9cwtgxs.streamlit.app/)

**Try it live:** [https://diabetes-predictor-tumtwcikhbk7q6l9cwtgxs.streamlit.app/](https://diabetes-predictor-tumtwcikhbk7q6l9cwtgxs.streamlit.app/)

**GitHub:** [https://github.com/Tanvi2230/Smart-Diabetes-predictor](https://github.com/Tanvi2230/Smart-Diabetes-predictor)

> **Disclaimer:** This tool is intended for educational and screening support purposes only. It is not a substitute for professional medical diagnosis or clinical advice.

---

Smart Diabetes Predictor is an end-to-end machine learning project that predicts whether a patient is likely to be diabetic based on eight medical indicators. It compares multiple supervised learning algorithms, automatically selects the best model, and exposes a polished Streamlit interface for single and bulk predictions.

## Features

- Preprocessing pipeline with missing-value handling, scaling, and feature selection
- Model comparison across Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine
- Automatic best-model selection using F1-first ranking
- Probability score, risk classification, and short explanation for each prediction
- Visual analytics: correlation heatmap, feature importance, distributions, confusion matrix
- Local prediction history logging
- Bulk CSV prediction through the UI
- Fully responsive — works on desktop and mobile

## Project Structure

```text
diabetes-predictor/
│
├── data/
│   └── diabetes.csv
├── models/
│   ├── model.pkl
│   ├── model_metrics.json
│   └── plots/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── train_model.py
│   └── utils.py
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Windows fallback helper:

```powershell
.\setup_windows.ps1
```

3. Train the model:

```bash
python -m src.train_model
```

4. Launch the UI:

```bash
streamlit run app/app.py
```

## Input Parameters

| Parameter | Description | Range |
|---|---|---|
| Pregnancies | Number of pregnancies | 0 – 20 |
| Glucose | Plasma glucose concentration (mg/dL) | 0 – 250 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 0 – 140 |
| SkinThickness | Triceps skin fold thickness (mm) | 0 – 100 |
| Insulin | 2-hour serum insulin (mu U/ml) | 0 – 900 |
| BMI | Body mass index (kg/m²) | 0 – 70 |
| DiabetesPedigreeFunction | Diabetes pedigree function score | 0.05 – 3.0 |
| Age | Age in years | 10 – 100 |

## Example

**Input:**
```
Pregnancies: 4
Glucose: 148
BloodPressure: 72
SkinThickness: 35
Insulin: 0
BMI: 33.6
DiabetesPedigreeFunction: 0.627
Age: 50
```

**Output:**
```
Prediction: Diabetic
Probability score: 78.00%
Risk level: High
Explanation: Top influencing factors: Glucose (higher risk, impact 0.682),
             BMI (higher risk, impact 0.311), Age (higher risk, impact 0.248).
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose your GitHub repo.
4. Set the main file path to `app/app.py`.
5. Click **Deploy**.

Streamlit Cloud will automatically use `requirements.txt` for dependencies and `runtime.txt` to pin the Python version.

Before pushing, keep `models/model.pkl` and `models/model_metrics.json` in the repo so the app loads instantly without retraining.
