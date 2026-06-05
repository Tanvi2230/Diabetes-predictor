# Smart Diabetes Predictor

Smart Diabetes Predictor is an end-to-end machine learning project that predicts whether a patient is likely to be diabetic based on eight medical indicators. It compares multiple supervised learning algorithms, automatically chooses the best model, saves the trained bundle, and exposes a polished Streamlit interface for single and bulk predictions.

## Features

- Preprocessing pipeline with missing-value handling, scaling, and feature selection
- Model comparison across Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine
- Automatic best-model selection using F1-first ranking
- Probability score, risk classification, and short explanation for each prediction
- Visual analytics: correlation heatmap, feature importance, distributions, confusion matrix
- Local prediction history logging
- Bulk CSV prediction through the UI

## Project Structure

```text
diabetes-predictor/
│
├── data/
│   └── diabetes.csv
├── models/
│   └── model.pkl
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

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## Example Input

```text
Pregnancies: 4
Glucose: 148
BloodPressure: 72
SkinThickness: 35
Insulin: 0
BMI: 33.6
DiabetesPedigreeFunction: 0.627
Age: 50
```

## Example Output

```text
Prediction: Diabetic
Probability score: 78.00%
Risk level: High
Explanation: Top influencing factors: Glucose (higher risk, impact 0.682), BMI (higher risk, impact 0.311), Age (higher risk, impact 0.248).
```

## Notes

- The bundled dataset is offline-friendly and uses the same medical fields as the Pima-style diabetes schema.
- This project is intended for educational screening support and not as a substitute for professional medical diagnosis.
- If your machine has a mixed Python setup, the included `setup_windows.ps1` clears `PYTHONPATH`, uses a local `.venv`, installs requirements, and trains the initial model bundle.

## Deploy To Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Choose your GitHub repo.
4. Set the main file path to `app/app.py`.
5. Click Deploy.

If Streamlit asks for dependencies, it will automatically use `requirements.txt`.
The project also includes `runtime.txt` to pin a compatible Python version for deployment.

Recommended before pushing:

- Keep `models/model.pkl` and `models/model_metrics.json` in the repo
- Do not upload `.venv`, `.tmp`, `.matplotlib_cache`, or `__pycache__`
- Use the included `.gitignore`
