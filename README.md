# 🏥 Insurance Charges Predictor

This project predicts medical insurance charges based on demographic and health factors (age, BMI, smoking status, etc.).

## 🚀 Features
- Exploratory Data Analysis (EDA)
- Machine Learning models (Linear, Random Forest, Gradient Boosting)
- Feature importance analysis
- Saved model pipeline for easy deployment

## 📊 Model Performance
| Model | MAE | RMSE | R² |
|--------|------|------|----|
| Gradient Boosting | 2443 | 4329 | 0.879 |

## 🧠 Key Insights
- Smoking and BMI have the largest impact on medical costs.
- Age correlates strongly with charges.
- Region plays a minor role.

## 💾 Usage
```python
import joblib, pandas as pd
model = joblib.load("insurance_gb_model.joblib")

sample = pd.DataFrame([{
    "age": 45, "sex": "male", "bmi": 31.2,
    "children": 2, "smoker": "yes", "region": "southeast"
}])
print(model.predict(sample))
