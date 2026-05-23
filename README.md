# Bank Customer Churn Risk Manager

A production-ready Streamlit application for predicting bank customer churn using
Logistic Regression, XGBoost (+ CTGAN), and Optuna-tuned XGBoost models, with
SHAP-based explainability.

## Features

- **Overview Dashboard** — model comparison and portfolio overview
- **Single Customer Analysis** — interactive risk prediction with SHAP explanations
- **Batch Prediction** — CSV upload for portfolio-wide scoring
- **Risk Segmentation** — automatic Low/Medium/High risk classification
- **Action Recommendations** — rule-based retention strategies

## Methodology

| Step | Technique |
|---|---|
| Baseline model | Logistic Regression (statsmodels) |
| Class balancing | CTGAN (Conditional Tabular GAN) |
| High-performance model | XGBoost |
| Hyperparameter optimization | Optuna (Bayesian) |
| Explainability | SHAP (TreeExplainer) |

## Final Model Performance

| Metric | LR | XGBoost + CTGAN | XGBoost + Optuna |
|---|---|---|---|
| Accuracy | 0.90 | 0.97 | 0.97 |
| Recall (churn) | 0.54 | 0.86 | **0.89** |
| F1-Score | 0.64 | 0.89 | **0.90** |
| AUC | 0.917 | 0.991 | 0.990 |

## Dataset

[Credit Card Customers — Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
10,127 customers × 21 features, target: `Attrition_Flag`.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

##  Project Team & Roles

This project was developed for the Probability course led by Sinem Bozkurt Keser at Eskişehir Osmangazi University.

* **Ece Gül Yüksel:** Technical Lead. Architected the complete machine learning pipeline and developed the interactive Streamlit web application. Conducted the core literature research and authored the "Experimental Results" and "Final Conclusions" sections of the academic report.

* **Selin İnce:** Research & Documentation Lead. Compiled and structured the academic report based on the analytical outputs. Responsible for drafting the introductory sections, formatting the literature review, documenting the hypothesis definitions, and designing the final presentation deck.

* **Didem Hışırcı:** Research & Documentation Lead. Compiled and structured the academic report based on the analytical outputs. Responsible for drafting the introductory sections, formatting the literature review, documenting the hypothesis definitions, and designing the final presentation deck.