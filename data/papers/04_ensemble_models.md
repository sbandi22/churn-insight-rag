# Ensemble Models for High-Accuracy Churn Prediction

**Authors:** M. Owusu, T. Becker, V. Rao (synthetic)
**Venue:** Conference on Applied Machine Learning Systems (synthetic, 2024)

## Abstract

We benchmark single learners against ensemble methods for binary churn
classification on the public Telco dataset and two larger proprietary datasets.
Gradient-boosted trees (XGBoost, LightGBM) and a stacked ensemble combining
boosting, random forests, and a logistic meta-learner consistently outperformed
individual models. On the Telco dataset the stacked ensemble reached an AUC of
0.86 and recall of 0.81 at a fixed 20% intervention budget, versus 0.78 AUC for
plain logistic regression. SHAP analysis identified contract type, tenure,
monthly charges, and internet-service type as the dominant churn drivers.
Critically, we show that calibrating predicted probabilities (isotonic
regression) is essential before using churn scores for budget allocation, as
uncalibrated boosting models systematically over-predict risk.

## Key Findings

- Stacked ensembles add 4-8 AUC points over the best single model.
- Probability calibration matters as much as raw ranking accuracy for ROI.
- SHAP makes ensembles explainable enough for retention-team adoption.

## Practical Takeaway

Use a calibrated gradient-boosting or stacked ensemble as the production model,
with SHAP explanations attached to each high-risk customer for the CS team.
