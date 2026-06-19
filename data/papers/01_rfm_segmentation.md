# RFM-Based Customer Segmentation for Churn Prediction

**Authors:** A. Mehta, L. Fernandez, R. Okoye (synthetic)
**Venue:** Journal of Marketing Analytics (synthetic, 2023)

## Abstract

Recency-Frequency-Monetary (RFM) analysis remains one of the most interpretable
foundations for churn prediction. In this study we segment a base of 42,000
subscription customers into nine RFM tiers and measure 90-day churn within each
tier. Customers in the low-recency / low-frequency quadrant churned at 4.7x the
rate of high-value loyal customers. We show that augmenting a logistic-regression
churn model with RFM tier indicators improved AUC from 0.71 to 0.79 without any
additional behavioral data. RFM scoring is computationally cheap, updates daily,
and gives non-technical stakeholders a transparent way to prioritize retention
spend. We recommend RFM as a strong baseline before investing in heavier models.

## Key Findings

- Recency is the single strongest RFM dimension for predicting near-term churn.
- "At risk" tier (high past value, declining recency) is the most profitable
  segment to target with win-back campaigns, with an estimated 3.2x ROI.
- RFM tiers are stable enough to recompute weekly rather than in real time.

## Practical Takeaway

Start churn programs with RFM segmentation to identify high-value declining
customers, then layer machine-learning models on top for marginal lift.
