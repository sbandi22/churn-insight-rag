# Deep Sequence Models on Behavioral Event Streams for Churn

**Authors:** Y. Tanaka, F. Moreau (synthetic)
**Venue:** Workshop on Deep Learning for Customer Analytics (synthetic, 2024)

## Abstract

Aggregated features discard the temporal order of customer behavior. We model
raw event streams (logins, feature usage, payments, support contacts) with LSTM
and Temporal Convolutional Network (TCN) architectures to predict churn directly
from sequences. On a SaaS dataset of 90,000 accounts with up to 18 months of
daily events, the TCN achieved an AUC of 0.88, modestly beating a strong
LightGBM model on hand-engineered features (0.85) while removing most manual
feature engineering. Attention weights highlighted a recurring pre-churn pattern:
a gradual decline in active-feature breadth followed by a final burst of support
tickets. Deep models excelled most on high-activity enterprise accounts with rich
event histories and struggled on sparse, low-engagement SMB accounts, where
simpler models remained competitive.

## Key Findings

- Sequence models capture "slow fade" churn patterns that snapshot features miss.
- Benefits concentrate in data-rich enterprise segments; ROI is weaker for SMB.
- Attention/saliency maps give interpretable early-warning signals over time.

## Practical Takeaway

Reserve deep sequence models for high-value, data-rich accounts; keep tree-based
models for the long tail of low-activity customers.
