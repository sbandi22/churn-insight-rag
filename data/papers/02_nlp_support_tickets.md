# Mining Support Tickets with NLP to Predict Churn

**Authors:** S. Bandyopadhyay, K. Larsson (synthetic)
**Venue:** Proceedings on Applied NLP for Business (synthetic, 2024)

## Abstract

Customer support interactions contain early, high-signal indicators of churn
intent that structured usage data often misses. We apply transformer-based text
classification to 180,000 support tickets from a B2B SaaS provider, labeling
each customer as churned or retained within the following 120 days. A fine-tuned
sentence-embedding model combined with sentiment and topic features achieved an
F1 of 0.74 for churn prediction, outperforming a usage-only gradient-boosted
baseline (F1 0.62). The most predictive linguistic signals were repeated
mentions of pricing, escalations to "cancel" or "downgrade", unresolved
tickets reopened more than twice, and a sharp drop in message politeness. We
argue that NLP on support text should be a first-class feature source in any
modern churn model, especially for relationship-driven enterprise accounts.

## Key Findings

- Negative sentiment trajectory over a 30-day window is more predictive than any
  single negative ticket.
- Tickets mentioning competitors precede churn in 38% of enterprise cases.
- Combining text features with usage features yields the best overall model.

## Practical Takeaway

Pipe support-ticket text into the churn feature store; even simple keyword and
sentiment features materially improve enterprise churn detection.
