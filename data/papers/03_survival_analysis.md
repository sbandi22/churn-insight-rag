# Survival Analysis for Subscription Churn and Customer Lifetime

**Authors:** D. Nguyen, P. Castellano (synthetic)
**Venue:** International Journal of Forecasting Practice (synthetic, 2023)

## Abstract

Most churn models answer "who will churn" but not "when". We frame subscription
churn as a time-to-event problem and apply Cox proportional-hazards and
accelerated-failure-time models to a cohort of 65,000 telecom subscribers
observed over 24 months. The survival framing produces calibrated, time-varying
churn-risk curves and enables direct estimation of expected customer lifetime
value. Month-to-month contract type carried the largest hazard ratio (HR 2.9)
relative to two-year contracts, while fiber-optic customers without tech-support
add-ons showed elevated early-tenure hazard. Survival curves revealed a critical
risk window between months 2 and 5, after which hazard declined sharply for
surviving customers. We recommend onboarding and engagement interventions
concentrated in this early window.

## Key Findings

- Hazard is highly non-constant: early-tenure customers are far more fragile.
- Two-year contracts reduce instantaneous churn hazard by roughly 65%.
- Time-to-event models give finance teams credible CLV and revenue-at-risk numbers.

## Practical Takeaway

Use survival models to time interventions, not just target them; the first
90 days of tenure deserve the heaviest retention investment.
