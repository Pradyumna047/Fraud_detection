# 🔍 Financial Fraud Detection System
### AML Transaction Monitoring · XGBoost + SHAP Explainability · Streamlit Demo

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.938-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Project Overview

An end-to-end machine learning system for detecting fraudulent financial 
transactions, built to mirror the architecture of production AML 
(Anti-Money Laundering) platforms used by banks and payment providers globally.

The system combines **traditional rule-based detection** with 
**explainable gradient boosting models** — the same hybrid approach used by 
leading FinTech fraud detection platforms. Every model decision is fully 
explainable at the individual transaction level using SHAP values, meeting 
the transparency requirements of AML compliance frameworks.

**Key results on IEEE-CIS dataset (590,540 real transactions):**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.938 |
| PR-AUC | 0.629 (18× above random baseline) |
| Fraud caught @ threshold 0.63 | 75.1% |
| False positive rate @ threshold 0.63 | 5.1% |
| Training set size | 472,432 transactions |
| Test set size | 118,108 transactions |

---

## 🏗️ Architecture
```
Raw CSV data (590k transactions)
        │
        ▼
┌───────────────────┐
│  Phase 1: EDA     │  Class imbalance analysis · missing value strategy
│  01_eda.ipynb     │  time patterns · amount distributions
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Phase 2:         │  Velocity features · amount z-scores
│  Feature Eng.     │  email domain risk · card history
│  02_features.ipynb│  missing value indicators · graph signals
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Phase 3:         │  Logistic Regression (PR-AUC 0.21)
│  Baseline Models  │  Random Forest      (PR-AUC 0.60)
│  03_baseline.ipynb│  XGBoost ★          (PR-AUC 0.63)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Phase 4:         │  SHAP TreeExplainer · waterfall plots
│  XAI + Rules      │  Hybrid rule engine · threshold tuning
│  04_advanced.ipynb│  Per-transaction explanations
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Phase 5:         │  Live transaction scoring
│  Streamlit App    │  Real-time SHAP explanations
│  src/app.py       │  Interactive threshold tuning
└───────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Pradyumna047/Fraud_detection.git
cd Fraud_detection
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection):
- `train_transaction.csv` → `data/raw/`
- `train_identity.csv` → `data/raw/`

### 5. Run the notebooks in order
```
notebooks/01_eda.ipynb
notebooks/02_features.ipynb
notebooks/03_baseline_models.ipynb
notebooks/04_advanced_models.ipynb
```

### 6. Launch the demo app
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501`

---

## 📁 Project Structure
```
Fraud_detection/
│
├── data/
│   ├── raw/                    ← Original Kaggle CSVs (not committed)
│   └── processed/              ← Engineered parquet files
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory data analysis
│   ├── 02_features.ipynb       ← Feature engineering pipeline
│   ├── 03_baseline_models.ipynb← Logistic regression, RF, XGBoost
│   └── 04_advanced_models.ipynb← SHAP explainability + rule engine
│
├── models/                     ← Saved model artifacts
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   └── shap_explainer_xgb.pkl
│
├── src/
│   └── app.py                  ← Streamlit demo application
│
├── reports/                    ← Generated plots and visualizations
│   ├── class_distribution.png
│   ├── shap_summary_xgb.png
│   ├── shap_waterfall_xgb.png
│   └── model_curves.png
│
├── requirements.txt
└── README.md
```

---

## 🔧 Feature Engineering

40+ domain-specific features engineered from raw transaction data:

**Time-based signals**
- `hour` / `is_night` — fraud spikes 2–4am when cardholders are asleep
- `is_weekend` — behavioural pattern shift on weekends
- `day_of_week` — weekly fraud cycle detection

**Amount anomaly detection**
- `log_amount` — handles right-skewed distributions
- `amt_zscore_product` — standard deviations from product-type baseline
- `is_round_amount` — fraudsters test cards with round numbers ($100, $500)

**Velocity features** *(core AML signal)*
- `card1_day_count` — transactions per card in rolling window
- `card1_time_since_last` — hours since previous transaction on this card
- `card1_rapid_fire` — flag: < 10 min between transactions

**Identity risk signals**
- `R_email_highrisk` — recipient using disposable/free webmail
- `P_email_highrisk` — purchaser email domain risk
- `email_domain_match` — purchaser/recipient domain mismatch
- `missing_count` — number of missing key identity fields

**Card history**
- `is_new_card` — first-ever use of this card
- `card1_total_txns` — lifetime transaction count
- `card_type_fraud_rate` — historical fraud rate by card network

---

## 🤖 Model Selection

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| Logistic Regression | 0.830 | 0.214 | Linear — misses complex patterns |
| Random Forest | 0.920 | 0.597 | Strong but slow at inference |
| **XGBoost ★** | **0.938** | **0.629** | Champion — best PR-AUC |

**Why PR-AUC over ROC-AUC:**  
With 96.5% of transactions being legitimate, a naive model predicting 
"not fraud" achieves 96.5% accuracy. ROC-AUC is misleadingly optimistic 
on imbalanced data. PR-AUC directly measures the precision/recall tradeoff 
— the metric that actually reflects the cost of false positives on 
fraud analyst workload.

**Class imbalance handling:**  
- `scale_pos_weight` in XGBoost weights fraud cases by 27× 
- Stratified train/test split preserves 3.5% fraud rate in both sets
- Threshold tuning to meet business constraints (catch rate vs FPR)

---

## 🔎 Explainability (SHAP)

Every model decision is explained at the individual transaction level 
using SHAP (SHapley Additive exPlanations).

**Example — why was this transaction flagged?**
```
Base fraud rate:  3.5%  (E[f(x)] = 0.035)

C14 = 0           +0.67  ← No prior transaction history on card
C13 = 0           +0.56  ← No historical footprint
R_email_highrisk  +0.53  ← Recipient using high-risk email domain
C11 = 3           +0.20  ← Unusual count pattern
missing_count     -0.30  ← Fewer missing fields (reduces suspicion)

Final fraud score: 0.9066  →  FLAGGED  ✓
```

This transparency is a legal requirement in AML compliance — 
investigators must be able to justify every flagged transaction 
to regulators.

---

## ⚙️ Hybrid Rule Engine

Hard rules layered on top of the ML model — mirroring production 
AML architectures used by platforms like Hawk AI:
```python
Rules (fire first — deterministic):
  ├── Amount > $5,000                    → FLAG
  ├── Rapid-fire (< 10min since last)    → FLAG  
  ├── New card + amount > $500           → FLAG
  └── Night transaction + amount > $1000 → FLAG

ML model (handles ambiguous cases):
  └── XGBoost score ≥ threshold          → FLAG
```

Benefits of the hybrid approach:
- Rules catch obvious fraud instantly with zero latency
- ML handles subtle, evolving fraud patterns rules can't encode
- Investigators see both the rule trigger AND the ML score
- Threshold can be tuned per business risk appetite

---

## 📊 Business Impact

At the optimal threshold of 0.63:

| Metric | Value | Business meaning |
|--------|-------|-----------------|
| Fraud caught | 75.1% | 3 in 4 fraud cases detected |
| False positive rate | 5.1% | 5 legitimate customers flagged per 100 |
| Precision | 34.7% | 1 in 3 alerts is real fraud |
| Alerts generated | ~7,400 | Manageable analyst workload |

**Threshold flexibility:**  
The interactive demo shows how adjusting the threshold shifts 
the operating point — from aggressive (91% fraud caught, 22% FPR) 
to conservative (56% fraud caught, 1.2% FPR). Real AML teams 
set this based on their analyst capacity and regulatory requirements.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn |
| Class imbalance | imbalanced-learn (SMOTE) |
| Graph features | NetworkX |
| Experiment tracking | MLflow |
| Demo app | Streamlit |
| Environment | Python 3.13, virtualenv |
| Version control | Git, GitHub |

---

## 📖 Key Learnings

**1. Metric choice is everything**  
Accuracy is useless for fraud detection. PR-AUC is the right metric 
because it directly captures the false positive cost that drives 
AML analyst workload.

**2. Feature engineering beats model complexity**  
The biggest performance jump came from engineered velocity and 
behavioural features — not from switching algorithms. A well-engineered 
feature set with logistic regression often beats a raw-feature neural network.

**3. Explainability is not optional in AML**  
SHAP isn't a nice-to-have — it's a compliance requirement. Regulators 
require financial institutions to justify adverse actions. A black-box 
model with better AUC is less useful than an explainable model with 
slightly lower AUC.

**4. Threshold tuning is a business decision, not a technical one**  
The optimal threshold depends on analyst headcount, regulatory environment, 
and the cost of missed fraud vs customer friction. The model provides 
the full tradeoff curve — the business decides where to operate on it.

---

## 👤 Author

**Pradyumna Shetty**  
M.Eng. Information & Electrical Engineering · Hochschule Wismar  
[LinkedIn](https://www.linkedin.com/in/pradyumna-shetty-534a14132/) · 
[GitHub](https://github.com/Pradyumna047)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built as a portfolio project demonstrating production-grade AML 
fraud detection — feature engineering, model evaluation, explainable AI, 
and interactive deployment.*
