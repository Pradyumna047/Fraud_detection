import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    model     = joblib.load(os.path.join(ROOT, 'models', 'xgboost.pkl'))
    explainer = joblib.load(os.path.join(ROOT, 'models', 'shap_explainer_xgb.pkl'))
    return model, explainer

model, explainer = load_models()

@st.cache_data
def load_sample_data():
    df   = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', '04_champion_predictions.parquet'))
    slim = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', '02_features.parquet'))
    DROP = ['TransactionID', 'isFraud', 'TransactionDT']
    X = slim.drop(columns=DROP)
    for col in X.select_dtypes(exclude=['number']).columns:
        X[col] = pd.Categorical(X[col]).codes.astype('int16')
    X = X.fillna(X.median(numeric_only=True))
    return X, df

X_all, preds = load_sample_data()

st.title("🔍 Fraud Detection System")
st.markdown(
    "**AML Transaction Monitoring** — XGBoost + SHAP Explainability  \n"
    "Trained on IEEE-CIS dataset · 590,540 transactions · PR-AUC: 0.629"
)
st.divider()

st.sidebar.header("⚙️ Model Controls")

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.01, max_value=0.99,
    value=0.63, step=0.01,
    help="Lower = catch more fraud but more false positives"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Threshold impact")

y_prob = preds['prob_xgb'].values
y_true = preds['y_true'].values
y_pred = (y_prob >= threshold).astype(int)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

st.sidebar.metric("Fraud caught",       f"{tp/(tp+fn):.1%}")
st.sidebar.metric("False positive rate",f"{fp/(fp+tn):.1%}")
st.sidebar.metric("Precision",          f"{tp/(tp+fp):.1%}" if (tp+fp)>0 else "N/A")
st.sidebar.metric("Alerts generated",   f"{tp+fp:,}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Score a transaction")

    st.markdown("Adjust the transaction parameters and see the model's decision:")

    # Input widgets
    amount      = st.number_input("Transaction amount ($)",
                                   min_value=0.01, max_value=20000.0,
                                   value=250.0, step=10.0)
    hour        = st.slider("Hour of day", 0, 23, 14)
    is_night    = 1 if hour <= 5 else 0
    is_weekend  = st.selectbox("Day type", ["Weekday", "Weekend"]) == "Weekend"
    email_risk  = st.selectbox("Recipient email domain",
                                ["Low risk (corporate)", "High risk (gmail/yahoo)"])
    r_highrisk  = 1 if "High" in email_risk else 0
    new_card    = st.selectbox("Card history",
                                ["Existing card", "New card (first use)"]) == "New card (first use)"
    rapid_fire  = st.selectbox("Transaction pace",
                                ["Normal", "Rapid fire (<10min since last)"]) == "Rapid fire (<10min since last)"

    base_row = pd.DataFrame([X_all.median()], columns=X_all.columns)
    base_row['TransactionAmt']      = amount
    base_row['log_amount']          = np.log1p(amount)
    base_row['amt_zscore_product']  = (amount - 250) / 200
    base_row['hour']                = hour
    base_row['is_night']            = is_night
    base_row['is_weekend']          = int(is_weekend)
    base_row['R_email_highrisk']    = r_highrisk
    base_row['is_new_card']         = int(new_card)
    base_row['card1_rapid_fire']    = int(rapid_fire)
    base_row['is_round_amount']     = int(amount % 1 == 0)

    score    = model.predict_proba(base_row)[0, 1]
    flagged  = score >= threshold

    st.markdown("---")

    if flagged:
        st.error(f"🚨 **FLAGGED AS FRAUD**")
        st.metric("Fraud probability", f"{score:.1%}",
                  delta=f"{score - threshold:+.1%} above threshold")
    else:
        st.success(f"✅ **TRANSACTION APPROVED**")
        st.metric("Fraud probability", f"{score:.1%}",
                  delta=f"{score - threshold:.1%} below threshold")

    rules_fired = []
    if amount > 5000:
        rules_fired.append("⚠️ Amount > $5,000")
    if rapid_fire:
        rules_fired.append("⚠️ Rapid-fire transaction")
    if new_card and amount > 500:
        rules_fired.append("⚠️ New card + high amount")
    if is_night and amount > 1000:
        rules_fired.append("⚠️ Night transaction + high amount")

    if rules_fired:
        st.warning("**Hard rules triggered:**\n" +
                   "\n".join(rules_fired))

with col2:
    st.subheader("🔎 Why this decision? (SHAP)")

    shap_vals = explainer.shap_values(base_row)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    exp_val = (explainer.expected_value[1]
               if isinstance(explainer.expected_value, list)
               else explainer.expected_value)

    shap_exp = shap.Explanation(
        values        = sv[0],
        base_values   = exp_val,
        data          = base_row.iloc[0].values,
        feature_names = base_row.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap_exp, max_display=12, show=False)
    plt.title("Feature contributions to this decision",
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        "🔴 Red bars push toward fraud · "
        "🔵 Blue bars push away from fraud · "
        "Width = magnitude of impact"
    )

st.divider()
st.subheader("📈 Model performance overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",       "XGBoost")
c2.metric("ROC-AUC",     "0.938")
c3.metric("PR-AUC",      "0.629")
c4.metric("Training set","472,432 txns")

from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)

fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(recall, precision, color='#e74c3c', linewidth=2)
ax2.axhline(y=y_true.mean(), color='gray', linestyle='--',
            alpha=0.6, label=f'Baseline ({y_true.mean():.3f})')
ax2.fill_between(recall, precision, alpha=0.1, color='#e74c3c')
ax2.set_xlabel('Recall (fraud caught)')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall curve — test set', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.caption(
    "Built by Pradyumna Shetty · "
    "IEEE-CIS Fraud Detection dataset · "
    "XGBoost + SHAP explainability"
)