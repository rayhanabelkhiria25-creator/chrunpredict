"""
=============================================================
  Telco Churn Prediction — Streamlit Dashboard
  Run with: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json, os

# ── page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {background:#1e1e2e;border-radius:10px;padding:16px;text-align:center;}
    .stMetric label {font-size:13px!important;}
    h1 {color:#7c9fd4;}
</style>
""", unsafe_allow_html=True)

# ── paths ────────────────────────────────────────────────
DATA_PATH    = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH   = "best_model.pkl"
METRICS_PATH = "metrics_summary.json"
PLOTS        = {
    "ROC Curves":           "01_roc_curves.png",
    "Confusion Matrices":   "02_confusion_matrices.png",
    "PR Curves":            "03_pr_curves.png",
    "Metrics Comparison":   "04_metrics_comparison.png",
    "Feature Importance":   "05_feature_importance.png",
    "Probability Dist.":    "06_prob_distribution.png",
}

# ── load artefacts ───────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NumAddOns"] = df[["OnlineSecurity","OnlineBackup","DeviceProtection",
                            "TechSupport","StreamingTV","StreamingMovies"]].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                                labels=["0-1yr","1-2yr","2-4yr","4-6yr"], include_lowest=True)
    df["HighValueAtRisk"] = (
        (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)) &
        (df["Contract"] == "Month-to-month")
    ).astype(int)
    return df

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

model   = load_model()
df      = load_data()
metrics = load_metrics()

# ══════════════════════════════════════════════════════════
#  SIDEBAR — navigation
# ══════════════════════════════════════════════════════════
st.sidebar.title("📡 Churn Predictor")
st.sidebar.caption("Telco Customer Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Exploratory Analysis",
    "🤖 Model Performance",
    "🔮 Predict a Customer",
])

# ══════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Telco Customer Churn Dashboard")
    st.markdown("Full ML pipeline: **Data Prep → Feature Engineering → SMOTE + Class Weights → Evaluation**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churned Customers", f"{df['Churn_bin'].sum():,}")
    c3.metric("Churn Rate", f"{df['Churn_bin'].mean()*100:.1f}%")
    c4.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")

    st.divider()
    st.subheader("Pipeline Architecture")
    st.markdown("""
    | Step | Description | Detail |
    |------|-------------|--------|
    | 1 | **Data Loading** | 7,043 rows, 21 columns |
    | 2 | **Cleaning** | TotalCharges coercion, 11 NaN imputed |
    | 3 | **Feature Engineering** | AvgMonthlySpend, TenureGroup, NumAddOns, HighValueAtRisk |
    | 4 | **Preprocessing** | StandardScaler (num) + OHE (cat) + SimpleImputer |
    | 5 | **Imbalance** | SMOTE oversampling + `class_weight='balanced'` / `scale_pos_weight` |
    | 6 | **Models** | Logistic Regression, Random Forest, XGBoost |
    | 7 | **Evaluation** | ROC-AUC, F1, Avg-Precision, MCC, 5-fold StratifiedKFold |
    """)

    st.divider()
    st.subheader("Model Leaderboard")
    lb = pd.DataFrame([
        {"Model": k, **{kk: vv for kk, vv in v.items() if kk != "Model"}}
        for k, v in metrics.items()
    ]).set_index("Model")
    st.dataframe(lb.style.highlight_max(axis=0, color="#2b4d7e"), use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Churn by Feature", "Correlations"])

    with tab1:
        col = st.selectbox("Select numeric feature", ["MonthlyCharges","TotalCharges","tenure","NumAddOns","AvgMonthlySpend"])
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, grp in df.groupby("Churn"):
            ax.hist(grp[col], bins=40, alpha=0.6, label=label)
        ax.set_xlabel(col); ax.set_ylabel("Count"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    with tab2:
        cat_feat = st.selectbox("Select categorical feature", [
            "Contract","InternetService","PaymentMethod","TenureGroup",
            "gender","SeniorCitizen","Partner","Dependents","PaperlessBilling"])
        churn_rate = df.groupby(cat_feat)["Churn_bin"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        churn_rate.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="white")
        ax.set_title(f"Churn Rate by {cat_feat}"); ax.set_ylabel("Churn Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30); ax.grid(axis="y", alpha=0.3)
        ax.axhline(df["Churn_bin"].mean(), color="red", linestyle="--", lw=1.5, label="Overall avg")
        ax.legend(); st.pyplot(fig)

    with tab3:
        num_feats = ["tenure","MonthlyCharges","TotalCharges","AvgMonthlySpend","NumAddOns","HighValueAtRisk","Churn_bin"]
        corr = df[num_feats].corr()
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)
        ax.set_title("Correlation Matrix"); st.pyplot(fig)

# ══════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    plot_name = st.selectbox("Select chart", list(PLOTS.keys()))
    img_path  = PLOTS[plot_name]
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f"Plot not found at {img_path}. Run pipeline.py first.")

    st.divider()
    st.subheader("Cross-Validation Summary")
    st.markdown("""
    All three models were evaluated with **5-fold Stratified K-Fold CV** on the training set.
    - **SMOTE** was applied *inside* each fold to prevent data leakage.
    - `class_weight='balanced'` / `scale_pos_weight` adds a second layer of imbalance correction.
    - Evaluation prioritises **ROC-AUC** and **Average Precision** (better for imbalanced data than plain accuracy).
    """)
    lb = pd.DataFrame([
        {"Model": k, **{kk: vv for kk, vv in v.items() if kk != "Model"}}
        for k, v in metrics.items()
    ]).set_index("Model")
    st.dataframe(lb.style.highlight_max(axis=0, color="#2b4d7e"), use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE 4 — PREDICT A CUSTOMER
# ══════════════════════════════════════════════════════════
elif page == "🔮 Predict a Customer":
    st.title("🔮 Predict Churn for a Customer")
    st.markdown("Fill in the form below to get a real-time churn probability from the best model.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Demographics")
            gender          = st.selectbox("Gender",           ["Male","Female"])
            senior          = st.selectbox("Senior Citizen",   ["No","Yes"])
            partner         = st.selectbox("Partner",          ["Yes","No"])
            dependents      = st.selectbox("Dependents",       ["Yes","No"])

        with c2:
            st.subheader("Services")
            internet        = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            phone           = st.selectbox("Phone Service",    ["Yes","No"])
            multiple_lines  = st.selectbox("Multiple Lines",   ["Yes","No","No phone service"])
            online_sec      = st.selectbox("Online Security",  ["Yes","No","No internet service"])
            online_bk       = st.selectbox("Online Backup",    ["Yes","No","No internet service"])
            device_prot     = st.selectbox("Device Protection",["Yes","No","No internet service"])
            tech_sup        = st.selectbox("Tech Support",     ["Yes","No","No internet service"])
            streaming_tv    = st.selectbox("Streaming TV",     ["Yes","No","No internet service"])
            streaming_mv    = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

        with c3:
            st.subheader("Account")
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            contract        = st.selectbox("Contract",         ["Month-to-month","One year","Two year"])
            paperless       = st.selectbox("Paperless Billing",["Yes","No"])
            payment         = st.selectbox("Payment Method",   [
                                "Electronic check","Mailed check",
                                "Bank transfer (automatic)","Credit card (automatic)"])
            monthly         = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total           = st.number_input("Total Charges ($)",   0.0, 9000.0, float(monthly*tenure), step=1.0)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        addons = sum([online_sec=="Yes", online_bk=="Yes", device_prot=="Yes",
                      tech_sup=="Yes", streaming_tv=="Yes", streaming_mv=="Yes"])
        tenure_group = pd.cut([tenure], bins=[0,12,24,48,72],
                               labels=["0-1yr","1-2yr","2-4yr","4-6yr"],
                               include_lowest=True)[0]
        high_value_risk = int(monthly > df["MonthlyCharges"].quantile(0.75) and contract=="Month-to-month")

        row = pd.DataFrame([{
            "gender":          gender,
            "SeniorCitizen":   1 if senior=="Yes" else 0,
            "Partner":         partner,
            "Dependents":      dependents,
            "tenure":          tenure,
            "PhoneService":    phone,
            "MultipleLines":   multiple_lines,
            "InternetService": internet,
            "OnlineSecurity":  online_sec,
            "OnlineBackup":    online_bk,
            "DeviceProtection":device_prot,
            "TechSupport":     tech_sup,
            "StreamingTV":     streaming_tv,
            "StreamingMovies": streaming_mv,
            "Contract":        contract,
            "PaperlessBilling":paperless,
            "PaymentMethod":   payment,
            "MonthlyCharges":  monthly,
            "TotalCharges":    total,
            "AvgMonthlySpend": total / (tenure + 1),
            "TenureGroup":     tenure_group,
            "NumAddOns":       addons,
            "HighValueAtRisk": high_value_risk,
        }])

        prob = model.predict_proba(row)[0][1]
        pred = model.predict(row)[0]

        st.divider()
        col_a, col_b = st.columns([1, 2])

        with col_a:
            color = "#e74c3c" if prob > 0.5 else "#2ecc71"
            st.markdown(f"""
            <div style='background:{color}22;border:2px solid {color};border-radius:12px;padding:20px;text-align:center'>
                <h2 style='color:{color};margin:0'>{'⚠️ LIKELY CHURN' if pred else '✅ LIKELY STAYS'}</h2>
                <h1 style='color:{color};margin:8px 0'>{prob*100:.1f}%</h1>
                <p style='color:#aaa;margin:0'>Churn Probability</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            # gauge-style bar
            fig, ax = plt.subplots(figsize=(7, 2))
            ax.barh(["Churn Probability"], [prob], color="#e74c3c" if prob>0.5 else "#2ecc71",
                    height=0.5, edgecolor="white")
            ax.barh(["Churn Probability"], [1-prob], left=[prob], color="#333",
                    height=0.5, edgecolor="white")
            ax.axvline(0.5, color="orange", linestyle="--", lw=2, label="Threshold 0.5")
            ax.set_xlim(0, 1); ax.set_xlabel("Probability")
            ax.legend(); ax.grid(axis="x", alpha=0.3)
            ax.set_facecolor("#111"); fig.patch.set_facecolor("#111")
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
            st.pyplot(fig)

            st.markdown("**Key risk factors entered:**")
            st.write(f"- Contract: `{contract}` {'⚠️' if contract=='Month-to-month' else '✅'}")
            st.write(f"- Tenure: `{tenure} months` {'⚠️ New customer' if tenure<12 else '✅ Loyal'}")
            st.write(f"- Monthly Charges: `${monthly:.2f}` {'⚠️ High' if monthly>70 else ''}")
            st.write(f"- Add-On Services: `{addons}` subscribed")