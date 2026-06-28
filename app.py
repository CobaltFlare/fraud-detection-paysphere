import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PaySphere Fraud Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

THRESHOLD = 0.55

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0b1220 0%, #111827 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(14,165,233,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.risk-low {
    color: #22c55e;
    font-weight: 700;
}
.risk-med {
    color: #f59e0b;
    font-weight: 700;
}
.risk-high {
    color: #ef4444;
    font-weight: 700;
}
.small-note {
    font-size: 0.9rem;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def map_action(score: float, threshold: float = THRESHOLD) -> str:
    if score >= 0.90:
        return "Hard Block"
    elif score >= 0.75:
        return "Device Lock + Manual Review"
    elif score >= 0.55:
        return "OTP Challenge"
    elif score >= threshold:
        return "Soft Review"
    else:
        return "Approve"

def risk_band(score: float, threshold: float = THRESHOLD) -> str:
    if score >= 0.75:
        return "High Risk"
    elif score >= threshold:
        return "Medium Risk"
    return "Low Risk"

def engineer_features(df):
    df = df.copy()
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
    df["amount_deviation_ratio"] = df["amount"] / (df["avg_amount_last_24h"] + 1)
    df["amount_deviation_diff"] = df["amount"] - df["avg_amount_last_24h"]
    df["high_velocity_flag"] = (df["txn_count_last_24h"] >= 5).astype(int)
    df["late_night_flag"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    df["risky_merchant_flag"] = df["merchant_category"].isin(["Gaming", "Travel", "Electronics"]).astype(int)
    df["risky_payment_flag"] = df["payment_method"].isin(["CARD", "WALLET"]).astype(int)
    df["low_device_trust_flag"] = (df["device_trust_score"] < 0.3).astype(int)
    df["high_ip_risk_flag"] = (df["ip_address_risk_score"] > 0.7).astype(int)
    df["low_otp_success_flag"] = (df["otp_success_rate_customer"] < 0.5).astype(int)
    df["change_risk_score"] = df["device_change_flag"] + df["location_change_flag"]
    df["customer_history_risk"] = (
        df["past_fraud_count_customer"] * 0.6 +
        df["past_disputes_customer"] * 0.4
    )
    return df

def score_rule_engine(df):
    df = engineer_features(df)

    raw_score = (
        1.8 * df["high_ip_risk_flag"] +
        1.6 * df["low_device_trust_flag"] +
        1.2 * df["high_velocity_flag"] +
        1.0 * df["late_night_flag"] +
        1.0 * df["risky_merchant_flag"] +
        0.8 * df["risky_payment_flag"] +
        1.1 * df["low_otp_success_flag"] +
        1.1 * df["change_risk_score"] +
        0.9 * (df["is_international"]) +
        0.8 * np.clip(df["merchant_historical_fraud_rate"] * 5, 0, 2) +
        0.7 * np.clip(df["amount_deviation_ratio"] - 1, 0, 3) +
        0.5 * np.clip(df["customer_history_risk"], 0, 3) -
        2.2
    )

    df["fraud_score"] = sigmoid(raw_score)
    df["prediction"] = (df["fraud_score"] >= THRESHOLD).astype(int)
    df["recommended_action"] = df["fraud_score"].apply(map_action)
    df["risk_band"] = df["fraud_score"].apply(risk_band)
    return df

def get_risk_flags(row):
    flags = []
    if row["ip_address_risk_score"] > 0.7:
        flags.append("High IP reputation risk")
    if row["device_trust_score"] < 0.3:
        flags.append("Low device trust")
    if row["txn_count_last_24h"] >= 5:
        flags.append("High recent transaction velocity")
    if row["location_change_flag"] == 1:
        flags.append("Location change detected")
    if row["device_change_flag"] == 1:
        flags.append("Device change detected")
    if row["otp_success_rate_customer"] < 0.5:
        flags.append("Low OTP success rate")
    if row["merchant_historical_fraud_rate"] > 0.1:
        flags.append("Merchant has elevated fraud history")
    if row["amount"] > (row["avg_amount_last_24h"] * 2):
        flags.append("Transaction amount deviates sharply from recent average")
    if row["is_international"] == 1:
        flags.append("International transaction")
    if row["hour_of_day"] in [0, 1, 2, 3, 4, 23]:
        flags.append("Late-night activity")
    return flags if flags else ["No major risk flags triggered"]

st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown(f"**Fraud threshold:** `{THRESHOLD:.2f}`")
st.sidebar.info("Rules-based fraud scoring for deployment-safe real-time decisioning.")
st.sidebar.markdown("### Action policy")
st.sidebar.markdown("""
- Approve  
- Soft Review  
- OTP Challenge  
- Device Lock + Manual Review  
- Hard Block
""")

st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:0.2rem;">🛡️ PaySphere Fraud Risk Engine</h1>
    <p class="small-note">
        Advanced fraud risk dashboard with single transaction scoring, batch scoring,
        risk flags, and operational actions.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Single Transaction", "Batch Scoring", "Decision Logic"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card"><h4>Engine Status</h4><h2>Active</h2><p class="small-note">Deployment-safe scoring engine</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><h4>Decision Threshold</h4><h2>{THRESHOLD:.2f}</h2><p class="small-note">Fraud cut-off</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><h4>Decision Layers</h4><h2>5 Actions</h2><p class="small-note">Approve to hard block</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi-card"><h4>Primary Focus</h4><h2>Fraud Risk</h2><p class="small-note">Operational screening</p></div>', unsafe_allow_html=True)

    st.markdown("### Monitoring Notes")
    st.write("Use the Single Transaction tab for analyst simulation and the Batch Scoring tab for CSV-level screening.")

with tab2:
    st.subheader("Single Transaction Scoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_id = st.number_input("Transaction ID", value=999999)
        customer_id = st.number_input("Customer ID", value=1001)
        device_id = st.number_input("Device ID", value=2001)
        merchant_id = st.number_input("Merchant ID", value=501)
        amount = st.number_input("Amount", value=5000.0, min_value=0.0)

    with col2:
        payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
        is_international = st.selectbox("Is International", [0, 1])
        merchant_category = st.selectbox("Merchant Category", ["Travel", "Electronics", "Fashion", "Utilities", "Gaming"])
        authentication_method = st.selectbox("Authentication Method", ["OTP", "PIN", "NONE"])
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)

    with col3:
        ip_address_risk_score = st.slider("IP Address Risk Score", 0.0, 1.0, 0.30)
        device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.70)
        txn_count_last_24h = st.number_input("Txn Count Last 24h", value=2, min_value=0)
        avg_amount_last_24h = st.number_input("Avg Amount Last 24h", value=3000.0, min_value=0.0)
        merchant_diversity_last_7d = st.number_input("Merchant Diversity Last 7d", value=2, min_value=0)

    c4, c5, c6 = st.columns(3)
    with c4:
        device_change_flag = st.selectbox("Device Change Flag", [0, 1])
        location_change_flag = st.selectbox("Location Change Flag", [0, 1])
    with c5:
        otp_success_rate_customer = st.slider("OTP Success Rate Customer", 0.0, 1.0, 0.80)
        past_fraud_count_customer = st.number_input("Past Fraud Count Customer", value=0, min_value=0)
    with c6:
        past_disputes_customer = st.number_input("Past Disputes Customer", value=0, min_value=0)
        merchant_historical_fraud_rate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, 0.10)
        day_of_week = st.slider("Day of Week", 0, 6, 2)

    if st.button("🚀 Score Transaction", use_container_width=True):
        input_df = pd.DataFrame([{
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "device_id": device_id,
            "merchant_id": merchant_id,
            "amount": amount,
            "payment_method": payment_method,
            "is_international": is_international,
            "merchant_category": merchant_category,
            "ip_address_risk_score": ip_address_risk_score,
            "device_trust_score": device_trust_score,
            "txn_count_last_24h": txn_count_last_24h,
            "avg_amount_last_24h": avg_amount_last_24h,
            "merchant_diversity_last_7d": merchant_diversity_last_7d,
            "device_change_flag": device_change_flag,
            "location_change_flag": location_change_flag,
            "authentication_method": authentication_method,
            "otp_success_rate_customer": otp_success_rate_customer,
            "past_fraud_count_customer": past_fraud_count_customer,
            "past_disputes_customer": past_disputes_customer,
            "merchant_historical_fraud_rate": merchant_historical_fraud_rate,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week
        }])

        scored = score_rule_engine(input_df)
        row = scored.iloc[0]
        flags = get_risk_flags(row)

        a, b, c = st.columns(3)
        a.metric("Fraud Score", f"{row['fraud_score']:.4f}")
        b.metric("Prediction", "Fraud" if row["prediction"] == 1 else "Genuine")
        c.metric("Recommended Action", row["recommended_action"])

        if row["risk_band"] == "High Risk":
            st.markdown(f"<p class='risk-high'>Risk Band: {row['risk_band']}</p>", unsafe_allow_html=True)
        elif row["risk_band"] == "Medium Risk":
            st.markdown(f"<p class='risk-med'>Risk Band: {row['risk_band']}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='risk-low'>Risk Band: {row['risk_band']}</p>", unsafe_allow_html=True)

        st.markdown("### Top Risk Flags")
        for flag in flags:
            st.write(f"- {flag}")

        st.markdown("### Scored Transaction Data")
        st.dataframe(scored, use_container_width=True)

with tab3:
    st.subheader("Batch Transaction Scoring")
    uploaded_file = st.file_uploader("Upload CSV file for batch scoring", type=["csv"])
    st.caption("CSV should contain the raw fraud input columns used by the scoring workflow.")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        required_cols = [
            "transaction_id","customer_id","device_id","merchant_id","amount",
            "payment_method","is_international","merchant_category",
            "ip_address_risk_score","device_trust_score","txn_count_last_24h",
            "avg_amount_last_24h","merchant_diversity_last_7d","device_change_flag",
            "location_change_flag","authentication_method","otp_success_rate_customer",
            "past_fraud_count_customer","past_disputes_customer",
            "merchant_historical_fraud_rate","hour_of_day","day_of_week"
        ]

        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            scored_batch = score_rule_engine(batch_df)
            st.success("Batch scoring completed.")
            st.dataframe(scored_batch.head(20), use_container_width=True)

            csv = scored_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Scored CSV",
                data=csv,
                file_name="scored_transactions.csv",
                mime="text/csv"
            )

with tab4:
    st.subheader("Decision Logic")
    st.write("This app uses a deployment-safe fraud rules engine with engineered behavioral and risk signals.")
    st.write(f"Current fraud threshold: **{THRESHOLD:.2f}**")

    st.markdown("### Operational Action Mapping")
    st.markdown("""
- **Approve**: Low risk  
- **Soft Review**: Slightly above threshold  
- **OTP Challenge**: Moderate suspicious behavior  
- **Device Lock + Manual Review**: Strong risk signals  
- **Hard Block**: Extremely high fraud probability
""")

    st.markdown("### Risk Signals Used")
    st.markdown("""
- IP address risk score  
- Device trust score  
- Transaction velocity  
- Amount deviation vs recent average  
- Merchant fraud history  
- Device and location change indicators  
- OTP behavior  
- International usage  
- Time-of-day activity  
- Merchant category risk
""")
