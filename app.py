import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="PaySphere Fraud Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load saved artifacts

model = joblib.load("models/fraud_model.pkl")
threshold = joblib.load("models/fraud_threshold.pkl")


# Styling

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


# Helper functions

def map_action(score: float, threshold: float) -> str:
    if score >= 0.90:
        return "Hard Block"
    elif score >= 0.75:
        return "Device Lock + Manual Review"
    elif score >= 0.50:
        return "OTP Challenge"
    elif score >= threshold:
        return "Soft Review"
    else:
        return "Approve"

def risk_band(score: float, threshold: float) -> str:
    if score >= 0.75:
        return "High Risk"
    elif score >= threshold:
        return "Medium Risk"
    return "Low Risk"

def build_features(
    transaction_id, customer_id, device_id, merchant_id, amount,
    payment_method, is_international, merchant_category,
    ip_address_risk_score, device_trust_score, txn_count_last_24h,
    avg_amount_last_24h, merchant_diversity_last_7d,
    device_change_flag, location_change_flag, authentication_method,
    otp_success_rate_customer, past_fraud_count_customer,
    past_disputes_customer, merchant_historical_fraud_rate,
    hour_of_day, day_of_week
):
    is_weekend = 1 if day_of_week in [5, 6] else 0

    amount_deviation_ratio = amount / (avg_amount_last_24h + 1)
    amount_deviation_diff = amount - avg_amount_last_24h
    high_velocity_flag = int(txn_count_last_24h >= 5)
    late_night_flag = int(hour_of_day in [0, 1, 2, 3, 4, 23])
    risky_merchant_flag = int(merchant_category in ["Gaming", "Travel", "Electronics"])
    risky_payment_flag = int(payment_method in ["CARD", "WALLET"])
    low_device_trust_flag = int(device_trust_score < 0.3)
    high_ip_risk_flag = int(ip_address_risk_score > 0.7)
    low_otp_success_flag = int(otp_success_rate_customer < 0.5)
    change_risk_score = device_change_flag + location_change_flag
    customer_history_risk = (past_fraud_count_customer * 0.6) + (past_disputes_customer * 0.4)

    combined_risk_index = (
        0.25 * ip_address_risk_score +
        0.20 * (1 - device_trust_score) +
        0.15 * merchant_historical_fraud_rate +
        0.15 * amount_deviation_ratio / (amount_deviation_ratio + 1e-6) +
        0.10 * is_international +
        0.10 * change_risk_score +
        0.05 * high_velocity_flag
    )

    return pd.DataFrame([{
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
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "amount_deviation_ratio": amount_deviation_ratio,
        "amount_deviation_diff": amount_deviation_diff,
        "high_velocity_flag": high_velocity_flag,
        "late_night_flag": late_night_flag,
        "risky_merchant_flag": risky_merchant_flag,
        "risky_payment_flag": risky_payment_flag,
        "low_device_trust_flag": low_device_trust_flag,
        "high_ip_risk_flag": high_ip_risk_flag,
        "low_otp_success_flag": low_otp_success_flag,
        "change_risk_score": change_risk_score,
        "customer_history_risk": customer_history_risk,
        "combined_risk_index": combined_risk_index
    }])

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
    if row["amount_deviation_ratio"] > 2:
        flags.append("Transaction amount deviates from recent average")
    if row["is_international"] == 1:
        flags.append("International transaction")
    if row["hour_of_day"] in [0, 1, 2, 3, 4, 23]:
        flags.append("Late-night activity")
    return flags if flags else ["No major risk flags triggered"]

# Sidebar

st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown(f"**Fraud threshold:** `{threshold:.4f}`")
st.sidebar.info("This app scores transactions and maps them to operational fraud actions.")
st.sidebar.markdown("### Action policy")
st.sidebar.markdown("""
- Approve  
- Soft Review  
- OTP Challenge  
- Device Lock + Manual Review  
- Hard Block
""")


# Hero

st.markdown("""
<div class="hero">
    <h1 style="margin-bottom:0.2rem;">🛡️ PaySphere Fraud Risk Engine</h1>
    <p class="small-note">
        Real-time fraud scoring dashboard for online transactions with threshold-based actions,
        single-transaction simulation, and batch scoring.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Single Transaction", "Batch Scoring", "Model & Actions"
])

# Tab 1: Overview

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card">\
        <h4>Model Status</h4><h2>Active</h2><p class="small-note">Production-ready scoring flow</p></div>',
        unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card">\
        <h4>Decision Threshold</h4><h2>{threshold:.4f}</h2><p class="small-note">Tuned fraud cut-off</p></div>',
        unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card">\
        <h4>Primary Use</h4><h2>Fraud Detection</h2><p class="small-note">Transaction-level risk scoring</p></div>',
        unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi-card">\
        <h4>Decision Engine</h4><h2>5 Actions</h2><p class="small-note">Approve to hard block</p></div>',
        unsafe_allow_html=True)

    st.markdown("### Monitoring Notes")
    st.write(
        "Use the Single Transaction tab for live simulation and the Batch Scoring tab for CSV-based fraud screening."
    )


# Tab 2: Single transaction

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
        merchant_category = st.selectbox(
            "Merchant Category",
            ["Travel", "Electronics", "Fashion", "Utilities", "Gaming"]
        )
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
        input_df = build_features(
            transaction_id, customer_id, device_id, merchant_id, amount,
            payment_method, is_international, merchant_category,
            ip_address_risk_score, device_trust_score, txn_count_last_24h,
            avg_amount_last_24h, merchant_diversity_last_7d,
            device_change_flag, location_change_flag, authentication_method,
            otp_success_rate_customer, past_fraud_count_customer,
            past_disputes_customer, merchant_historical_fraud_rate,
            hour_of_day, day_of_week
        )

        fraud_score = model.predict_proba(input_df)[0, 1]
        prediction = int(fraud_score >= threshold)
        action = map_action(fraud_score, threshold)
        band = risk_band(fraud_score, threshold)
        flags = get_risk_flags(input_df.iloc[0])

        a, b, c = st.columns(3)
        a.metric("Fraud Score", f"{fraud_score:.4f}")
        b.metric("Prediction", "Fraud" if prediction == 1 else "Genuine")
        c.metric("Recommended Action", action)

        if band == "High Risk":
            st.markdown(f"<p class='risk-high'>Risk Band: {band}</p>", unsafe_allow_html=True)
        elif band == "Medium Risk":
            st.markdown(f"<p class='risk-med'>Risk Band: {band}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='risk-low'>Risk Band: {band}</p>", unsafe_allow_html=True)

        st.markdown("### Top Risk Flags")
        for flag in flags:
            st.write(f"- {flag}")

        st.markdown("### Scored Transaction Data")
        st.dataframe(input_df, use_container_width=True)

# Tab 3: Batch scoring

with tab3:
    st.subheader("Batch Transaction Scoring")
    uploaded_file = st.file_uploader("Upload CSV file for batch scoring", type=["csv"])

    st.caption("CSV should contain the raw input columns used by the model workflow.")

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
            batch_df["is_weekend"] = batch_df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
            batch_df["amount_deviation_ratio"] = batch_df["amount"] / (batch_df["avg_amount_last_24h"] + 1)
            batch_df["amount_deviation_diff"] = batch_df["amount"] - batch_df["avg_amount_last_24h"]
            batch_df["high_velocity_flag"] = (batch_df["txn_count_last_24h"] >= 5).astype(int)
            batch_df["late_night_flag"] = batch_df["hour_of_day"].isin([0, 1, 2, 3, 4, 23]).astype(int)
            batch_df["risky_merchant_flag"] = batch_df["merchant_category"].isin(
                ["Gaming", "Travel", "Electronics"]
            ).astype(int)
            batch_df["risky_payment_flag"] = batch_df["payment_method"].isin(["CARD", "WALLET"]).astype(int)
            batch_df["low_device_trust_flag"] = (batch_df["device_trust_score"] < 0.3).astype(int)
            batch_df["high_ip_risk_flag"] = (batch_df["ip_address_risk_score"] > 0.7).astype(int)
            batch_df["low_otp_success_flag"] = (batch_df["otp_success_rate_customer"] < 0.5).astype(int)
            batch_df["change_risk_score"] = batch_df["device_change_flag"] + batch_df["location_change_flag"]
            batch_df["customer_history_risk"] = (
                batch_df["past_fraud_count_customer"] * 0.6 +
                batch_df["past_disputes_customer"] * 0.4
            )
            batch_df["combined_risk_index"] = (
                0.25 * batch_df["ip_address_risk_score"] +
                0.20 * (1 - batch_df["device_trust_score"]) +
                0.15 * batch_df["merchant_historical_fraud_rate"] +
                0.15 * batch_df["amount_deviation_ratio"] / (batch_df["amount_deviation_ratio"] + 1e-6) +
                0.10 * batch_df["is_international"] +
                0.10 * batch_df["change_risk_score"] +
                0.05 * batch_df["high_velocity_flag"]
            )

            batch_scores = model.predict_proba(batch_df)[:, 1]
            batch_df["fraud_score"] = batch_scores
            batch_df["prediction"] = (batch_df["fraud_score"] >= threshold).astype(int)
            batch_df["recommended_action"] = batch_df["fraud_score"].apply(lambda x: map_action(x, threshold))

            st.success("Batch scoring completed.")
            st.dataframe(batch_df.head(20), use_container_width=True)

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Scored CSV",
                data=csv,
                file_name="scored_transactions.csv",
                mime="text/csv"
            )


# Tab 4: Model info

with tab4:
    st.subheader("Model & Action Logic")
    st.write("This fraud engine uses a trained classification pipeline and a tuned fraud threshold.")
    st.write(f"Current deployed fraud threshold: **{threshold:.4f}**")

    st.markdown("### Operational Action Mapping")
    st.markdown("""
- **Approve**: Low fraud probability  
- **Soft Review**: Slightly above tuned threshold  
- **OTP Challenge**: Medium suspicious behavior  
- **Device Lock + Manual Review**: Strong risk indicators  
- **Hard Block**: Very high fraud probability
""")

    st.markdown("### Risk Signals Used")
    st.markdown("""
- IP address risk score  
- Device trust score  
- Transaction velocity  
- Amount deviation vs recent history  
- Merchant fraud rate  
- Device/location change indicators  
- OTP success behavior  
- Time-of-day and category risk flags
""")