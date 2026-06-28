
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="PaySphere Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

model = joblib.load("models/fraud_model.pkl")
threshold = joblib.load("models/fraud_threshold.pkl")

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

st.title("🛡️ PaySphere Fraud Detection Dashboard")
st.caption("Real-time fraud scoring and business action engine")

col1, col2, col3 = st.columns(3)

with col1:
    transaction_id = st.number_input("Transaction ID", value=999999)
    customer_id = st.number_input("Customer ID", value=1001)
    device_id = st.number_input("Device ID", value=2001)
    merchant_id = st.number_input("Merchant ID", value=501)
    amount = st.number_input("Amount", value=5000.0)

with col2:
    payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
    is_international = st.selectbox("Is International", [0, 1])
    merchant_category = st.selectbox("Merchant Category", ["Travel", "Electronics", "Fashion", "Utilities", "Gaming"])
    authentication_method = st.selectbox("Authentication Method", ["OTP", "PIN", "NONE"])
    hour_of_day = st.slider("Hour of Day", 0, 23, 12)

with col3:
    ip_address_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.3)
    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.7)
    txn_count_last_24h = st.number_input("Txn Count Last 24h", value=2)
    avg_amount_last_24h = st.number_input("Avg Amount Last 24h", value=3000.0)
    merchant_diversity_last_7d = st.number_input("Merchant Diversity Last 7d", value=2)

device_change_flag = st.selectbox("Device Change Flag", [0, 1])
location_change_flag = st.selectbox("Location Change Flag", [0, 1])
otp_success_rate_customer = st.slider("OTP Success Rate Customer", 0.0, 1.0, 0.8)
past_fraud_count_customer = st.number_input("Past Fraud Count Customer", value=0)
past_disputes_customer = st.number_input("Past Disputes Customer", value=0)
merchant_historical_fraud_rate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, 0.1)
day_of_week = st.slider("Day of Week", 0, 6, 2)
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

if st.button("Score Transaction"):
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

    fraud_score = model.predict_proba(input_df)[0, 1]
    prediction = int(fraud_score >= threshold)
    action = map_action(fraud_score, threshold)

    st.metric("Fraud Score", f"{fraud_score:.4f}")
    st.metric("Prediction", "Fraud" if prediction == 1 else "Genuine")
    st.metric("Recommended Action", action)
    st.dataframe(input_df)
