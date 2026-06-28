#Streamlit_link : https://fraud-detection-paysphere-eglb9kyakn5vebfknukkks.streamlit.app/#5-actions

PaySphere Fraud Detection in Online Transactions
End‑to‑end fraud detection project for PaySphere Digital Payments Pvt. Ltd., focused on identifying suspicious online transactions, handling severe class imbalance, tuning fraud thresholds, and mapping scores to business actions.

1. Project Objective
Design a machine‑learning driven fraud risk engine that can:

Detect fraudulent transactions in near real time

Reduce financial loss from chargebacks

Control false positives so genuine customers are not over‑blocked

Provide business‑ready score bands for actions: Approve, Soft Review, OTP Challenge, Device Lock, Hard Block

All work is aligned to the Data Science with GenAI brief “Fraud Detection in Online Transactions”.2. Business Problem
PaySphere processes high‑volume payments across UPI, cards, net banking, and wallets for ecommerce, travel, gaming, and utility platforms. Over the last quarter they observed:

38% rise in attempted fraudulent transactions

Fraud share < 0.5% of total volume (severe class imbalance)

Increase in false negatives (missed fraud → chargebacks)

Increase in false positives (genuine customers blocked, revenue loss)

This makes fraud detection a rare‑event classification problem where simple accuracy is misleading and operational cost of errors is very high.

3. Data Overview
The dataset combines transaction, customer, device, merchant, temporal, and risk‑score features.

Key columns (simplified):

transactionid, customerid, deviceid, merchantid

timestamp, amount, paymentmethod, isinternational, merchantcategory

ipaddressriskscore, devicetrustscore, locationriskscore

velocity1h, velocity24h, velocity7d

customertenuredays, historicalfraudrate, merchanthistoricalfraudrate

ipaddresscountrymatch, previouschargebackcount

timeofday, dayofweek, isweekend, transactionsuccessratecustomer

isfraud (target label)

These fields allow both behavioural modelling (velocity, tenure, history) and risk‑based modelling (IP/device/merchant scores).

4. Project Structure
   fraud-detection-paysphere/
├── app.py                  # Streamlit UI
├── train.py                # Model training pipeline
├── inference.py            # Loading model + prediction helpers
├── preprocess.py           # Cleaning and encoding
├── feature_engineering.py  # Behavioural & risk feature creation
├── thresholding.py         # Score → action mapping
├── requirements.txt
├── packages.txt
├── README.md
├── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│   └── threshold_config.json
├── notebooks/
│   └── fraud_analysis.ipynb
├── assets/
│   └── styles.css
└── reports/
    └── business_report.pdf
   5. Methodology
5.1 Data Validation
Before modelling:

Check primary/foreign key consistency and uniqueness of transactionid

Validate ranges for risk scores (0–1) and temporal fields (timeofday 0–23, dayofweek 0–6)

Handle null / incorrect values and logical rule violations (e.g., negative amounts, impossible tenure)

Ensure chronological order where required (for velocity and history features).

5.2 Exploratory Data Analysis (EDA)
Core questions:

What is the base fraud rate and how imbalanced is the dataset?

How does fraud vary by amount, payment method, merchant category, international flag?

How do velocity fields (velocity1h, velocity24h, velocity7d) behave for fraud vs genuine customers?

Which ranges of device trust, IP risk, location risk, historical fraud rate show spikes in fraud?

Because fraud is rare, EDA focuses on precision‑recall patterns, not just overall distributions.

5.3 Feature Engineering
Guided by the project’s feature‑engineering suggestions:

Behavioural & amount features

Transaction velocity, recent count patterns

Average transaction amount per customer

Amount deviation score (current vs usual behaviour)

Device & identity consistency

Device familiarity score

Account–device and account–IP matching indicators

IP geolocation distance / country mismatch

Merchant & category risk

Merchant risk score and historical fraud rate

Merchant consistency for the customer

Time‑based indicators

Time‑of‑day risk, day‑of‑week risk, weekend flag

Network & fraud history

IP risk score, historical fraud rate for customer/device/merchant

Composite signals

Behavioural anomaly score

Combined risk index (weighted blend of key risk dimensions)

5.4 Modelling & Imbalance Handling
Split data into train/validation/test

Use class‑imbalance techniques such as:

Class weights in algorithms

Oversampling (e.g., SMOTE variants) on the minority class

Train baseline Logistic Regression and tree‑based models (Random Forest / Gradient Boosted Trees) for better fraud ranking.

Main evaluation metrics:

Recall / True Positive Rate (fraud capture rate)

Precision (false‑positive control)

F1 score, PR‑AUC, ROC‑AUC

Confusion matrix analysis at different thresholds

5.5 Threshold Tuning & Actions
Generate predicted fraud probabilities and choose score bands rather than a single cut‑off. Example mapping:

def map_action(score: float) -> str:
    if score >= 0.90:
        return "Hard Block"
    elif score >= 0.75:
        return "Device Lock + Manual Review"
    elif score >= 0.50:
        return "OTP Challenge"
    elif score >= 0.30:
        return "Soft Review"
    return "Approve"

    These bands align with the brief’s requirement for operational actions (hard block, soft review, OTP, device lock) instead of only a flag.

6. Streamlit App
The Streamlit UI serves as a Fraud Control Center with:

Overview — high‑level KPIs and explanation of the risk engine

Single Transaction Scoring — form to enter transaction details, output probability + action + key risk drivers

Batch Scoring — CSV upload, bulk scoring and downloadable results

Model Metrics — confusion matrix, PR/ROC curves, feature importance

Decision Logic — documentation of threshold bands and business rationale

This matches the deliverable of a fully functioning interactive UI.

7. Installation & Usage
7.1 Local setup
bash
git clone https://github.com/<your-username>/fraud-detection-paysphere.git
cd fraud-detection-paysphere

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
7.2 Streamlit Cloud (optional)
Create a packages.txt for system libraries if required (e.g. Pillow):

text
zlib1g-dev
libjpeg-dev
Then:

Push repo to GitHub

On Streamlit Community Cloud, create a new app from this repo

Set app.py as entrypoint

Deploy and monitor build logs

8. Stakeholders
The solution supports multiple teams inside PaySphere:

Fraud Risk & Compliance — reduce fraud loss, ensure regulatory adherence

Payments Engineering — integrate scoring into the transaction gateway

Data Science & AI — maintain and improve models

Customer Experience — monitor false positives and customer friction

Finance & Chargeback — track fraud losses and recoveries
