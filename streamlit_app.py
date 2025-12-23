import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "fraud_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

clf = model.named_steps["clf"] if hasattr(model, "named_steps") and "clf" in model.named_steps else model
if not hasattr(clf, "multi_class"):
    clf.multi_class = "auto"

st.set_page_config(page_title="Fraud Detector", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and the model will predict whether it looks fraudulent.")

st.subheader("Transaction details")

distance_from_home = st.number_input(
    "Distance from home (km)",
    min_value=0.0,
    value=5.0,
    help="How far from the cardholder's home this transaction took place."
)

distance_from_last_transaction = st.number_input(
    "Distance from last transaction (km)",
    min_value=0.0,
    value=2.0,
    help="How far from the previous transaction location."
)

ratio_to_median_purchase_price = st.number_input(
    "Purchase amount vs typical (ratio)",
    min_value=0.0,
    value=1.0,
    help="1.0 means typical. 2.0 means 2× higher than the user's median purchase."
)

st.markdown("### Payment & merchant info")

repeat_retailer = st.selectbox(
    "Repeat retailer?",
    ["No", "Yes"],
    help="Is this purchase from a retailer the user has bought from before?"
)

used_chip = st.selectbox(
    "Used chip?",
    ["No", "Yes"],
    help="Was the card chip used for the transaction?"
)

used_pin_number = st.selectbox(
    "PIN used?",
    ["No", "Yes"],
    help="Was a PIN used in the transaction?"
)

online_order = st.selectbox(
    "Online order?",
    ["No", "Yes"],
    help="Was this transaction made online?"
)

def yn_to_int(v: str) -> int:
    return 1 if v == "Yes" else 0

if st.button("Predict"):
    row = pd.DataFrame([{
        "distance_from_home": distance_from_home,
        "distance_from_last_transaction": distance_from_last_transaction,
        "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
        "repeat_retailer": yn_to_int(repeat_retailer),
        "used_chip": yn_to_int(used_chip),
        "used_pin_number": yn_to_int(used_pin_number),
        "online_order": yn_to_int(online_order),
    }])

    proba = float(model.predict_proba(row)[0][1])
    pred = int(model.predict(row)[0])

    st.metric("Fraud probability", f"{proba:.6f}")
    st.caption(f"({proba*100:.2f}%)")
    st.progress(proba)


    if proba < 0.30:
        risk = "Low"
    elif proba < 0.70:
        risk = "Medium"
    else:
        risk = "High"

    st.metric("Risk level", risk)

    if pred == 1:
        st.error("Prediction: Fraudulent 🚨")
    else:
        st.success("Prediction: Legit ✅")


    st.divider()
    st.subheader("How to interpret this")

    st.write(
        """
        This tool outputs a **fraud probability** (0 to 1) and a **prediction** (Fraudulent / Legit).

        **Fraud probability**
        - A value closer to **1.0** means the transaction looks **more similar to fraud patterns** seen in the training data.
        - A value closer to **0.0** means it looks **more similar to legitimate transactions**.

        **Risk level**
        - **Low (< 0.30):** Unlikely to be fraud based on the model.
        - **Medium (0.30–0.70):** Some warning signs; may need a manual review.
        - **High (≥ 0.70):** Strong warning signs; often worth flagging.

        **Important notes**
        - This is a **student portfolio demo**, not a real banking system.
        - The model can make mistakes:
          - **False positive:** Legit transaction flagged as fraud.
          - **False negative:** Fraud transaction predicted as legit.
        - The prediction is based only on the features you enter (distance, typical spend ratio, chip/PIN, online, repeat retailer).
        """
    )
