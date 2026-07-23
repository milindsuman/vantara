import streamlit as st
import requests

API_URL = "https://vantara-api-sw3a.onrender.com"

st.set_page_config(page_title="Vantara", layout="centered")
st.title("Vantara")
st.caption("Customer segmentation, recommendations & demand forecasting")

# ---------------- Customer Analysis ----------------
st.subheader("Analyze Customer")
customer_id = st.text_input("Customer ID", value="13085")

if st.button("Analyze Customer"):
    if not customer_id:
        st.warning("Enter a customer ID.")
    else:
        with st.spinner("Fetching segment..."):
            seg_resp = requests.get(f"{API_URL}/segment/{customer_id}")
        if seg_resp.status_code == 200:
            seg = seg_resp.json()
            st.markdown("**Segment**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Recency", f"{seg['Recency']:.0f}")
            col2.metric("Frequency", f"{seg['Frequency']:.0f}")
            col3.metric("Monetary", f"{seg['Monetary']:.2f}")
            col4.metric("Segment", f"{seg['Segment']:.0f}")
        else:
            st.error(f"Segment: {seg_resp.json().get('detail', 'Not found')}")

        with st.spinner("Fetching recommendations..."):
            rec_resp = requests.get(f"{API_URL}/recommend/{customer_id}", params={"n": 5})
        if rec_resp.status_code == 200:
            recs = rec_resp.json()["recommendations"]
            st.markdown("**Recommended Products**")
            for r in recs:
                st.write(f"{r['description']}  ·  `{r['stock_code']}`  ·  score {r['score']:.3f}")
        else:
            st.error(f"Recommendations: {rec_resp.json().get('detail', 'Not found')}")

st.divider()

# ---------------- Similar Products ----------------
st.subheader("Find Similar Products")
stock_code = st.text_input("Stock Code", value="85123A", key="similar_input")

if st.button("Find Similar"):
    with st.spinner("Finding similar products..."):
        sim_resp = requests.get(f"{API_URL}/similar-products/{stock_code}", params={"n": 5})
    if sim_resp.status_code == 200:
        sims = sim_resp.json()["similar_products"]
        for code, score in sims:
            st.write(f"`{code}`  ·  similarity {score:.3f}")
    else:
        st.error(f"Similar products: {sim_resp.json().get('detail', 'Not found')}")

st.divider()

# ---------------- Demand Forecast ----------------
st.subheader("Demand Forecast")
forecast_code = st.text_input("Stock Code", value="85123A", key="forecast_input")

if st.button("Get Forecast"):
    with st.spinner("Running forecast..."):
        forecast_resp = requests.get(f"{API_URL}/forecast/{forecast_code}")
    if forecast_resp.status_code == 200:
        forecast = forecast_resp.json()
        st.metric(
            label=f"Predicted demand — {forecast['forecast_month']}",
            value=f"{forecast['predicted_quantity']:.0f} units"
        )
    else:
        st.error(f"Forecast: {forecast_resp.json().get('detail', 'Not found')}")