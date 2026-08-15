"""
Sales Forecasting Dashboard
----------------------------
Interactive demo of a LightGBM-based sales forecasting system built for a
mid-size supermarket chain. Loads real trained models and a sample of the
held-out test period to show live forecasts, uncertainty ranges, and a
newsvendor-based recommended order quantity.

Run locally:   streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Load data and models (cached so this only runs once, not on every click)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("demo_data.csv", parse_dates=["date"])
    for col in ["item_id", "dept_id", "cat_id", "store_id"]:
        df[col] = df[col].astype("category")
    return df


@st.cache_resource
def load_models():
    final_model = joblib.load("model_final.pkl")
    model_low = joblib.load("model_low.pkl")
    model_high = joblib.load("model_high.pkl")
    model_safety_stock = joblib.load("model_safety_stock.pkl")
    return final_model, model_low, model_high, model_safety_stock


data = load_data()
final_model, model_low, model_high, model_safety_stock = load_models()

FEATURES = [
    "lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "sell_price", "price_change_pct", "is_weekend", "day_of_week_num",
    "month", "year", "snap_CA", "item_id", "dept_id", "cat_id", "store_id",
]

# ---------------------------------------------------------------------------
# Sidebar — selection controls
# ---------------------------------------------------------------------------

st.sidebar.title("Sales Forecasting")
st.sidebar.caption("Time series forecasting for a multi-store retail chain")

categories = sorted(data["cat_id"].unique())
selected_cat = st.sidebar.selectbox("Category", categories)

items_in_cat = sorted(data[data["cat_id"] == selected_cat]["item_id"].unique())
selected_item = st.sidebar.selectbox("Item", items_in_cat)

stores_for_item = sorted(data[data["item_id"] == selected_item]["store_id"].unique())
selected_store = st.sidebar.selectbox("Store", stores_for_item)

st.sidebar.markdown("---")
with st.sidebar.expander("About this project"):
    st.markdown(
        """
        Built for a mid-size supermarket chain's Inventory Manager, who
        currently orders stock based on rough year-over-year guesswork.

        This model forecasts daily demand per item, per store, and
        recommends an order quantity that accounts for the cost of a
        stockout versus the cost of excess stock — not just a single
        point prediction.

        **Data:** Kaggle M5 Forecasting (Walmart), California stores
        **Model:** LightGBM, tuned, 69.47% WAPE vs. 81.49% naive baseline
        """
    )

# ---------------------------------------------------------------------------
# Filter to the selected item + store, and generate predictions
# ---------------------------------------------------------------------------

subset = data[
    (data["item_id"] == selected_item) & (data["store_id"] == selected_store)
].sort_values("date").copy()

if subset.empty:
    st.warning("No data for this combination — try a different selection.")
    st.stop()

X = subset[FEATURES]
subset["pred"] = final_model.predict(X).clip(min=0)
subset["pred_low"] = model_low.predict(X).clip(min=0)
subset["pred_high"] = np.maximum(subset["pred_low"], model_high.predict(X).clip(min=0))
subset["recommended_order"] = model_safety_stock.predict(X).clip(min=0)

latest = subset.iloc[-1]
item_wape = (
    (subset["sales"] - subset["pred"]).abs().sum() / max(subset["sales"].sum(), 1) * 100
)

# ---------------------------------------------------------------------------
# Main panel — KPIs
# ---------------------------------------------------------------------------

st.title(f"{selected_item} — {selected_store}")
st.caption(f"Category: {selected_cat}  ·  Test period: {subset['date'].min().date()} to {subset['date'].max().date()}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest actual sales", f"{int(latest['sales'])} units")
col2.metric("Latest forecast", f"{latest['pred']:.1f} units")
col3.metric("Recommended order", f"{latest['recommended_order']:.1f} units",
            help="Newsvendor-based recommendation: assumes a stockout costs 3x more than excess stock")
col4.metric("This item's WAPE", f"{item_wape:.1f}%",
            help="Forecast error for this specific item over the shown period")

# ---------------------------------------------------------------------------
# Main panel — chart: actual vs forecast with uncertainty band
# ---------------------------------------------------------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=subset["date"], y=subset["pred_high"],
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=subset["date"], y=subset["pred_low"],
    fill="tonexty", fillcolor="rgba(99, 110, 250, 0.15)",
    line=dict(width=0), name="80% prediction interval", hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=subset["date"], y=subset["sales"],
    mode="lines+markers", name="Actual sales", line=dict(color="#2c3e50", width=2),
))
fig.add_trace(go.Scatter(
    x=subset["date"], y=subset["pred"],
    mode="lines+markers", name="Forecast", line=dict(color="#636efa", width=2, dash="dash"),
))

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Units sold",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Shaded band shows the model's 80% prediction interval (10th–90th percentile). "
    "This is a backtest: 'Forecast' shows what the model would have predicted for each "
    "day using only information available up to that point, compared against what actually happened."
)

# ---------------------------------------------------------------------------
# Recent data table
# ---------------------------------------------------------------------------

with st.expander("View underlying data"):
    display_cols = ["date", "sales", "pred", "pred_low", "pred_high",
                     "recommended_order", "sell_price"]
    st.dataframe(
        subset[display_cols].rename(columns={
            "date": "Date", "sales": "Actual", "pred": "Forecast",
            "pred_low": "Low (10th pct)", "pred_high": "High (90th pct)",
            "recommended_order": "Recommended Order", "sell_price": "Price",
        }).round(2),
        use_container_width=True, hide_index=True,
    )
