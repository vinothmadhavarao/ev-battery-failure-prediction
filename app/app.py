import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EV Battery Failure Dashboard", layout="wide")

st.title("EV Battery Failure Prediction Dashboard")
st.write("Explore the dataset and see what drives battery failure.")

# Load data
df = pd.read_csv("../data/ev_battery_failure_clean.csv")

# Sidebar filter
st.sidebar.header("Filters")
drive_type = st.sidebar.multiselect(
    "Drive Type", options=df["drive_type"].unique(), default=df["drive_type"].unique()
)
df_filtered = df[df["drive_type"].isin(drive_type)]

# Key stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Batteries", len(df_filtered))
col2.metric("Failed Batteries", int(df_filtered["battery_failure"].sum()))
failure_rate = df_filtered["battery_failure"].mean() * 100
col3.metric("Failure Rate", f"{failure_rate:.1f}%")

st.divider()

# Feature selector + boxplot
st.subheader("Feature vs Battery Failure")
numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "battery_failure"]
feature = st.selectbox("Choose a feature", numeric_cols, index=numeric_cols.index("thermal_runaway_risk") if "thermal_runaway_risk" in numeric_cols else 0)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_filtered, x="battery_failure", y=feature, ax=ax)
ax.set_xlabel("Battery Failure (0 = Healthy, 1 = Failed)")
st.pyplot(fig)

st.divider()

# Top correlations
st.subheader("Top Predictors of Battery Failure")
corr = df_filtered.corr(numeric_only=True)["battery_failure"].sort_values(ascending=False)
corr = corr.drop("battery_failure")
top10 = corr.head(10)
fig2, ax2 = plt.subplots(figsize=(8, 5))
top10.plot(kind="barh", ax=ax2, color="#0f3460")
ax2.invert_yaxis()
ax2.set_xlabel("Correlation with battery_failure")
st.pyplot(fig2)

st.caption("Data: EV Battery Health Prediction Dataset (20K), Kaggle — synthetic data")