"""
app.py — Sales Forecasting & Demand Intelligence Dashboard (Task 7)

Single-file Streamlit app. Run with:
    pip install -r requirements.txt
    streamlit run app.py

Pages (as tabs):
  1. Sales Overview Dashboard
  2. Forecast Explorer (Prophet — the best model from Task 3)
  3. Anomaly Report (Isolation Forest + Z-score, from Task 5)
  4. Product Demand Segments (K-Means, from Task 6)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📊", layout="wide")

REGIONS = ["West", "East", "Central", "South"]
CATEGORY_SUBCATS = {
    "Furniture": ["Bookcases", "Chairs", "Tables", "Furnishings"],
    "Office Supplies": ["Storage", "Binders", "Paper", "Art", "Labels"],
    "Technology": ["Phones", "Machines", "Accessories", "Copiers"],
}
CLUSTER_LABELS = {
    0: "High volume, stable demand",
    1: "Declining demand",
    2: "Low volume, high volatility",
    3: "Growing demand",
}


# ----------------------------------------------------------------------------
# Data loading & cleaning (Task 1)
# ----------------------------------------------------------------------------

def generate_sample_data(seed: int = 42, n_months: int = 48) -> pd.DataFrame:
    """Superstore-shaped synthetic data so the app has something to show
    if train.csv isn't uploaded yet."""
    rng = np.random.default_rng(seed)
    subcat_base = {}
    for cat, subs in CATEGORY_SUBCATS.items():
        for sc in subs:
            subcat_base[sc] = {"cat": cat, "base": 40 + rng.random() * 160}

    region_mult = {"West": 1.15, "East": 1.05, "Central": 0.85, "South": 0.9}
    start = pd.Timestamp("2015-01-01")
    rows = []
    for m in range(n_months):
        month_date = start + pd.DateOffset(months=m)
        month = month_date.month
        seasonal = 1 + 0.55 * np.sin((month - 10) / 12 * 2 * np.pi) + (0.5 if month >= 11 else 0)
        year_growth = 1 + (m / n_months) * 0.55
        n_orders = 55 + rng.integers(0, 35)
        for _ in range(n_orders):
            day = rng.integers(1, 28)
            order_date = month_date.replace(day=int(day))
            ship_days = int(rng.integers(1, 8))
            ship_date = order_date + pd.Timedelta(days=ship_days)
            subcat = rng.choice(list(subcat_base.keys()))
            meta = subcat_base[subcat]
            region = rng.choice(REGIONS)
            sales = max(
                5,
                meta["base"] * seasonal * year_growth * region_mult[region]
                * (0.4 + rng.random() * 1.3),
            )
            rows.append(
                {
                    "Order Date": order_date,
                    "Ship Date": ship_date,
                    "Category": meta["cat"],
                    "Sub-Category": subcat,
                    "Region": region,
                    "Sales": round(sales, 2),
                    "Postal Code": int(10000 + rng.integers(0, 89999)),
                }
            )
    return pd.DataFrame(rows)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Order Date"])

    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Week Number"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Order Day of Week"] = df["Order Date"].dt.dayofweek
    df["Order Quarter"] = df["Order Date"].dt.quarter

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    df["Order Season"] = df["Order Month"].apply(get_season)

    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].fillna(0).astype(int).astype(str)

    df = df.drop_duplicates()

    if "Ship Date" in df.columns:
        df["Shipping Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


@st.cache_data
def load_data(file_bytes, is_sample_flag):
    if is_sample_flag:
        return clean_data(generate_sample_data())
    raw = pd.read_csv(file_bytes)
    return clean_data(raw)


def monthly_sales(df: pd.DataFrame) -> pd.Series:
    s = df.set_index("Order Date").resample("ME")["Sales"].sum()
    s.index.freq = "ME"
    return s


def weekly_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    s = df.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
    s.columns = ["Order Date", "Weekly Sales"]
    return s


# ----------------------------------------------------------------------------
# Sidebar: data upload
# ----------------------------------------------------------------------------

st.title("📊 Sales Forecasting & Demand Intelligence Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload train.csv (Order Date, Ship Date, Sales, Category, Sub-Category, Region, Postal Code)",
    type=["csv"],
)
is_sample = uploaded_file is None
df = load_data(uploaded_file, is_sample)

if is_sample:
    st.sidebar.info("Showing synthetic sample data. Upload train.csv to use your real data.")
else:
    st.sidebar.success(f"Loaded {len(df):,} rows.")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Date range", f"{df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
c3.metric("Total revenue", f"${df['Sales'].sum():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Sales Overview Dashboard",
    "🔮 Forecast Explorer",
    "🚨 Anomaly Report",
    "📦 Product Demand Segments",
])


# ----------------------------------------------------------------------------
# PAGE 1 — Sales Overview Dashboard
# ----------------------------------------------------------------------------
with tab1:
    st.header("Sales Overview Dashboard")

    fc1, fc2 = st.columns(2)
    regions_sel = fc1.multiselect(
        "Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()), key="ov_region"
    )
    categories_sel = fc2.multiselect(
        "Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()), key="ov_category"
    )
    filtered = df[df["Region"].isin(regions_sel) & df["Category"].isin(categories_sel)]

    st.subheader("Total sales by year")
    yearly = filtered.groupby("Order Year")["Sales"].sum().reset_index()
    fig_year = px.bar(yearly, x="Order Year", y="Sales", text_auto=".2s")
    fig_year.update_layout(yaxis_tickprefix="$", xaxis=dict(type="category"))
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("Monthly sales trend")
    monthly = monthly_sales(filtered).reset_index()
    monthly.columns = ["Order Date", "Sales"]
    fig_trend = px.line(monthly, x="Order Date", y="Sales", markers=True)
    fig_trend.update_layout(yaxis_tickprefix="$", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Sales by region and category")
    g1, g2 = st.columns(2)
    region_sales = filtered.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
    fig_region = px.bar(region_sales, x="Sales", y="Region", orientation="h", color_discrete_sequence=["#378ADD"])
    fig_region.update_layout(xaxis_tickprefix="$")
    g1.plotly_chart(fig_region, use_container_width=True)

    category_sales = filtered.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
    fig_category = px.bar(category_sales, x="Sales", y="Category", orientation="h", color_discrete_sequence=["#0f6e56"])
    fig_category.update_layout(xaxis_tickprefix="$")
    g2.plotly_chart(fig_category, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# ----------------------------------------------------------------------------
with tab2:
    st.header("Forecast Explorer")
    st.caption("Uses Prophet — the model recommended for production in Task 3 (lowest MAE and MAPE).")

    fc1, fc2, fc3 = st.columns(3)
    dimension = fc1.selectbox("Forecast by", ["Category", "Region"], key="fc_dimension")
    options = sorted(df[dimension].unique())
    value = fc2.selectbox(dimension, options, key="fc_value")
    horizon = fc3.select_slider("Forecast horizon (months)", options=[1, 2, 3], value=3, key="fc_horizon")

    def segment_monthly(data_df, col, val):
        segment = data_df[data_df[col] == val].copy()
        m = segment.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
        m.columns = ["ds", "y"]
        return m

    @st.cache_data(show_spinner=False)
    def fit_and_forecast(monthly_df, periods):
        model_full = Prophet(seasonality_mode="additive", yearly_seasonality=True, changepoint_prior_scale=0.05)
        model_full.fit(monthly_df)
        future = model_full.make_future_dataframe(periods=periods, freq="ME", include_history=False)
        forecast = model_full.predict(future)

        mae = rmse = np.nan
        if len(monthly_df) > periods + 6:
            train = monthly_df.iloc[:-periods]
            test = monthly_df.iloc[-periods:]
            model_holdout = Prophet(seasonality_mode="additive", yearly_seasonality=True, changepoint_prior_scale=0.05)
            model_holdout.fit(train)
            future_holdout = model_holdout.make_future_dataframe(periods=periods, freq="ME", include_history=False)
            pred_holdout = model_holdout.predict(future_holdout)
            mae = np.mean(np.abs(test["y"].values - pred_holdout["yhat"].values))
            rmse = np.sqrt(np.mean((test["y"].values - pred_holdout["yhat"].values) ** 2))
        return forecast, mae, rmse

    monthly_seg = segment_monthly(df, dimension, value)

    if len(monthly_seg) < 6:
        st.warning(f"Not enough monthly history for {value} to fit Prophet reliably.")
    else:
        with st.spinner("Fitting Prophet..."):
            forecast, mae, rmse = fit_and_forecast(monthly_seg, horizon)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_seg["ds"], y=monthly_seg["y"], name="Actual", line=dict(color="#0f6e56")))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#378ADD", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
            y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(55,138,221,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Confidence interval",
        ))
        fig.update_layout(title=f"{value} — {horizon}-month forecast", yaxis_tickprefix="$", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Model performance (holdout of last {horizon} months)")
        p1, p2 = st.columns(2)
        p1.metric("MAE", f"${mae:,.0f}" if mae == mae else "n/a — not enough history")
        p2.metric("RMSE", f"${rmse:,.0f}" if rmse == rmse else "n/a — not enough history")

        st.subheader("Forecasted values")
        table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        table.columns = ["Month", "Forecast", "Low", "High"]
        table["Month"] = table["Month"].dt.strftime("%b %Y")
        st.dataframe(table.round(0), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# ----------------------------------------------------------------------------
with tab3:
    st.header("Anomaly Report")
    st.caption("Isolation Forest (global outlier detection) and rolling Z-score (local deviation), from Task 5.")

    a1, a2, a3 = st.columns(3)
    contamination = a1.slider("Isolation Forest contamination", 0.01, 0.10, 0.05, step=0.01, key="an_contam")
    window_size = a2.slider("Z-score rolling window (weeks)", 2, 12, 4, key="an_window")
    z_threshold = a3.slider("Z-score threshold", 1.5, 4.0, 2.0, step=0.1, key="an_zthresh")

    weekly = weekly_sales_df(df)

    X_if = weekly[["Weekly Sales"]].copy()
    model_if = IsolationForest(random_state=42, contamination=contamination)
    weekly["isolation_forest_anomaly"] = model_if.fit_predict(X_if)
    if_anomalies = weekly[weekly["isolation_forest_anomaly"] == -1]

    weekly["rolling_mean"] = weekly["Weekly Sales"].rolling(window=window_size, center=True).mean()
    weekly["rolling_std"] = weekly["Weekly Sales"].rolling(window=window_size, center=True).std()
    weekly["z_score"] = (weekly["Weekly Sales"] - weekly["rolling_mean"]) / weekly["rolling_std"]
    weekly["z_score_anomaly"] = np.where(weekly["z_score"].abs() > z_threshold, -1, 1)
    z_anomalies = weekly[weekly["z_score_anomaly"] == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Order Date"], y=weekly["Weekly Sales"], name="Weekly sales", line=dict(color="#565c66", width=1.5)))
    fig.add_trace(go.Scatter(
        x=if_anomalies["Order Date"], y=if_anomalies["Weekly Sales"],
        mode="markers", marker=dict(color="#a32d2d", size=11, symbol="circle"), name="Isolation Forest anomaly",
    ))
    fig.add_trace(go.Scatter(
        x=z_anomalies["Order Date"], y=z_anomalies["Weekly Sales"],
        mode="markers", marker=dict(color="#7f5cd6", size=11, symbol="x"), name="Z-score anomaly",
    ))
    fig.update_layout(yaxis_tickprefix="$", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if_dates = set(if_anomalies["Order Date"].dt.date)
    z_dates = set(z_anomalies["Order Date"].dt.date)
    common = if_dates & z_dates

    m1, m2, m3 = st.columns(3)
    m1.metric("Isolation Forest anomalies", len(if_dates))
    m2.metric("Z-score anomalies", len(z_dates))
    m3.metric("Flagged by both", len(common))

    st.subheader("Detected anomaly weeks")
    combined = weekly[(weekly["isolation_forest_anomaly"] == -1) | (weekly["z_score_anomaly"] == -1)].copy()
    combined["Detected by"] = combined.apply(
        lambda r: "Both" if (r["isolation_forest_anomaly"] == -1 and r["z_score_anomaly"] == -1)
        else ("Isolation Forest" if r["isolation_forest_anomaly"] == -1 else "Z-score"),
        axis=1,
    )
    display_table = combined[["Order Date", "Weekly Sales", "Detected by"]].sort_values("Order Date")
    display_table["Order Date"] = display_table["Order Date"].dt.date
    st.dataframe(display_table.round(0), use_container_width=True, hide_index=True)

    if len(combined) == 0:
        st.info("No anomalies flagged at the current thresholds — try lowering contamination or the Z-score threshold above.")


# ----------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# ----------------------------------------------------------------------------
with tab4:
    st.header("Product Demand Segments")
    st.caption("Same features as Task 6: total sales volume, average order value, sales volatility, YoY growth rate.")

    n_clusters = st.slider("Number of clusters (k)", 2, 6, 4, key="seg_k")

    @st.cache_data
    def perform_clustering(data_df, k):
        subcategory_summary = data_df.groupby("Sub-Category").agg(
            total_sales_volume=("Sales", "sum"),
            avg_order_value=("Sales", "mean"),
        ).reset_index()

        monthly_by_subcat = (
            data_df.set_index("Order Date")
            .groupby([pd.Grouper(freq="ME"), "Sub-Category"])["Sales"]
            .sum()
            .unstack(fill_value=0)
        )
        sales_volatility = monthly_by_subcat.std().reset_index(name="sales_volatility")

        yearly_by_subcat = data_df.groupby(["Order Year", "Sub-Category"])["Sales"].sum().unstack(fill_value=0)
        years = sorted(yearly_by_subcat.index)
        if len(years) >= 2:
            first_year, last_year = years[0], years[-1]
            first_sales = yearly_by_subcat.loc[first_year]
            last_sales = yearly_by_subcat.loc[last_year]
            growth = ((last_sales - first_sales) / (first_sales + 1e-6)).reset_index(name="sales_growth_rate")
        else:
            growth = pd.DataFrame({"Sub-Category": subcategory_summary["Sub-Category"], "sales_growth_rate": 0.0})

        seg = subcategory_summary.merge(sales_volatility, on="Sub-Category", how="left")
        seg = seg.merge(growth, on="Sub-Category", how="left")
        seg["sales_growth_rate"] = seg["sales_growth_rate"].fillna(0)
        seg["sales_volatility"] = seg["sales_volatility"].fillna(0)

        X = seg.set_index("Sub-Category")
        X_scaled = StandardScaler().fit_transform(X)

        k_eff = min(k, len(seg))
        kmeans = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
        seg["Cluster"] = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        seg["PC1"], seg["PC2"] = components[:, 0], components[:, 1]
        return seg, k_eff

    segmentation_df, k_eff = perform_clustering(df, n_clusters)

    st.subheader("Cluster chart")
    fig = px.scatter(
        segmentation_df, x="PC1", y="PC2", color=segmentation_df["Cluster"].astype(str),
        hover_data=["Sub-Category", "total_sales_volume", "avg_order_value"],
        labels={"color": "Cluster"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-categories by demand cluster")
    segmentation_df["Segment"] = segmentation_df["Cluster"].map(lambda c: CLUSTER_LABELS.get(c, f"Cluster {c}"))
    display_cols = ["Sub-Category", "Segment", "total_sales_volume", "avg_order_value", "sales_volatility", "sales_growth_rate"]
    st.dataframe(
        segmentation_df[display_cols].sort_values("Segment").round(2),
        use_container_width=True, hide_index=True,
    )
