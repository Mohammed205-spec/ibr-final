import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st

# ------------- Page Config -------------
st.set_page_config(
    page_title="Tesla Actual vs Predicted — Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS polish for cards & captions
st.markdown("""
<style>
.small { font-size: 0.85rem; color: #A9B0BD; }
.caption { color:#A9B0BD; font-size:0.9rem; margin-top:-10px; }
.metric-card { padding: 1rem; border-radius: 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); }
.hr { border-top: 1px solid rgba(255,255,255,0.08); margin: 0.75rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ------------- Helpers -------------
NUMERIC_DTYPES = ["int16","int32","int64","float16","float32","float64"]

def to_datetime_safe(s: pd.Series):
    for fmt in [None, "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"]:
        try:
            if fmt:
                return pd.to_datetime(s, format=fmt, errors="raise")
            else:
                return pd.to_datetime(s, errors="raise", infer_datetime_format=True)
        except Exception:
            continue
    return None

def mape(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mask = yt != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0

def smape(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100.0

def compute_metrics(df, actual_col, pred_col, groupby=None):
    data = df.dropna(subset=[actual_col, pred_col]).copy()
    if data.empty:
        return pd.DataFrame([{"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE%": np.nan, "sMAPE%": np.nan, "n": 0}])

    if groupby and groupby in data.columns:
        rows = []
        for g, part in data.groupby(groupby):
            part = part.dropna(subset=[actual_col, pred_col])
            if part.empty:
                rows.append({groupby: g, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE%": np.nan, "sMAPE%": np.nan, "n": 0})
                continue
            yt = part[actual_col].values
            yp = part[pred_col].values
            rows.append({
                groupby: g,
                "R2": r2_score(yt, yp) if len(part) >= 2 else np.nan,
                "RMSE": np.sqrt(mean_squared_error(yt, yp)),
                "MAE": mean_absolute_error(yt, yp),
                "MAPE%": mape(yt, yp),
                "sMAPE%": smape(yt, yp),
                "n": len(part),
            })
        return pd.DataFrame(rows)
    else:
        yt = data[actual_col].values
        yp = data[pred_col].values
        return pd.DataFrame([{
            "R2": r2_score(yt, yp) if len(data) >= 2 else np.nan,
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "MAPE%": mape(yt, yp),
            "sMAPE%": smape(yt, yp),
            "n": len(data),
        }])

def add_residuals(df, actual_col, pred_col):
    out = df.copy()
    out["residual"] = out[pred_col] - out[actual_col]
    out["abs_error"] = np.abs(out["residual"])
    with np.errstate(divide='ignore', invalid='ignore'):
        out["ape%"] = np.where(out[actual_col] != 0, (out["abs_error"] / np.abs(out[actual_col])) * 100, np.nan)
    return out

def identify_columns(df):
    cols = list(df.columns)

    # Date-like
    date_candidates = [c for c in cols if "date" in c.lower() or "time" in c.lower() or c.lower() in ["ds","timestamp"]]
    date_col = date_candidates[0] if date_candidates else None

    # Numeric columns
    numeric_cols = [c for c in cols if str(df[c].dtype) in NUMERIC_DTYPES]

    # Guess actual/pred
    actual_candidates = [c for c in numeric_cols if "actual" in c.lower() or c.lower() in ["y","y_true","target"]]
    pred_candidates = [c for c in numeric_cols if "pred" in c.lower() or "forecast" in c.lower() or c.lower() in ["yhat","y_pred"]]

    # Optional
    model_col = None
    for c in cols:
        cl = c.lower()
        if cl in ["model","algo","algorithm","estimator"] or "model" in cl:
            model_col = c
            break

    ticker_col = None
    for c in cols:
        if c.lower() == "ticker":
            ticker_col = c
            break

    return {
        "date_col": date_col,
        "numeric_cols": numeric_cols,
        "actual_candidates": actual_candidates or (numeric_cols[:1] if numeric_cols else []),
        "pred_candidates": pred_candidates or (numeric_cols[1:2] if len(numeric_cols) > 1 else numeric_cols[:1]),
        "model_col": model_col,
        "ticker_col": ticker_col
    }

def line_overlay(df, x, y_cols, title=""):
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, r=10, l=10, b=10),
        height=420
    )
    return fig

def scatter_parity(df, actual_col, pred_col, title="Predicted vs Actual"):
    # base scatter
    fig = px.scatter(
        df, x=actual_col, y=pred_col,
        opacity=0.85, title=title,
        labels={actual_col:"Actual", pred_col:"Predicted"}
    )

    # Parity line (y = x)
    try:
        min_val = float(np.nanmin([df[actual_col].min(), df[pred_col].min()]))
        max_val = float(np.nanmax([df[actual_col].max(), df[pred_col].max()]))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Perfect fit", line=dict(dash="dash")
        ))
    except Exception:
        pass

    # OLS fit line via numpy (no statsmodels needed)
    clean = df[[actual_col, pred_col]].dropna()
    if len(clean) >= 2:
        x = clean[actual_col].values.astype(float)
        y = clean[pred_col].values.astype(float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines", name="OLS fit"
            ))
        except Exception:
            pass

    fig.update_layout(margin=dict(t=50, r=10, l=10, b=10), height=420)
    return fig

def residual_timeseries(df, x, residual_col="residual"):
    fig = px.line(df, x=x, y=residual_col, title="Residuals over Time")
    fig.add_hline(y=0, line_dash="dash", opacity=0.7)
    fig.update_layout(margin=dict(t=50, r=10, l=10, b=10), height=350)
    return fig

def residual_distribution(df, residual_col="residual"):
    fig = px.histogram(df, x=residual_col, nbins=40, marginal="box",
                       title="Residual Distribution", opacity=0.85)
    fig.update_layout(margin=dict(t=50, r=10, l=10, b=10), height=350)
    return fig

def calibration_curve(df, actual_col, pred_col, bins=10, title="Calibration (Binned Means)"):
    tmp = df[[actual_col, pred_col]].dropna().copy()
    if tmp.empty:
        return go.Figure()
    tmp["bin"] = pd.qcut(tmp[pred_col], q=bins, duplicates="drop")
    grouped = tmp.groupby("bin").agg(actual_mean=(actual_col, "mean"),
                                     predicted_mean=(pred_col, "mean"),
                                     n=("bin","size")).reset_index()
    fig = px.scatter(grouped, x="predicted_mean", y="actual_mean", size="n",
                     title=title, labels={"predicted_mean":"Predicted (bin mean)","actual_mean":"Actual (bin mean)"})
    # parity
    both = grouped[["predicted_mean","actual_mean"]].values.flatten()
    if len(both):
        minv, maxv = float(np.nanmin(both)), float(np.nanmax(both))
        fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv],
                                 mode="lines", name="Perfect calibration", line=dict(dash="dash")))
    fig.update_layout(margin=dict(t=50, r=10, l=10, b=10), height=350)
    return fig

def download_bytesio(df_dict, fname="export.xlsx"):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for sheet, frame in df_dict.items():
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)
    bio.seek(0)
    return bio

# ------------- Sidebar: Data Input -------------
st.sidebar.header("📥 Data & Settings")

# Provided file path (bundled in repo)
provided_path = os.path.join("sample_data", "Actual_vs_Predicted_Results.xlsx")
provided_sheets = []
if os.path.exists(provided_path):
    try:
        xls = pd.ExcelFile(provided_path)
        provided_sheets = xls.sheet_names
    except Exception as e:
        st.sidebar.error(f"Failed to read provided Excel: {e}")

source = st.sidebar.radio(
    "Data source",
    (["Use provided file (Actual_vs_Predicted_Results.xlsx)"] if os.path.exists(provided_path) else []) + ["Upload file"],
    index=0 if os.path.exists(provided_path) else 0
)

df = None
if source.startswith("Use provided") and os.path.exists(provided_path):
    if len(provided_sheets) > 1:
        sheet = st.sidebar.selectbox("Choose Excel sheet", provided_sheets, index=0)
    else:
        sheet = provided_sheets[0] if provided_sheets else 0
    df = pd.read_excel(provided_path, sheet_name=sheet)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = st.sidebar.selectbox("Choose Excel sheet", xls.sheet_names, index=0)
            df = pd.read_excel(uploaded, sheet_name=sheet)

if df is None or df.empty:
    st.error("No data loaded. Please use the provided Excel or upload your file.")
    st.stop()

# Clean column names (trim)
df.columns = [c.strip() for c in df.columns]

# Column detection & mapping
det = identify_columns(df)

with st.sidebar.expander("🔎 Column Mapping", expanded=True):
    date_col = st.selectbox("Date/Time column (optional)", [None] + list(df.columns),
                            index=(df.columns.tolist().index(det["date_col"]) + 1) if det["date_col"] in df.columns else 0)
    numeric_cols = det["numeric_cols"] if det["numeric_cols"] else []
    if not numeric_cols:
        st.sidebar.error("No numeric columns detected. Please verify your data.")
        st.stop()
    actual_default = det["actual_candidates"][0] if det["actual_candidates"] else numeric_cols[0]
    actual_col = st.selectbox("Actual column", numeric_cols,
                              index=(numeric_cols.index(actual_default) if actual_default in numeric_cols else 0))
    pred_defaults = det["pred_candidates"] if det["pred_candidates"] else (numeric_cols[1:2] if len(numeric_cols) > 1 else numeric_cols[:1])
    pred_cols = st.multiselect("Predicted column(s)", numeric_cols,
                               default=[c for c in pred_defaults if c in numeric_cols] or [numeric_cols[0]])
    model_col = st.selectbox("Model column (optional)", [None] + list(df.columns),
                             index=(df.columns.tolist().index(det["model_col"]) + 1) if det["model_col"] in df.columns else 0)
    ticker_col = st.selectbox("Ticker column (optional)", [None] + list(df.columns),
                              index=(df.columns.tolist().index(det["ticker_col"]) + 1) if det["ticker_col"] in df.columns else 0)

# Parse date if provided
if date_col:
    parsed = to_datetime_safe(df[date_col])
    if parsed is not None:
        df[date_col] = parsed
    else:
        st.sidebar.warning("Could not parse dates; displaying as plain text.")
else:
    idx_name = "row_index"
    while idx_name in df.columns:
        idx_name += "_"
    df[idx_name] = np.arange(1, len(df) + 1)
    date_col = idx_name

# Optional filters
if ticker_col:
    tickers = sorted(df[ticker_col].dropna().astype(str).unique().tolist())
    selected_tickers = st.sidebar.multiselect("Filter tickers", tickers, default=tickers)
    if selected_tickers:
        df = df[df[ticker_col].astype(str).isin(selected_tickers)]

# Date range filter
if np.issubdtype(df[date_col].dtype, np.datetime64):
    min_d, max_d = df[date_col].min(), df[date_col].max()
    d1, d2 = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if d1 and d2:
        df = df[(df[date_col] >= pd.to_datetime(d1)) & (df[date_col] <= pd.to_datetime(d2))]

st.title("📈 Tesla Actual vs Predicted — Dashboard")
st.caption("Analyze model performance with premium, presentation-ready visuals.")

# Select which predicted series to focus on (for deep dive)
if not pred_cols:
    st.error("Please select at least one predicted column.")
    st.stop()

focus_pred = st.selectbox("Focus predicted series", pred_cols, index=0)

# If no model column but many predicted columns, create one for per-series grouping
work = df.copy()
if model_col is None and len(pred_cols) > 1:
    melted = []
    for pc in pred_cols:
        tmp = work[[date_col, actual_col]].copy()
        tmp["model"] = pc
        tmp["predicted"] = work[pc]
        melted.append(tmp)
    work = pd.concat(melted, ignore_index=True)
    model_col = "model"
    pred_col = "predicted"
else:
    pred_col = "predicted"
    work[pred_col] = work[focus_pred]

# Overall metrics (and per model if available)
overall_metrics = compute_metrics(work, actual_col, pred_col, groupby=None)
with st.container():
    st.subheader("Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def fmt(x, nd=3):
        if pd.isna(x):
            return "—"
        return f"{x:.{nd}f}"
    with c1: st.metric("R²", fmt(overall_metrics["R2"].iloc[0], 3))
    with c2: st.metric("RMSE", fmt(overall_metrics["RMSE"].iloc[0], 2))
    with c3: st.metric("MAE", fmt(overall_metrics["MAE"].iloc[0], 2))
    with c4: st.metric("MAPE %", fmt(overall_metrics["MAPE%"].iloc[0], 2))
    with c5: st.metric("sMAPE %", fmt(overall_metrics["sMAPE%"].iloc[0], 2))
    with c6: st.metric("Observations", int(overall_metrics["n"].iloc[0]))

# Time-series overlay (Actual vs Predicted)
with st.container():
    st.markdown("### Actual vs Predicted (Time-Series)")
    overlay_df = work[[date_col, actual_col, pred_col]].rename(columns={date_col:"x"})
    fig_ts = line_overlay(overlay_df, x="x", y_cols=[actual_col, pred_col], title="")
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown('<div class="caption">Use the date range filter in the sidebar to zoom into specific periods.</div>', unsafe_allow_html=True)

# Diagnostics tabs
tabs = st.tabs(["🔬 Diagnostics", "📊 Compare Models", "🧾 Data & Export"])

with tabs[0]:
    deep = add_residuals(work[[date_col, actual_col, pred_col]], actual_col, pred_col)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Predicted vs Actual")
        st.plotly_chart(scatter_parity(deep, actual_col, pred_col), use_container_width=True)
    with c2:
        st.markdown("#### Calibration")
        st.plotly_chart(calibration_curve(deep, actual_col, pred_col, bins=10), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Residuals over Time")
        st.plotly_chart(residual_timeseries(deep, x=date_col), use_container_width=True)
    with c4:
        st.markdown("#### Residual Distribution")
        st.plotly_chart(residual_distribution(deep), use_container_width=True)

with tabs[1]:
    st.markdown("#### Model Comparison")
    if "model" in work.columns and work["model"].nunique() > 0:
        model_key = "model"
        metrics_by_model = compute_metrics(work, actual_col, pred_col, groupby=model_key).sort_values("R2", ascending=False)
        st.dataframe(metrics_by_model, use_container_width=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Bar chart for RMSE by model
        fig_bar = px.bar(metrics_by_model, x=model_key, y="RMSE", text="RMSE", title="RMSE by Model")
        fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_bar.update_layout(yaxis_title="RMSE", xaxis_title="", height=420, margin=dict(t=50, r=10, l=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Add a **model** column or select multiple predicted series to unlock model comparison.")

with tabs[2]:
    st.markdown("#### Preview Data")
    st.dataframe(work.head(200), use_container_width=True)

    # Build exports
    deep = add_residuals(work[[date_col, actual_col, pred_col]], actual_col, pred_col)
    per_model = compute_metrics(work, actual_col, pred_col, groupby="model") if "model" in work.columns else pd.DataFrame()
    overall = compute_metrics(work, actual_col, pred_col, groupby=None)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    colE1, colE2 = st.columns(2)
    with colE1:
        st.download_button(
            "⬇️ Download Residuals (CSV)",
            data=deep.to_csv(index=False).encode("utf-8"),
            file_name="residuals.csv",
            mime="text/csv"
        )
    with colE2:
        xls_bytes = download_bytesio({"Residuals": deep, "Metrics_by_Model": per_model if not per_model.empty else pd.DataFrame(), "Metrics_overall": overall})
        st.download_button(
            "⬇️ Download Metrics + Residuals (Excel)",
            data=xls_bytes,
            file_name="metrics_and_residuals.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.markdown('<span class="small">Tip: Map your columns in the sidebar. Add a <code>model</code> column or choose multiple predicted columns to compare models.</span>', unsafe_allow_html=True)
