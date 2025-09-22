# Tesla Actual vs Predicted — Streamlit Dashboard

A polished, production-ready Streamlit dashboard to **analyze and visualize Actual vs Predicted** performance for Tesla (or any asset) with best‑practice UI/UX.

https://github.com/ — Add this repo and deploy to Streamlit Community Cloud in minutes.

---

## ✨ Features
- **Upload CSV or Excel** (multi-sheet) or use the included **provided Tesla Actual vs Predicted Excel file**.
- **Smart column mapping** for *Actual*, *Predicted*, *Date/Index*, and optional *Model* column.
- **Beautiful, interactive plots** (Plotly): time‑series overlay, predicted‑vs‑actual scatter, residuals timeline, error distribution, and calibration.
- **Metrics suite**: R², RMSE, MAE, MAPE, sMAPE; auto‑computed **per model** and **overall**.
- **Model comparison** view with league table and bar chart.
- **Date range filter** and **rolling error** diagnostics.
- **Download** processed data (with residuals) and metrics as CSV/Excel.
- **Crisp theming** via `.streamlit/config.toml` for a premium look.

> Works out‑of‑the‑box with the included sample file: `sample_data/Actual_vs_Predicted_Results.xlsx`.

---

## 📦 Project Structure
```
tesla-actual-vs-predicted-dashboard/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .streamlit/
│  └─ config.toml
├─ sample_data/
│  └─ tesla_actual_vs_predicted_sample.csv
└─ assets/   (optional: logos, hero images)
```

---

## 🚀 Quickstart (Local)
1. **Create & activate** a virtual env (recommended).
2. **Install deps**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run**:
   ```bash
   streamlit run app.py
   ```
4. Open the provided local URL in your browser.

---

## 📂 Data Input
- **CSV/Excel** formats supported.
- Recommended columns (case‑insensitive):
  - `date` (or `time`, `timestamp`, `ds`) — optional but preferred
  - `actual` — numeric
  - `predicted` — numeric (or multiple predicted columns)
  - `model` — optional; enables per‑model comparisons (e.g., *XGBoost*, *LSTM*, *ARIMA*)
  - `ticker` — optional (defaults to TSLA if absent)
- If your file has multiple predicted columns, you can **select one or many** to compare.

> If no date column is present, the app will index rows as 1..N.

---

## 🧮 Metrics Definitions
- **R²**: Coefficient of determination.
- **RMSE**: Root Mean Squared Error.
- **MAE**: Mean Absolute Error.
- **MAPE**: Mean Absolute Percentage Error (skips zeros).
- **sMAPE**: Symmetric MAPE.

---

## ☁️ Deploy to Streamlit Community Cloud
1. Push this folder to a **GitHub repo**.
2. Go to **share.streamlit.io** → **New app** → select your repo/branch.
3. Main file path: `app.py`
4. Add any **secrets** if needed (not required for this app).

---

## 🧰 Tips
- Keep column names simple. The mapper lets you pick any columns.
- Add a `model` column to compare multiple models in one file.
- Use the **date filter** to inspect performance in specific windows.
- Export **residuals & metrics** for your report slides.

---

## 📝 License
MIT
