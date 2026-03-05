# 🛡️ AEGIS — Arms & Escalation Geopolitical Intelligence System

A comprehensive, interactive Streamlit dashboard that performs **Descriptive, Diagnostic, Predictive, and Prescriptive analysis** on global arms transfer data — answering the central question: **Where are potentially destabilising arms buildups occurring, and where should peacekeeping resources be prioritised?**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly)

---

## 🎯 Objective

To understand **what drives escalation risk in the global arms trade** through four layers of analytics:

| Analysis Type | Intelligence Analogy | Question Answered | Techniques Used |
|---|---|---|---|
| **Descriptive** | SIGINT | What does the arms landscape look like? | Sankey flows, treemaps, timelines, KPIs |
| **Diagnostic** | HUMINT | What drives dangerous accumulation? | Correlation analysis, Chi-Square, Cramér's V, risk combos |
| **Predictive** | MASINT | Can we predict escalation risk? | Logistic Regression, Random Forest, Gradient Boosting, ROC |
| **Prescriptive** | OSINT | Where should intervention go? | Risk simulator, policy recommendations, impact matrix |

---

## 🚀 Quick Start

### Local Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/aegis-arms-intelligence.git
cd aegis-arms-intelligence

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

---

## 📊 Dashboard Features

- **Interactive Sidebar Filters** — slice by year, exporter, region, weapon category, risk level, and conflict proximity
- **6 KPI Cards** — real-time intelligence summary
- **Sankey Flow Diagram** — arms flow from exporter to importer regions by value
- **Treemap Drill-Downs** — weapon category → subtype → offensive/defensive classification
- **Radar Charts** — high-risk vs low-risk country profile comparison
- **Chi-Square Tests** — statistical significance with Cramér's V effect sizes
- **3 ML Models** — with cross-validated AUC, ROC curves, and feature importance
- **Escalation Risk Simulator** — interactive gauge to estimate risk for hypothetical transfers
- **Impact vs Feasibility Matrix** — prioritise policy interventions
- **Embargo Circumvention Analysis** — who is still arming embargoed states?

---

## 📁 Project Structure

```
aegis-arms-intelligence/
├── app.py                  # Main Streamlit application
├── arms_trade.csv          # Synthetic arms transfer dataset (1,500 × 25)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── .gitignore
└── README.md
```

---

## 📦 Dataset

- **1,500 arms transfers** × **25 features**
- Target variable: `Escalation_Risk` (High / Medium / Low)
- Features include deal characteristics, weapon classification (offensive/defensive), exporter alliance type, importer geopolitical indicators (stability, democracy, conflict proximity, territorial disputes, resource dependence), and arms import trends
- Synthetic dataset modelled on real-world SIPRI Arms Transfers Database patterns

---

## 🛠️ Tech Stack

- **Streamlit** — Dashboard framework
- **Plotly** — Interactive visualisations
- **scikit-learn** — Predictive models
- **SciPy** — Statistical tests
- **Pandas / NumPy** — Data processing

---

## 📄 License

MIT License — free to use, modify, and distribute.
