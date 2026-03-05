import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS — Arms & Escalation Geopolitical Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ── Header bar ── */
    .header-bar {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1rem 1.8rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    .header-bar::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00c896, #0ea5e9, #00c896);
    }
    .header-left h1 {
        font-family: 'Inter', sans-serif;
        color: #f0f6fc;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 2px;
        display: inline;
    }
    .header-left .subtitle {
        color: #8b949e;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }
    .pill {
        display: inline-block;
        font-size: 0.58rem;
        padding: 2px 9px;
        border-radius: 10px;
        letter-spacing: 0.8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        vertical-align: middle;
        margin-left: 0.6rem;
    }
    .pill-live { background: rgba(0,200,150,0.15); color: #00c896; }
    .pill-critical { background: rgba(248,81,73,0.15); color: #f85149; }
    .pill-high { background: rgba(210,153,34,0.15); color: #d29922; }
    .pill-medium { background: rgba(0,200,150,0.12); color: #00c896; }
    .pill-new { background: rgba(14,165,233,0.15); color: #0ea5e9; }

    /* ── Compact KPI strip ── */
    .kpi-strip {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .kpi-item {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        flex: 1;
        text-align: center;
    }
    .kpi-item:hover { border-color: #30363d; }
    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .kpi-lbl {
        color: #8b949e;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.15rem;
        font-weight: 500;
    }
    .kpi-sub {
        font-size: 0.65rem;
        margin-top: 0.1rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub.red { color: #f85149; }
    .kpi-sub.grn { color: #00c896; }
    .kpi-sub.ylw { color: #d29922; }
    .kpi-sub.muted { color: #8b949e; }

    /* ── Panel card ── */
    .panel {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        height: 100%;
    }
    .panel-title {
        color: #f0f6fc;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .panel-title .pill { margin-left: auto; }

    /* ── Stat row inside panel ── */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid #21262d;
        font-size: 0.78rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-row .label { color: #8b949e; }
    .stat-row .value { color: #f0f6fc; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
    .stat-row .value.red { color: #f85149; }
    .stat-row .value.grn { color: #00c896; }

    /* ── Section divider ── */
    .sec-div {
        border-left: 3px solid #00c896;
        padding: 0.4rem 0.8rem;
        margin: 1.2rem 0 0.6rem 0;
    }
    .sec-div h3 {
        color: #f0f6fc;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    .sec-div p { color: #8b949e; font-size: 0.72rem; margin: 0.1rem 0 0 0; }

    /* ── Insight callout ── */
    .callout {
        background: rgba(0,200,150,0.04);
        border: 1px solid rgba(0,200,150,0.12);
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.78rem;
        line-height: 1.6;
        color: #c9d1d9;
        margin: 0.4rem 0;
    }
    .callout strong { color: #58d5a8; }

    /* ── Rx card ── */
    .rx-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .rx-card h4 {
        color: #f0f6fc;
        margin: 0 0 0.3rem 0;
        font-size: 0.82rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .rx-card p { color: #8b949e; margin: 0; font-size: 0.76rem; line-height: 1.55; }

    /* ── Model card (for predictive tab) ── */
    .model-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 1.2rem;
        text-align: center;
    }
    .model-card .model-name { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }
    .model-card .model-auc {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .model-card .model-std { color: #8b949e; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; }

    /* ── Tabs ── */
    div[data-testid="stTabs"] button {
        background: transparent !important;
        color: #8b949e !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.76rem !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #00c896 !important;
        border-bottom: 2px solid #00c896 !important;
    }

    /* ── Sidebar ── */
    .stSidebar > div { background: #0d1117; }

    /* ── Misc ── */
    div[data-testid="stExpander"] { border: 1px solid #21262d; border-radius: 6px; }
    header[data-testid="stHeader"] { background: #0d1117; }

    /* Tighten default streamlit spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("arms_trade.csv")
    df['Escalation_Flag'] = df['Escalation_Risk'].map({'High': 2, 'Medium': 1, 'Low': 0})
    df['High_Risk_Flag'] = (df['Escalation_Risk'] == 'High').astype(int)
    df['Offensive_Flag'] = (df['Weapon_Class'] == 'Offensive').astype(int)
    df['YearGroup'] = pd.cut(df['Year'], bins=[2004,2009,2014,2019,2025],
                              labels=['2005-09','2010-14','2015-19','2020-24'])
    df['DealSize'] = pd.cut(df['Deal_Value_USD_M'], bins=[0,20,80,200,9999],
                             labels=['Small (<$20M)','Medium ($20-80M)','Large ($80-200M)','Mega (>$200M)'])
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Slice the intelligence dataset")
    year_range = st.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()),
                           (int(df['Year'].min()), int(df['Year'].max())))
    exp_filter = st.multiselect("Exporter", sorted(df['Exporter'].unique()), default=sorted(df['Exporter'].unique()))
    imp_region_filter = st.multiselect("Importer Region", df['Importer_Region'].unique(), default=df['Importer_Region'].unique())
    weapon_filter = st.multiselect("Weapon Category", df['Weapon_Category'].unique(), default=df['Weapon_Category'].unique())
    risk_filter = st.multiselect("Escalation Risk", ['High','Medium','Low'], default=['High','Medium','Low'])
    conflict_filter = st.multiselect("Conflict Proximity", ['Yes','No'], default=['Yes','No'])

mask = (
    df['Year'].between(year_range[0], year_range[1]) &
    df['Exporter'].isin(exp_filter) &
    df['Importer_Region'].isin(imp_region_filter) &
    df['Weapon_Category'].isin(weapon_filter) &
    df['Escalation_Risk'].isin(risk_filter) &
    df['Importer_Conflict_Proximity'].isin(conflict_filter)
)
dff = df[mask].copy()

# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
RISK_COLORS = {'High': '#f85149', 'Medium': '#d29922', 'Low': '#00c896'}
CLASS_COLORS = {'Offensive': '#f85149', 'Defensive': '#0ea5e9'}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, -apple-system, sans-serif', color='#c9d1d9', size=11),
    margin=dict(l=40, r=60, t=40, b=35),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color='#c9d1d9')),
    title_font=dict(size=13, color='#f0f6fc', family='Inter, sans-serif'),
)

def styled_chart(fig, height=400):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor='rgba(201,209,217,0.06)', zerolinecolor='rgba(201,209,217,0.06)')
    fig.update_yaxes(gridcolor='rgba(201,209,217,0.06)', zerolinecolor='rgba(201,209,217,0.06)')
    return fig


# ─────────────────────────────────────────────────────────────
# COMPUTED VALUES
# ─────────────────────────────────────────────────────────────
total = len(dff)
total_value = dff['Deal_Value_USD_M'].sum()
high_risk_count = dff['High_Risk_Flag'].sum()
high_risk_pct = (high_risk_count / total * 100) if total > 0 else 0
offensive_pct = (dff['Offensive_Flag'].sum() / total * 100) if total > 0 else 0
top_exporter = dff['Exporter'].value_counts().index[0] if total > 0 else 'N/A'
top_exporter_n = dff['Exporter'].value_counts().iloc[0] if total > 0 else 0
accel_count = len(dff[dff['Arms_Import_Trend']=='Accelerating'])
accel_pct = (accel_count / total * 100) if total > 0 else 0
conflict_deals = len(dff[dff['Importer_Conflict_Proximity']=='Yes'])
conflict_pct = (conflict_deals / total * 100) if total > 0 else 0


# ═════════════════════════════════════════════════════════════
# HEADER BAR
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='header-bar'>
    <div class='header-left'>
        <h1>AEGIS</h1><span class='pill pill-live'>LIVE</span>
        <div class='subtitle'>Arms & Escalation Geopolitical Intelligence System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# KPI STRIP (compact, single row)
# ═════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='kpi-strip'>
    <div class='kpi-item'>
        <div class='kpi-val'>{total:,}</div>
        <div class='kpi-lbl'>Transfers</div>
        <div class='kpi-sub muted'>${total_value:,.0f}M</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{high_risk_count}</div>
        <div class='kpi-lbl'>High Risk</div>
        <div class='kpi-sub red'>{high_risk_pct:.1f}%</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{offensive_pct:.0f}%</div>
        <div class='kpi-lbl'>Offensive</div>
        <div class='kpi-sub ylw'>{dff['Offensive_Flag'].sum()} systems</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{top_exporter}</div>
        <div class='kpi-lbl'>Top Exporter</div>
        <div class='kpi-sub muted'>{top_exporter_n} deals</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{accel_pct:.0f}%</div>
        <div class='kpi-lbl'>Accelerating</div>
        <div class='kpi-sub red'>{accel_count} importers</div>
    </div>
    <div class='kpi-item'>
        <div class='kpi-val'>{conflict_deals}</div>
        <div class='kpi-lbl'>Conflict Zone</div>
        <div class='kpi-sub ylw'>{conflict_pct:.0f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# COMMAND OVERVIEW (before tabs — most critical info visible immediately)
# ═════════════════════════════════════════════════════════════

# Three unequal columns: Timeline (wide) | Risk donut + top stats (narrow) | Regional summary (medium)
ov1, ov2, ov3 = st.columns([5, 2, 3])

with ov1:
    # Timeline — the single most important overview chart
    yearly = dff.groupby('Year').agg(
        Deals=('Year','count'), Value=('Deal_Value_USD_M','sum'),
        High_Risk=('High_Risk_Flag','sum')
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['Deals'], name='Total Deals',
                         marker_color='rgba(14,165,233,0.35)'), secondary_y=False)
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['High_Risk'], name='High Risk',
                         marker_color='rgba(248,81,73,0.45)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Value'], name='Value ($M)',
                             mode='lines+markers', line=dict(color='#00c896', width=2),
                             marker=dict(size=5)), secondary_y=True)
    fig.update_layout(barmode='overlay', yaxis_title='Deals', yaxis2_title='Value ($M)',
                      title='Arms Transfers Over Time', legend=dict(orientation='h', y=-0.15))
    st.plotly_chart(styled_chart(fig, 340), use_container_width=True)

with ov2:
    # Risk distribution donut (compact)
    risk_counts = dff['Escalation_Risk'].value_counts()
    fig = go.Figure(go.Pie(
        labels=risk_counts.index, values=risk_counts.values,
        hole=0.7, marker=dict(colors=[RISK_COLORS.get(x, '#0ea5e9') for x in risk_counts.index]),
        textinfo='percent', textfont=dict(size=11, family='Inter'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
        sort=False
    ))
    fig.update_layout(showlegend=True, legend=dict(orientation='h', y=-0.1, font=dict(size=9)),
                      title='Risk Distribution',
                      annotations=[dict(text=f'<b>{high_risk_pct:.0f}%</b><br>HIGH', x=0.5, y=0.5, font_size=18,
                                        font_color='#f85149', showarrow=False, font_family='JetBrains Mono')])
    st.plotly_chart(styled_chart(fig, 340), use_container_width=True)

with ov3:
    # Regional threat summary — compact stat panel
    region_risk = dff.groupby('Importer_Region').agg(
        Deals=('Year','count'), HR=('High_Risk_Flag','sum'), Val=('Deal_Value_USD_M','sum')
    ).reset_index()
    region_risk['HR_Pct'] = (region_risk['HR'] / region_risk['Deals'] * 100).round(1)
    region_risk = region_risk.sort_values('HR_Pct', ascending=False)

    st.markdown("<div class='panel'><div class='panel-title'>Regional Threat Summary <span class='pill pill-live'>LIVE</span></div>", unsafe_allow_html=True)
    for _, row in region_risk.iterrows():
        risk_cls = 'red' if row['HR_Pct'] > 40 else 'value'
        st.markdown(f"""<div class='stat-row'>
            <span class='label'>{row['Importer_Region']}</span>
            <span class='value {risk_cls}'>{row['HR_Pct']:.0f}% high risk &nbsp; ${row['Val']:,.0f}M</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Situation Assessment Panel ──
# Compute dynamic insights
top_imp_region = dff.groupby('Importer_Region')['Deal_Value_USD_M'].sum().idxmax() if total > 0 else 'N/A'
top_exp_region = dff.groupby('Exporter_Region')['Deal_Value_USD_M'].sum().idxmax() if total > 0 else 'N/A'
recent_3yr = dff[dff['Year'] >= dff['Year'].max() - 2]
older_3yr = dff[dff['Year'] <= dff['Year'].min() + 2]
recent_val = recent_3yr['Deal_Value_USD_M'].sum()
older_val = older_3yr['Deal_Value_USD_M'].sum()
val_change = ((recent_val - older_val) / older_val * 100) if older_val > 0 else 0
top_risk_importer = dff[dff['Escalation_Risk']=='High'].groupby('Importer').size().idxmax() if high_risk_count > 0 else 'N/A'
top_risk_importer_n = dff[dff['Escalation_Risk']=='High'].groupby('Importer').size().max() if high_risk_count > 0 else 0
offensive_in_conflict = dff[(dff['Weapon_Class']=='Offensive') & (dff['Importer_Conflict_Proximity']=='Yes')]
offensive_conflict_val = offensive_in_conflict['Deal_Value_USD_M'].sum()

val_trend_word = 'increased' if val_change > 0 else 'decreased'
val_trend_cls = 'red' if val_change > 0 else 'grn'

st.markdown(f"""
<div class='panel' style='margin-top:0.5rem;'>
    <div class='panel-title'>Situation Assessment <span class='pill pill-critical'>ANALYSIS</span></div>
    <div class='stat-row'>
        <span class='label'>Transfer Volume Trend (last 3yr vs first 3yr)</span>
        <span class='value {val_trend_cls}'>{val_trend_word} {abs(val_change):.0f}%</span>
    </div>
    <div class='stat-row'>
        <span class='label'>Primary Corridor</span>
        <span class='value'>{top_exp_region} → {top_imp_region}</span>
    </div>
    <div class='stat-row'>
        <span class='label'>Highest-Risk Importer</span>
        <span class='value red'>{top_risk_importer} ({top_risk_importer_n} high-risk deals)</span>
    </div>
    <div class='stat-row'>
        <span class='label'>Offensive Systems to Conflict Zones</span>
        <span class='value red'>${offensive_conflict_val:,.0f}M ({len(offensive_in_conflict)} deals)</span>
    </div>
    <div style='margin-top:0.5rem; font-size:0.75rem; color:#8b949e; line-height:1.5;'>
        <strong style='color:#58d5a8;'>Assessment:</strong>
        {f"Arms transfer values have {val_trend_word} by {abs(val_change):.0f}% comparing the earliest and most recent 3-year windows."
        } The dominant export corridor runs from <strong style='color:#f0f6fc;'>{top_exp_region}</strong> to
        <strong style='color:#f0f6fc;'>{top_imp_region}</strong>.
        <strong style='color:#f85149;'>{top_risk_importer}</strong> is the single largest recipient of high-risk classified transfers.
        ${offensive_conflict_val:,.0f}M in offensive systems have been delivered to conflict-proximate states — these represent
        the highest escalation potential in the dataset.
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Descriptive", "Diagnostic", "Predictive", "Prescriptive"
])


# =============================================================
# TAB 1: DESCRIPTIVE
# =============================================================
with tab1:
    # ── Full-width Sankey ──
    st.markdown("<div class='sec-div'><h3>Arms Flow Patterns</h3><p>Dollar value flows between exporter and importer regions</p></div>", unsafe_allow_html=True)

    flow = dff.groupby(['Exporter_Region','Importer_Region'])['Deal_Value_USD_M'].sum().reset_index()
    flow = flow[flow['Deal_Value_USD_M'] > 0].nlargest(20, 'Deal_Value_USD_M')
    all_labels = list(pd.unique(flow[['Exporter_Region','Importer_Region']].values.ravel()))
    src_indices = [all_labels.index(x) for x in flow['Exporter_Region']]
    tgt_indices = [all_labels.index(x) for x in flow['Importer_Region']]
    exp_regions = set(flow['Exporter_Region'].unique())
    node_colors = ['rgba(0,200,150,0.8)' if l in exp_regions else 'rgba(14,165,233,0.8)' for l in all_labels]

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=all_labels, color=node_colors,
                  line=dict(color='#21262d', width=0.5)),
        link=dict(source=src_indices, target=tgt_indices,
                  value=flow['Deal_Value_USD_M'].values,
                  color='rgba(0,200,150,0.25)')
    ))
    fig.update_layout(title='Exporter Region → Importer Region (by $M value)')
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    # ── Flow analysis insight ──
    top_flow = flow.nlargest(1, 'Deal_Value_USD_M').iloc[0] if len(flow) > 0 else None
    if top_flow is not None:
        # Find concentration: how much of total value is in top 3 flows
        top3_val = flow.nlargest(3, 'Deal_Value_USD_M')['Deal_Value_USD_M'].sum()
        concentration = (top3_val / flow['Deal_Value_USD_M'].sum() * 100) if flow['Deal_Value_USD_M'].sum() > 0 else 0

        fa1, fa2 = st.columns([2, 3])
        with fa1:
            st.markdown(f"""<div class='panel'>
                <div class='panel-title'>Flow Concentration</div>
                <div class='stat-row'><span class='label'>Dominant corridor</span>
                    <span class='value'>{top_flow['Exporter_Region']} → {top_flow['Importer_Region']}</span></div>
                <div class='stat-row'><span class='label'>Corridor value</span>
                    <span class='value'>${top_flow['Deal_Value_USD_M']:,.0f}M</span></div>
                <div class='stat-row'><span class='label'>Top 3 corridors share</span>
                    <span class='value {"red" if concentration > 60 else ""}'>{concentration:.0f}% of all flows</span></div>
            </div>""", unsafe_allow_html=True)
        with fa2:
            diversification = 'highly concentrated' if concentration > 60 else 'moderately diversified' if concentration > 40 else 'well diversified'
            st.markdown(f"""<div class='callout'>
                <strong>Flow Analysis:</strong> Global arms transfers are <strong>{diversification}</strong> —
                the top 3 regional corridors account for {concentration:.0f}% of total dollar flows.
                The single largest corridor ({top_flow['Exporter_Region']} → {top_flow['Importer_Region']})
                moves ${top_flow['Deal_Value_USD_M']:,.0f}M. {"This concentration increases systemic risk — disruption to a single supplier relationship could destabilise multiple regional security architectures." if concentration > 50 else "Diversified flows reduce single-point-of-failure risk in the global arms supply chain."}
            </div>""", unsafe_allow_html=True)

    # ── Three columns: Exporters | Treemap | Importers ──
    st.markdown("<div class='sec-div'><h3>Key Players & Arsenal</h3></div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns([3, 4, 3])

    with d1:
        exp_agg = dff.groupby('Exporter').agg(Deals=('Year','count'), Value=('Deal_Value_USD_M','sum')).reset_index()
        exp_agg = exp_agg.sort_values('Value', ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            y=exp_agg['Exporter'], x=exp_agg['Value'], orientation='h',
            marker=dict(color=exp_agg['Value'], colorscale=[[0,'#0c4a3e'],[1,'#00c896']]),
            text=exp_agg.apply(lambda r: f"${r['Value']:,.0f}M", axis=1),
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Top Exporters ($M)')
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    with d2:
        tree_df = dff.groupby(['Weapon_Category','Weapon_Subtype','Weapon_Class']).size().reset_index(name='Count')
        fig = px.treemap(tree_df, path=['Weapon_Category','Weapon_Subtype','Weapon_Class'], values='Count',
                         color='Weapon_Class', color_discrete_map=CLASS_COLORS,
                         title='Weapon Hierarchy')
        fig.update_traces(textfont_size=11, textposition='middle center', insidetextfont=dict(size=10))
        fig.update_layout(uniformtext=dict(minsize=8, mode='show'))
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    with d3:
        imp_agg = dff.groupby('Importer').agg(Deals=('Year','count'), Value=('Deal_Value_USD_M','sum'),
                                               Avg_Risk=('Escalation_Flag','mean')).reset_index()
        imp_agg = imp_agg.sort_values('Value', ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            y=imp_agg['Importer'], x=imp_agg['Value'], orientation='h',
            marker=dict(color=imp_agg['Avg_Risk'],
                        colorscale=[[0,'#00c896'],[0.5,'#d29922'],[1,'#f85149']],
                        colorbar=dict(title='Risk', thickness=10)),
            text=imp_agg.apply(lambda r: f"${r['Value']:,.0f}M", axis=1),
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Top Importers (by risk)')
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    # ── Key players insight ──
    top3_exporters = dff.groupby('Exporter')['Deal_Value_USD_M'].sum().nlargest(3)
    top3_share = (top3_exporters.sum() / total_value * 100) if total_value > 0 else 0
    riskiest_importer = imp_agg.sort_values('Avg_Risk', ascending=False).iloc[0] if len(imp_agg) > 0 else None
    offensive_share_by_cat = dff.groupby('Weapon_Category')['Offensive_Flag'].mean().sort_values(ascending=False)
    most_offensive_cat = offensive_share_by_cat.index[0] if len(offensive_share_by_cat) > 0 else 'N/A'
    most_offensive_pct = (offensive_share_by_cat.iloc[0] * 100) if len(offensive_share_by_cat) > 0 else 0

    st.markdown(f"""<div class='callout'>
        <strong>Key Finding:</strong> The top 3 exporters ({', '.join(top3_exporters.index)}) control <strong>{top3_share:.0f}%</strong>
        of global transfer value — a significant supplier oligopoly.
        {f"<strong>{riskiest_importer['Importer']}</strong> stands out as the highest average-risk importer among top recipients (risk score {riskiest_importer['Avg_Risk']:.2f}/2.0)." if riskiest_importer is not None else ""}
        <strong>{most_offensive_cat}</strong> is the most offensively-skewed category at {most_offensive_pct:.0f}% offensive classification.
    </div>""", unsafe_allow_html=True)

    # ── Two columns: Offensive/defensive by region | Deal frameworks + Sunburst ──
    e1, e2 = st.columns([3, 2])
    with e1:
        class_region = dff.groupby(['Importer_Region','Weapon_Class']).size().reset_index(name='Count')
        fig = px.bar(class_region, x='Importer_Region', y='Count', color='Weapon_Class',
                     color_discrete_map=CLASS_COLORS, barmode='group',
                     title='Offensive vs Defensive by Region')
        fig.update_layout(xaxis_tickangle=-25, xaxis_tickfont_size=10, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    with e2:
        sun_df = dff.groupby(['Exporter_Alliance','Deal_Framework','Escalation_Risk']).size().reset_index(name='Count')
        fig = px.sunburst(sun_df, path=['Exporter_Alliance','Deal_Framework','Escalation_Risk'],
                          values='Count', color='Escalation_Risk', color_discrete_map=RISK_COLORS,
                          title='Alliance → Framework → Risk')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)


# =============================================================
# TAB 2: DIAGNOSTIC
# =============================================================
with tab2:

    # ── Row 1: Full-width correlation heatmap ──
    st.markdown("<div class='sec-div'><h3>Feature Correlations</h3><p>Which numeric features correlate with high escalation risk?</p></div>", unsafe_allow_html=True)

    heat_cols = ['Deal_Value_USD_M','Importer_GDP_Per_Capita','Importer_Political_Stability',
                 'Importer_Democracy_Index','Importer_Military_Spend_Pct_GDP',
                 'Offensive_Flag','Delivery_Timeline_Months','High_Risk_Flag']
    fig = px.imshow(dff[heat_cols].corr(), text_auto='.2f',
                    color_continuous_scale=[[0,'#00c896'],[0.5,'#0d1117'],[1,'#f85149']],
                    zmin=-1, zmax=1, title='Feature Correlation Matrix')
    fig.update_traces(textfont_size=10)
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    # ── Row 2: Three columns — Radar | Insight panel | Correlation bars ──
    st.markdown("<div class='sec-div'><h3>Risk Profile Analysis</h3><p>High-risk vs Low-risk importer profiles and key correlations</p></div>", unsafe_allow_html=True)

    profile_cols = ['Importer_Political_Stability','Importer_Democracy_Index',
                    'Importer_Military_Spend_Pct_GDP','Deal_Value_USD_M','Offensive_Flag']
    profile_labels = ['Political Stability','Democracy Index','Military Spend % GDP',
                      'Deal Value ($M)','Offensive Weapon Share']

    r1, r2, r3 = st.columns([3, 2, 3])

    with r1:
        high_vals, low_vals = [], []
        for col in profile_cols:
            h = dff[dff['Escalation_Risk']=='High'][col].mean()
            l = dff[dff['Escalation_Risk']=='Low'][col].mean()
            col_max = max(h, l, 0.01)
            high_vals.append(h / col_max)
            low_vals.append(l / col_max)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=high_vals + [high_vals[0]], theta=profile_labels + [profile_labels[0]],
                                       fill='toself', name='High Risk', line=dict(color='#f85149'),
                                       fillcolor='rgba(248,81,73,0.12)'))
        fig.add_trace(go.Scatterpolar(r=low_vals + [low_vals[0]], theta=profile_labels + [profile_labels[0]],
                                       fill='toself', name='Low Risk', line=dict(color='#00c896'),
                                       fillcolor='rgba(0,200,150,0.12)'))
        fig.update_layout(title='Risk Profile Radar',
                          polar=dict(radialaxis=dict(range=[0,1.1], gridcolor='rgba(201,209,217,0.08)'),
                                     angularaxis=dict(gridcolor='rgba(201,209,217,0.08)'),
                                     bgcolor='rgba(0,0,0,0)'),
                          legend=dict(orientation='h', y=-0.1))
        st.plotly_chart(styled_chart(fig, 370), use_container_width=True)

    with r2:
        # Compact stat panel — gap analysis
        gap_data = []
        for col, label in zip(profile_cols, profile_labels):
            h_mean = dff[dff['Escalation_Risk']=='High'][col].mean()
            l_mean = dff[dff['Escalation_Risk']=='Low'][col].mean()
            t_stat, p_val = stats.ttest_ind(
                dff[dff['Escalation_Risk']=='High'][col].dropna(),
                dff[dff['Escalation_Risk']=='Low'][col].dropna()
            )
            gap_data.append({'Factor': label, 'High Risk': round(h_mean,2), 'Low Risk': round(l_mean,2),
                             'Gap': round(abs(h_mean - l_mean),2), 'p-value': round(p_val,4)})
        gap_df = pd.DataFrame(gap_data)

        st.markdown("<div class='panel'><div class='panel-title'>Gap Analysis <span class='pill pill-new'>T-TEST</span></div>", unsafe_allow_html=True)
        for _, row in gap_df.iterrows():
            sig = '**' if row['p-value'] < 0.01 else '*' if row['p-value'] < 0.05 else ''
            st.markdown(f"""<div class='stat-row'>
                <span class='label'>{row['Factor']}</span>
                <span class='value'>Gap: {row['Gap']:.1f} {sig}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div class='callout'>
            <strong>Low political stability</strong> and <strong>low democracy index</strong> are the strongest
            differentiators between high and low risk importers.
        </div>""", unsafe_allow_html=True)

    with r3:
        numeric_cols = ['Deal_Value_USD_M','Quantity','Delivery_Timeline_Months',
                        'Importer_GDP_Per_Capita','Importer_Political_Stability',
                        'Importer_Democracy_Index','Importer_Military_Spend_Pct_GDP',
                        'Offensive_Flag','High_Risk_Flag']
        corr_matrix = dff[numeric_cols].corr()
        risk_corr = corr_matrix['High_Risk_Flag'].drop('High_Risk_Flag').sort_values()

        fig = go.Figure(go.Bar(
            y=risk_corr.index, x=risk_corr.values, orientation='h',
            marker=dict(color=risk_corr.values,
                        colorscale=[[0,'#00c896'],[0.5,'#484f58'],[1,'#f85149']], cmid=0),
            text=[f'{v:.3f}' for v in risk_corr.values], textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Correlation with Escalation Risk')
        st.plotly_chart(styled_chart(fig, 370), use_container_width=True)

    # ── Row 3: Chi-Square — bar left, table right ──
    st.markdown("<div class='sec-div'><h3>Statistical Significance</h3><p>Chi-Square tests for categorical factor association with escalation risk</p></div>", unsafe_allow_html=True)

    cat_test_cols = ['Weapon_Category','Weapon_Class','Deal_Framework','Exporter_Alliance',
                     'Importer_Conflict_Proximity','Active_Territorial_Dispute',
                     'Natural_Resource_Dependence','Arms_Import_Trend','UN_Embargo',
                     'Technology_Transfer','UNSC_Permanent_Member','Importer_Region']
    chi2_results = []
    for col in cat_test_cols:
        if col in dff.columns:
            ct = pd.crosstab(dff[col], dff['Escalation_Risk'])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape)-1)))
                chi2_results.append({'Feature': col, 'Chi2': round(chi2,2), 'p-value': round(p,5),
                                     "Cramers_V": round(cramers_v, 3),
                                     'Significant': 'Yes' if p < 0.05 else 'No'})
    chi_df = pd.DataFrame(chi2_results).sort_values("Cramers_V", ascending=False)

    ch1, ch2 = st.columns([3, 2])
    with ch1:
        fig = go.Figure(go.Bar(
            x=chi_df["Cramers_V"], y=chi_df['Feature'], orientation='h',
            marker=dict(color=chi_df["Cramers_V"], colorscale=[[0,'#0ea5e9'],[1,'#f85149']]),
            text=chi_df["Cramers_V"], textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title="Cramer's V — Effect Size")
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with ch2:
        st.dataframe(chi_df.set_index('Feature'), use_container_width=True, height=400)

    # Chi-square insight
    sig_features = chi_df[chi_df['Significant']=='Yes']
    top_assoc = chi_df.iloc[0] if len(chi_df) > 0 else None
    n_sig = len(sig_features)
    st.markdown(f"""<div class='callout'>
        <strong>Statistical Finding:</strong> {n_sig} of {len(chi_df)} categorical features show statistically significant
        association with escalation risk (p < 0.05).
        {f"<strong>{top_assoc['Feature']}</strong> has the strongest effect size (Cramer's V = {top_assoc['Cramers_V']:.3f}), meaning it provides the most discriminatory power for predicting risk category." if top_assoc is not None else ""}
        {f"Features with Cramer's V > 0.2 ({', '.join(sig_features[sig_features['Cramers_V'] > 0.2]['Feature'].tolist()) or 'none'}) represent strong associations that could form the basis of rule-based screening criteria." if len(sig_features) > 0 else ""}
    </div>""", unsafe_allow_html=True)

    # ── Row 4: Full-width risk factor combinations ──
    st.markdown("<div class='sec-div'><h3>Risk Factor Combinations</h3><p>Multi-factor profiles with highest escalation rates</p></div>", unsafe_allow_html=True)

    risk_combos = []
    for conflict in ['Yes','No']:
        for dispute in ['Yes','No']:
            for wclass in ['Offensive','Defensive']:
                for trend in ['Accelerating','Stable','Declining']:
                    subset = dff[(dff['Importer_Conflict_Proximity']==conflict) &
                                 (dff['Active_Territorial_Dispute']==dispute) &
                                 (dff['Weapon_Class']==wclass) &
                                 (dff['Arms_Import_Trend']==trend)]
                    if len(subset) >= 10:
                        rate = subset['High_Risk_Flag'].mean() * 100
                        risk_combos.append({
                            'Conflict': conflict, 'Dispute': dispute,
                            'Class': wclass, 'Trend': trend,
                            'Count': len(subset), 'High Risk %': round(rate,1)
                        })

    risk_cdf = pd.DataFrame(risk_combos).sort_values('High Risk %', ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=risk_cdf['High Risk %'],
        y=risk_cdf.apply(lambda r: f"{r['Conflict']}/{r['Dispute']}/{r['Class']}/{r['Trend']}", axis=1),
        orientation='h',
        marker=dict(color=risk_cdf['High Risk %'], colorscale=[[0,'#d29922'],[1,'#f85149']]),
        text=risk_cdf.apply(lambda r: f"{r['High Risk %']}% (n={r['Count']})", axis=1),
        textposition='outside', textfont=dict(size=9)
    ))
    fig.update_layout(title='Conflict / Dispute / Class / Trend → Escalation Rate',
                      xaxis_title='High Risk %', yaxis_title='')
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    # Risk combo insight
    if len(risk_cdf) > 0:
        deadliest = risk_cdf.iloc[0]
        deadliest_rate = deadliest['High Risk %']
        deadliest_label = f"Conflict:{deadliest['Conflict']} / Dispute:{deadliest['Dispute']} / {deadliest['Class']} / {deadliest['Trend']}"
        avg_base_rate = high_risk_pct
        risk_multiplier = (deadliest_rate / avg_base_rate) if avg_base_rate > 0 else 0

        st.markdown(f"""<div class='panel'>
            <div class='panel-title'>Diagnostic Summary <span class='pill pill-critical'>KEY FINDING</span></div>
            <div style='font-size:0.78rem; color:#c9d1d9; line-height:1.6;'>
                The deadliest risk profile is <strong style='color:#f85149;'>{deadliest_label}</strong> at
                <strong style='color:#f85149;'>{deadliest_rate:.0f}%</strong> high-risk rate — that's
                <strong style='color:#f0f6fc;'>{risk_multiplier:.1f}x</strong> the baseline rate of {avg_base_rate:.0f}%.
                This means arms transfers matching this profile are {risk_multiplier:.1f} times more likely to be classified
                as high escalation risk. Policy recommendation: transfers matching the top 3 risk profiles should trigger
                mandatory enhanced due diligence before approval.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Embargo (compact) ──
    embargo_df = dff[dff['UN_Embargo']=='Yes']
    if len(embargo_df) > 0:
        st.markdown("<div class='sec-div'><h3>Embargo Circumvention</h3></div>", unsafe_allow_html=True)
        emb_by_exp = embargo_df.groupby('Exporter').agg(Deals=('Year','count'), Value=('Deal_Value_USD_M','sum')).reset_index()
        emb_by_exp = emb_by_exp.sort_values('Value', ascending=True)
        fig = go.Figure(go.Bar(
            y=emb_by_exp['Exporter'], x=emb_by_exp['Value'], orientation='h',
            marker_color='#f85149',
            text=emb_by_exp.apply(lambda r: f"${r['Value']:,.0f}M ({r['Deals']})", axis=1),
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title='Arms to Embargoed Destinations by Exporter')
        st.plotly_chart(styled_chart(fig, 280), use_container_width=True)


# =============================================================
# TAB 3: PREDICTIVE
# =============================================================
with tab3:
    st.markdown("<div class='sec-div'><h3>Escalation Risk Classification</h3><p>ML models predicting High vs Non-High escalation risk</p></div>", unsafe_allow_html=True)

    @st.cache_data
    def run_predictive_models(data):
        df_ml = data.copy()
        cat_features = ['Exporter_Alliance','Weapon_Category','Weapon_Class','Deal_Framework',
                        'Importer_Conflict_Proximity','Active_Territorial_Dispute',
                        'Natural_Resource_Dependence','Arms_Import_Trend','UN_Embargo',
                        'Technology_Transfer','UNSC_Permanent_Member','Importer_Region']
        for c in cat_features:
            le = LabelEncoder()
            df_ml[c+'_enc'] = le.fit_transform(df_ml[c])
        feature_cols = ['Deal_Value_USD_M','Quantity','Delivery_Timeline_Months',
                        'Importer_GDP_Per_Capita','Importer_Political_Stability',
                        'Importer_Democracy_Index','Importer_Military_Spend_Pct_GDP'] + \
                       [c+'_enc' for c in cat_features]
        X = df_ml[feature_cols]; y = df_ml['High_Risk_Flag']
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
        }
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            model.fit(X_scaled, y)
            importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
            results[name] = {'auc_mean': scores.mean(), 'auc_std': scores.std(),
                             'importance': pd.Series(importance, index=feature_cols).sort_values(ascending=False)}
        roc_data = {}
        for name, model in models.items():
            y_prob = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:,1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        return results, roc_data, feature_cols

    results, roc_data, feature_cols = run_predictive_models(df)

    # ── Model cards (3 cols) instead of a bar chart ──
    model_colors = {'Logistic Regression': '#00c896', 'Random Forest': '#0ea5e9', 'Gradient Boosting': '#d29922'}
    mc1, mc2, mc3 = st.columns(3)
    for col_widget, (name, res) in zip([mc1, mc2, mc3], results.items()):
        color = model_colors[name]
        with col_widget:
            st.markdown(f"""
            <div class='model-card'>
                <div class='model-name'>{name}</div>
                <div class='model-auc' style='color:{color};'>{res['auc_mean']:.3f}</div>
                <div class='model-std'>AUC &plusmn; {res['auc_std']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── ROC curves (full width) ──
    fig = go.Figure()
    for name, rdata in roc_data.items():
        fig.add_trace(go.Scatter(x=rdata['fpr'], y=rdata['tpr'], mode='lines',
                                 name=f"{name} ({rdata['auc']:.3f})",
                                 line=dict(color=model_colors[name], width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline',
                             line=dict(color='#30363d', dash='dash', width=1)))
    fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    # ── Feature importance: selected model | consensus side by side ──
    st.markdown("<div class='sec-div'><h3>Feature Importance</h3></div>", unsafe_allow_html=True)

    f1, f2 = st.columns([1, 1])
    with f1:
        selected_model = st.selectbox("Model:", list(results.keys()), index=1)
        imp = results[selected_model]['importance'].head(12)
        fig = go.Figure(go.Bar(
            y=imp.index[::-1], x=imp.values[::-1], orientation='h',
            marker=dict(color=imp.values[::-1], colorscale=[[0,'#0c4a3e'],[1,'#f85149']]),
            text=[f'{v:.4f}' for v in imp.values[::-1]], textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(title=f'{selected_model} — Top 12 Features')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    with f2:
        all_imp = pd.DataFrame()
        for name, res in results.items():
            all_imp[name] = res['importance'] / res['importance'].max()
        all_imp['Mean'] = all_imp.mean(axis=1)
        all_imp = all_imp.sort_values('Mean', ascending=False).head(12)

        fig = go.Figure()
        for name in results.keys():
            fig.add_trace(go.Bar(name=name, y=all_imp.index[::-1], x=all_imp[name].values[::-1],
                                 orientation='h', marker_color=model_colors[name], opacity=0.75))
        fig.update_layout(title='Consensus Ranking (All Models)', barmode='group', xaxis_title='Normalized Importance')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    st.markdown("""<div class='callout'>
        <strong>Political stability, democracy index, and conflict proximity</strong> consistently emerge as the
        strongest predictors across all three models. Weapon class and arms import trend are secondary signals.
    </div>""", unsafe_allow_html=True)


# =============================================================
# TAB 4: PRESCRIPTIVE
# =============================================================
with tab4:

    # ── Row 1: Risk simulator gauge (left) + Regional bar chart (right) ──
    st.markdown("<div class='sec-div'><h3>Risk Assessment</h3></div>", unsafe_allow_html=True)

    p1, p2 = st.columns([2, 3])

    with p1:
        st.markdown("**Escalation Risk Simulator**")
        sim_stability = st.slider("Political Stability", 1.0, 10.0, 5.0, 0.5, key='s1')
        sim_democracy = st.slider("Democracy Index", 1.0, 10.0, 5.0, 0.5, key='s2')
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_conflict = st.selectbox("Conflict", ['Yes','No'], key='s3')
            sim_weapon = st.selectbox("Weapon Class", ['Offensive','Defensive'], key='s5')
        with sc2:
            sim_dispute = st.selectbox("Dispute", ['Yes','No'], key='s4')
            sim_trend = st.selectbox("Arms Trend", ['Accelerating','Stable','Declining'], key='s6')
        sim_milspend = st.slider("Military Spend % GDP", 0.5, 8.0, 2.5, 0.5, key='s7')
        sim_resource = st.selectbox("Resource Dependence", ['High','Medium','Low'], key='s8')

        risk_score = 0
        risk_score += (10 - sim_stability) * 3.0
        risk_score += (10 - sim_democracy) * 1.5
        if sim_conflict == 'Yes': risk_score += 12
        if sim_dispute == 'Yes': risk_score += 8
        if sim_weapon == 'Offensive': risk_score += 5
        if sim_trend == 'Accelerating': risk_score += 7
        elif sim_trend == 'Declining': risk_score -= 3
        if sim_milspend > 4.0: risk_score += 6
        elif sim_milspend > 2.5: risk_score += 3
        if sim_resource == 'High': risk_score += 4
        elif sim_resource == 'Medium': risk_score += 2
        risk_score = min(100, max(0, risk_score))

        risk_color = '#00c896' if risk_score < 28 else '#d29922' if risk_score < 45 else '#f85149'
        risk_label = 'LOW' if risk_score < 28 else 'ELEVATED' if risk_score < 45 else 'CRITICAL'

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': risk_label, 'font': {'size': 14, 'color': risk_color, 'family': 'Inter'}},
            gauge=dict(
                axis=dict(range=[0,100], tickwidth=1, tickcolor='#30363d'),
                bar=dict(color=risk_color),
                bgcolor='rgba(0,0,0,0)',
                steps=[
                    dict(range=[0,28], color='rgba(0,200,150,0.06)'),
                    dict(range=[28,45], color='rgba(210,153,34,0.06)'),
                    dict(range=[45,100], color='rgba(248,81,73,0.06)'),
                ],
                threshold=dict(line=dict(color='#f85149', width=2), thickness=0.75, value=risk_score)
            ),
            number=dict(suffix='/100', font=dict(size=36, color=risk_color, family='JetBrains Mono'))
        ))
        st.plotly_chart(styled_chart(fig, 250), use_container_width=True)

    with p2:
        # Regional threat bar chart
        region_full = dff.groupby('Importer_Region').agg(
            Total_Deals=('Year','count'), High_Risk_Deals=('High_Risk_Flag','sum'),
            Total_Value=('Deal_Value_USD_M','sum'),
            Avg_Stability=('Importer_Political_Stability','mean'),
            Avg_Democracy=('Importer_Democracy_Index','mean'),
            Offensive_Pct=('Offensive_Flag','mean'),
            Accel_Count=('Arms_Import_Trend', lambda x: (x=='Accelerating').sum())
        ).reset_index()
        region_full['High_Risk_Pct'] = (region_full['High_Risk_Deals'] / region_full['Total_Deals'] * 100).round(1)
        region_full['Offensive_Pct'] = (region_full['Offensive_Pct'] * 100).round(1)
        region_full = region_full.sort_values('High_Risk_Pct', ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=region_full['Importer_Region'], y=region_full['High_Risk_Pct'],
                             name='High Risk %', marker_color='#f85149'))
        fig.add_trace(go.Bar(x=region_full['Importer_Region'], y=region_full['Offensive_Pct'],
                             name='Offensive %', marker_color='rgba(14,165,233,0.5)'))
        fig.update_layout(title='Regional Threat Profile', barmode='group',
                          yaxis_title='%', xaxis_tickangle=-20, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

        st.dataframe(region_full.set_index('Importer_Region').rename(columns={
            'Total_Deals':'Deals', 'High_Risk_Deals':'High Risk', 'Total_Value':'Value ($M)',
            'Avg_Stability':'Stability', 'Avg_Democracy':'Democracy',
            'Offensive_Pct':'Offensive %', 'Accel_Count':'Accelerating', 'High_Risk_Pct':'Risk %'
        }), use_container_width=True, height=200)

    # ── Recommendations in 2-column grid ──
    st.markdown("<div class='sec-div'><h3>Strategic Recommendations</h3><p>Evidence-based policy interventions</p></div>", unsafe_allow_html=True)

    conflict_risk_rate = dff[dff['Importer_Conflict_Proximity']=='Yes']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Conflict_Proximity']=='Yes']) > 0 else 0
    no_conflict_rate = dff[dff['Importer_Conflict_Proximity']=='No']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Conflict_Proximity']=='No']) > 0 else 0
    accel_risk_rate = dff[dff['Arms_Import_Trend']=='Accelerating']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Arms_Import_Trend']=='Accelerating']) > 0 else 0
    offensive_risk_rate = dff[dff['Weapon_Class']=='Offensive']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Weapon_Class']=='Offensive']) > 0 else 0
    low_stab_rate = dff[dff['Importer_Political_Stability'] < 4]['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Political_Stability'] < 4]) > 0 else 0

    recommendations = [
        ("Arms Embargo Enforcement", f"Conflict-proximate importers: {conflict_risk_rate:.0f}% high-risk vs {no_conflict_rate:.0f}% non-conflict. Strengthen multilateral monitoring.", "CRITICAL"),
        ("Arms Acceleration Monitoring", f"Accelerating trends show {accel_risk_rate:.0f}% high-risk. Deploy real-time tracking and flag >20% YoY acceleration.", "CRITICAL"),
        ("Offensive Transfer Controls", f"Offensive systems: {offensive_risk_rate:.0f}% escalation rate. Stricter end-use certificates for combat aircraft, missiles, MLRS.", "HIGH"),
        ("Governance-Linked Licensing", f"Stability <4.0: {low_stab_rate:.0f}% risk. Binding governance thresholds in export frameworks.", "HIGH"),
        ("Diplomatic Corridors", "Territorial disputes are top escalation drivers. Prioritise mediation for top dispute dyads.", "MEDIUM"),
        ("Predictive Peacekeeping", "Use ML models as quarterly early-warning for UN DPPA. Shift to anticipatory posture.", "HIGH"),
    ]

    # Two-column card grid
    for i in range(0, len(recommendations), 2):
        rc1, rc2 = st.columns(2)
        for col_w, j in zip([rc1, rc2], [i, i+1]):
            if j < len(recommendations):
                title, desc, priority = recommendations[j]
                pill_cls = 'pill-critical' if priority == 'CRITICAL' else 'pill-high' if priority == 'HIGH' else 'pill-medium'
                with col_w:
                    st.markdown(f"""<div class='rx-card'>
                        <h4>{title} <span class='pill {pill_cls}'>{priority}</span></h4>
                        <p>{desc}</p>
                    </div>""", unsafe_allow_html=True)

    # ── Impact matrix (full width) ──
    st.markdown("<div class='sec-div'><h3>Impact vs Feasibility</h3></div>", unsafe_allow_html=True)

    impact_data = pd.DataFrame({
        'Intervention': ['Embargo Enforcement', 'Acceleration Monitoring', 'Offensive Controls',
                         'Governance Licensing', 'Diplomatic Corridors', 'Predictive Peacekeeping'],
        'Risk Reduction %': [8.5, 6.2, 5.0, 4.5, 3.8, 7.0],
        'Complexity': [4, 2, 3, 4, 3, 2],
        'Time (months)': [12, 4, 8, 18, 24, 6]
    })

    fig = px.scatter(impact_data, x='Complexity', y='Risk Reduction %',
                     size='Time (months)', text='Intervention',
                     color='Risk Reduction %',
                     color_continuous_scale=[[0,'#d29922'],[1,'#00c896']],
                     title='Intervention Prioritisation (size = time to impact)')
    fig.update_traces(textposition='top center', textfont=dict(size=10, family='Inter'))
    fig.update_layout(xaxis_title='Complexity (1=Easy, 5=Hard)', yaxis_title='Risk Reduction %')
    st.plotly_chart(styled_chart(fig, 380), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#484f58; font-size:0.68rem; padding:0.5rem; font-family:Inter, sans-serif;'>
    <span style='color:#00c896; font-weight:600;'>AEGIS</span> &mdash; Arms & Escalation Geopolitical Intelligence System
    &bull; 1,500 synthetic transfers &times; 25 features
    &bull; Descriptive &middot; Diagnostic &middot; Predictive &middot; Prescriptive
</div>
""", unsafe_allow_html=True)
