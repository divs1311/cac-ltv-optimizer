"""
Phase 4: Streamlit Dashboard — AI-Powered CAC vs LTV Optimizer
Fintech Growth Analytics Platform
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import load_and_compute, channel_summary
from llm_engine import get_ai_recommendations_sync, _build_flags

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAC·LTV Optimizer | Fintech Growth",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.flag-critical { color: #f87171; font-weight: 600; }
.flag-warning  { color: #fbbf24; font-weight: 600; }
.flag-ok       { color: #34d399; font-weight: 600; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 4px 0 16px 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 20px;
}
.rec-card {
    background: #0f172a;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.rec-action { font-weight: 700; color: #38bdf8; font-size: 0.85rem; }
.tag-scale    { background:#064e3b; color:#6ee7b7; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
.tag-sunset   { background:#450a0a; color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
.tag-optimize { background:#1c1917; color:#fcd34d; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
.tag-reallocate { background:#1e1b4b; color:#a5b4fc; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "saas_cac_ltv_data.csv")

@st.cache_data
def get_data():
    df, cohort = load_and_compute(DATA_PATH)
    ch = channel_summary(cohort)
    return df, cohort, ch

df, cohort, ch = get_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 GenAI Growth Consultant")
    st.markdown(
        "<span style='color:#64748b;font-size:0.8rem'>Powered by Claude · Anthropic</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    budget = st.slider(
        "Monthly Acquisition Budget ($)",
        min_value=100_000, max_value=2_000_000,
        value=500_000, step=50_000,
        format="$%d",
    )

    seg_filter = st.multiselect(
        "Segment Filter",
        options=["SMB", "Enterprise"],
        default=["SMB", "Enterprise"],
    )

    ch_filter = st.multiselect(
        "Channel Filter",
        options=df["Acquisition_Channel"].unique().tolist(),
        default=df["Acquisition_Channel"].unique().tolist(),
    )

    st.divider()
    run_ai = st.button("⚡ Run AI Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem;color:#475569'>
    <b style='color:#94a3b8'>Unit Economics Framework</b><br><br>
    <b>LTV</b> = ARPU ÷ Churn Rate<br>
    <b>Payback</b> = CAC ÷ Monthly Revenue<br>
    <b>Health</b>: CAC:LTV ≤ 0.20 🟢<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.20–0.33 🟡<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&gt;0.33 🔴<br><br>
    <b>SaaS Benchmarks</b><br>
    Payback &lt; 12 months ✓<br>
    LTV:CAC &gt; 3× ✓<br>
    Blended Churn &lt; 5% ✓
    </div>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
df_f     = df[df["Segment"].isin(seg_filter) & df["Acquisition_Channel"].isin(ch_filter)]
cohort_f = cohort[cohort["Segment"].isin(seg_filter) & cohort["Acquisition_Channel"].isin(ch_filter)]
ch_f     = channel_summary(cohort_f)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:"IBM Plex Mono",monospace;font-size:1.6rem;font-weight:600;
color:#f1f5f9;margin-bottom:2px'>CAC·LTV Optimizer</h1>
<p style='color:#64748b;font-size:0.85rem;margin-top:0'>
B2B SaaS · Fintech Growth Intelligence Platform · 10,000 Customer Cohort
</p>
""", unsafe_allow_html=True)

st.divider()

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
blended_cac = df_f["CAC"].mean()
blended_ltv = df_f["LTV"].mean()
blended_payback = df_f["Payback_Period"].mean()
blended_roi = df_f["ROI_Score"].mean()
high_cac_pct = df_f["High_CAC_Flag"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, f"${blended_cac:,.0f}",     "Blended CAC"),
    (c2, f"${blended_ltv:,.0f}",     "Blended LTV"),
    (c3, f"{blended_payback:.1f} mo", "Avg Payback Period"),
    (c4, f"{blended_roi:.1f}×",       "Blended ROI Score"),
    (c5, f"{high_cac_pct:.0%}",       "High-CAC Cohort %"),
]
for col, val, lbl in kpis:
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{val}</div>'
        f'<div class="metric-label">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── ROW 1: Scatter + Bar ──────────────────────────────────────────────────────
col_a, col_b = st.columns([1.4, 1])

with col_a:
    st.markdown('<p class="section-header">CAC vs LTV Scatter — Channel × Segment</p>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        cohort_f,
        x="Avg_CAC", y="Avg_LTV",
        color="Acquisition_Channel",
        symbol="Segment",
        size="Count",
        hover_data=["CAC_LTV_Ratio", "Avg_Payback_Mo", "ROI_Score", "Health_Grade"],
        color_discrete_map={
            "Paid Social": "#f87171",
            "Organic SEO": "#38bdf8",
            "Email":       "#fbbf24",
            "Referral":    "#34d399",
        },
        labels={"Avg_CAC": "Avg CAC ($)", "Avg_LTV": "Avg LTV ($)"},
        template="plotly_dark",
    )
    # Add threshold line: CAC = 0.33 * LTV  →  LTV = 3× CAC
    cac_line = [0, cohort_f["Avg_CAC"].max() * 1.2]
    ltv_line = [x * 3 for x in cac_line]
    fig_scatter.add_trace(go.Scatter(
        x=cac_line, y=ltv_line,
        mode="lines", name="3× CAC Threshold",
        line=dict(color="#64748b", dash="dash", width=1.5),
    ))
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#94a3b8"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=10, b=0), height=340,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    st.markdown('<p class="section-header">ROI Score by Channel</p>', unsafe_allow_html=True)
    bar_df = ch_f.sort_values("ROI_Score")
    colors = ["#f87171" if r < 5 else "#38bdf8" for r in bar_df["ROI_Score"]]
    fig_bar = go.Figure(go.Bar(
        x=bar_df["ROI_Score"],
        y=bar_df["Acquisition_Channel"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}×" for v in bar_df["ROI_Score"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#94a3b8"), xaxis_title="ROI Score (×)",
        margin=dict(l=0, r=30, t=10, b=0), height=340,
        yaxis=dict(gridcolor="#1e293b"), xaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── ROW 2: Payback heatmap + CAC:LTV gauge ────────────────────────────────────
col_c, col_d = st.columns([1, 1])

with col_c:
    st.markdown('<p class="section-header">Payback Period Heatmap (Months) · Channel × Segment</p>', unsafe_allow_html=True)
    pivot = cohort_f.pivot_table(values="Avg_Payback_Mo", index="Acquisition_Channel", columns="Segment")
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=["#0f4c2a", "#fbbf24", "#7f1d1d"],
        text_auto=".1f",
        template="plotly_dark",
        labels=dict(color="Payback (mo)"),
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(color="#94a3b8"), margin=dict(l=0, r=0, t=10, b=0), height=280,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_d:
    st.markdown('<p class="section-header">CAC:LTV Ratio by Channel</p>', unsafe_allow_html=True)
    fig_funnel = go.Figure()
    for _, row in ch_f.iterrows():
        color = "#f87171" if row["CAC_LTV_Ratio"] > 0.33 else ("#fbbf24" if row["CAC_LTV_Ratio"] > 0.20 else "#34d399")
        fig_funnel.add_trace(go.Indicator(
            mode="gauge+number",
            value=row["CAC_LTV_Ratio"],
            domain={"row": 0, "column": list(ch_f["Acquisition_Channel"]).index(row["Acquisition_Channel"])},
            title={"text": row["Acquisition_Channel"][:10], "font": {"size": 11, "color": "#94a3b8"}},
            number={"font": {"color": color, "size": 18}},
            gauge={
                "axis": {"range": [0, 0.8], "tickcolor": "#475569"},
                "bar": {"color": color},
                "bgcolor": "#0f172a",
                "bordercolor": "#334155",
                "threshold": {"line": {"color": "#f87171", "width": 2}, "thickness": 0.8, "value": 0.33},
            },
        ))
    fig_funnel.update_layout(
        grid={"rows": 1, "columns": len(ch_f), "pattern": "independent"},
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

# ── ROW 3: Cohort table ───────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">Cohort Unit Economics Table</p>', unsafe_allow_html=True)

display_cols = ["Acquisition_Channel", "Segment", "Count", "Avg_CAC", "Avg_LTV",
                "Avg_ARPU", "Avg_Churn_Month", "Avg_Payback_Mo", "CAC_LTV_Ratio",
                "ROI_Score", "High_CAC_Pct", "Health_Grade"]
styled = cohort_f[display_cols].rename(columns={
    "Avg_CAC": "CAC ($)", "Avg_LTV": "LTV ($)", "Avg_ARPU": "ARPU ($)",
    "Avg_Churn_Month": "Churn Mo", "Avg_Payback_Mo": "Payback Mo",
    "CAC_LTV_Ratio": "CAC:LTV", "High_CAC_Pct": "High-CAC %",
    "ROI_Score": "ROI ×",
})

def color_grade(val):
    if "🔴" in str(val): return "color: #f87171"
    if "🟡" in str(val): return "color: #fbbf24"
    return "color: #34d399"

st.dataframe(
    styled.style.map(color_grade, subset=["Health_Grade"])
          .format({
              "CAC ($)": "${:.0f}", "LTV ($)": "${:.0f}", "ARPU ($)": "${:.0f}",
              "Churn Mo": "{:.1f}", "Payback Mo": "{:.2f}",
              "CAC:LTV": "{:.3f}", "ROI ×": "{:.1f}", "High-CAC %": "{:.0%}",
          }),
    use_container_width=True, height=320,
)

# ── AI RECOMMENDATIONS PANEL ──────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">⚡ AI-Powered Marketing Mix Reallocation</p>', unsafe_allow_html=True)

if run_ai:
    with st.spinner("🧠 Claude is analyzing your unit economics..."):
        try:
            ch_dict     = ch_f.to_dict(orient="records")
            cohort_dict = cohort_f[["Acquisition_Channel","Segment","Avg_CAC","Avg_LTV",
                                    "Avg_Payback_Mo","CAC_LTV_Ratio","ROI_Score","High_CAC_Pct"]].to_dict(orient="records")
            recs = get_ai_recommendations_sync(ch_dict, cohort_dict, budget)
            st.session_state["ai_recs"] = recs
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            recs = None
else:
    recs = st.session_state.get("ai_recs")

if recs:
    # Executive summary
    st.info(f"**Executive Summary:** {recs.get('executive_summary','')}")

    # Critical flags
    flags = recs.get("critical_flags", [])
    if flags:
        st.markdown("**🚨 Critical Flags:**")
        for f in flags:
            st.markdown(f"- {f}")

    st.markdown("---")

    # Channel recommendations
    ch_recs = recs.get("channel_recommendations", [])
    if ch_recs:
        st.markdown("**📊 Channel-Level Recommendations:**")
        cols = st.columns(len(ch_recs))
        tag_map = {"SCALE": "tag-scale", "SUNSET": "tag-sunset",
                   "OPTIMIZE": "tag-optimize", "REALLOCATE": "tag-reallocate"}
        for i, rec in enumerate(ch_recs):
            action = rec.get("action","OPTIMIZE")
            tag_cls = tag_map.get(action, "tag-optimize")
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="rec-card">
                <span class="{tag_cls}">{action}</span><br>
                <b style='color:#f1f5f9;font-size:1rem'>{rec.get('channel','')}</b><br>
                <span style='color:#64748b;font-size:0.78rem'>Budget shift: 
                  <b style='color:#38bdf8'>{rec.get('budget_shift_pct',0):+d}%</b></span><br>
                <span style='color:#94a3b8;font-size:0.8rem'>{rec.get('rationale','')}</span><br><br>
                <span style='color:#64748b;font-size:0.75rem'>⏱ Payback compress: 
                  <b style='color:#34d399'>{rec.get('expected_payback_compression_months',0):.1f} mo</b></span><br>
                <span style='color:#64748b;font-size:0.75rem'>📈 LTV play: 
                  <b style='color:#a5b4fc'>{rec.get('ltv_expansion_tactic','')}</b></span>
                </div>
                """, unsafe_allow_html=True)

    col_ltv, col_wins = st.columns(2)
    with col_ltv:
        plays = recs.get("ltv_expansion_plays", [])
        if plays:
            st.markdown("**📈 LTV Expansion Plays:**")
            for p in plays:
                st.markdown(f"- {p}")

    with col_wins:
        wins = recs.get("30_day_quick_wins", [])
        moves = recs.get("90_day_strategic_moves", [])
        if wins:
            st.markdown("**⚡ 30-Day Quick Wins:**")
            for w in wins:
                st.markdown(f"- {w}")
        if moves:
            st.markdown("**🎯 90-Day Strategic Moves:**")
            for m in moves:
                st.markdown(f"- {m}")

    lift = recs.get("projected_blended_roi_lift_pct", 0)
    if lift:
        st.success(f"📊 **Projected Blended ROI Lift: +{lift}%** (post-reallocation estimate)")

else:
    st.markdown("""
    <div style='text-align:center;padding:40px;color:#475569'>
    <div style='font-size:2rem'>🤖</div>
    <b style='color:#64748b'>GenAI Growth Consultant</b><br>
    <span style='font-size:0.85rem'>Click <b>⚡ Run AI Analysis</b> in the sidebar to generate<br>
    Marketing Mix Reallocation Strategies powered by Claude.</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='color:#334155;font-size:0.72rem;text-align:center'>
CAC·LTV Optimizer · Fintech Growth Intelligence · Built with Streamlit + Claude API · 
Unit Economics Framework aligned to Skydo / Cross-border B2B SaaS
</p>
""", unsafe_allow_html=True)
