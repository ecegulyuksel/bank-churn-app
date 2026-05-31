"""
Bank Customer Churn Risk Manager — v2 (corrected)
Production-ready Streamlit app for bank churn risk intelligence.

NOTE: Best model is XGBoost + CTGAN (highest Recall & F1), consistent with the
project report. Optuna improved general accuracy slightly but traded off Recall,
so CTGAN is the final / primary model used for scoring throughout the app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Manager",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2 { font-family: 'Playfair Display', serif; color: #0A1929; }
h3, h4 { font-family: 'Inter', sans-serif; font-weight: 600; color: #0A1929; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1929 0%, #132F4C 100%);
}
[data-testid="stSidebar"] * { color: #E8F4FD !important; }

/* Buttons */
.stButton > button {
    background: #0A1929; color: #F5F5F0; border: none;
    border-radius: 6px; padding: 0.6rem 1.5rem;
    font-weight: 600; font-size: 0.85rem; letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #D4AF37; color: #0A1929; transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(212,175,55,0.4);
}

/* KPI cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important; font-weight: 700 !important;
    color: #0A1929 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important; text-transform: uppercase;
    letter-spacing: 1px; color: #6B7280 !important;
    font-weight: 600 !important;
}

/* Risk badges */
.badge-low    { background:#D1FAE5; color:#065F46; border-left:4px solid #10B981; padding:10px 18px; border-radius:6px; font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-medium { background:#FEF3C7; color:#92400E; border-left:4px solid #F59E0B; padding:10px 18px; border-radius:6px; font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-high   { background:#FEE2E2; color:#991B1B; border-left:4px solid #DC2626; padding:10px 18px; border-radius:6px; font-weight:700; font-size:1.1rem; display:inline-block; }

/* Action box */
.action-box { background:#EFF6FF; border:2px solid #3B82F6; border-radius:10px; padding:20px 24px; margin-top:12px; }
.action-box-urgent { background:#FFF7ED; border:2px solid #F97316; border-radius:10px; padding:20px 24px; margin-top:12px; }

/* Info callout */
.callout { background:#F0F9FF; border-left:4px solid #0EA5E9; padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; font-size:0.92rem; color:#0C4A6E; }
.callout-gold { background:#FFFBEB; border-left:4px solid #D4AF37; padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; font-size:0.92rem; color:#78350F; }

/* Section divider */
.divider { height:2px; background:linear-gradient(90deg,#D4AF37,transparent); margin:20px 0 28px 0; }

/* Hide streamlit chrome */
#MainMenu,footer,header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all():
    xgb   = joblib.load('xgb_model.pkl')
    opt   = joblib.load('xgb_optuna.pkl')
    sc    = joblib.load('scaler.pkl')
    art   = joblib.load('artifacts.pkl')
    beta  = np.load('lr_beta.npy')
    expl  = shap.TreeExplainer(xgb)
    return xgb, opt, sc, art, beta, expl

try:
    xgb_model, xgb_optuna, scaler, artifacts, lr_beta, explainer = load_all()
    READY = True
except Exception as e:
    READY = False; ERR = str(e)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
NUMERIC_COLS = artifacts['sayisal_sutunlar'] if READY else []

def scale_val(raw, col):
    idx = NUMERIC_COLS.index(col)
    return (raw - artifacts['scaler_mean'][idx]) / artifacts['scaler_scale'][idx]

def build_row(inputs: dict) -> pd.DataFrame:
    fn  = artifacts['feature_names']
    row = pd.DataFrame([[artifacts['feature_means'][c] for c in fn]], columns=fn)
    for col, val in inputs.items():
        if col in NUMERIC_COLS:
            row[col] = scale_val(val, col)
        elif col in fn:
            row[col] = val
    return row

def predict_trio(row):
    arr = np.concatenate([[1.0], row.astype(float).values[0]])
    lr  = float(1 / (1 + np.exp(-lr_beta.dot(arr))))
    xg  = float(xgb_model.predict_proba(row)[0][1])
    op  = float(xgb_optuna.predict_proba(row)[0][1])
    return lr, xg, op

def badge(p):
    if p < .30: return f'<span class="badge-low">✅ LOW RISK — {p*100:.1f}%</span>'
    if p < .60: return f'<span class="badge-medium">⚠️ MEDIUM RISK — {p*100:.1f}%</span>'
    return          f'<span class="badge-high">🚨 HIGH RISK — {p*100:.1f}%</span>'

def rlabel(p):
    return "Low" if p < .30 else ("Medium" if p < .60 else "High")

def gauge(prob, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        number={'suffix': '%', 'font': {'size': 36, 'color': '#0A1929'}},
        title={'text': title, 'font': {'size': 13, 'color': '#6B7280'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#9CA3AF'},
            'bar': {'color': '#DC2626' if prob >= .6 else ('#F59E0B' if prob >= .3 else '#10B981')},
            'steps': [
                {'range': [0,  30], 'color': '#D1FAE5'},
                {'range': [30, 60], 'color': '#FEF3C7'},
                {'range': [60, 100], 'color': '#FEE2E2'},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor='white', font_family='Inter')
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Churn Risk Manager")
    st.markdown("<div style='color:#D4AF37;font-size:.75rem;letter-spacing:2px;'>RISK INTELLIGENCE PANEL</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "📊 Overview",
        "👤 Single Customer",
        "📁 Batch Prediction",
        "🎯 Risk Segmentation",
        "💡 Action Guide",
        "ℹ️ About",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Active models**")
    st.markdown("— Logistic Regression\n— XGBoost + CTGAN ⭐\n— XGBoost + Optuna")
    if READY:
        st.success("Models loaded", icon="✅")
    else:
        st.error("Model load failed")

if not READY:
    st.error(f"Could not load model files: {ERR}")
    st.stop()


# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Bank Customer Churn Risk Manager")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-gold">
    <b>What is this tool?</b><br>
    This application uses machine learning to identify which bank customers are at risk
    of leaving (churning). By knowing who is likely to leave <i>before they actually do</i>,
    your retention team can take targeted action — saving relationships and revenue.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How to use this tool")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        **① Single Customer**
        Enter a customer's profile and instantly see their churn risk with a plain-language explanation.
        """)
    with c2:
        st.markdown("""
        **② Batch Prediction**
        Upload a CSV of customers and download a scored file with risk levels for your whole portfolio.
        """)
    with c3:
        st.markdown("""
        **③ Risk Segmentation**
        See your portfolio split into Low / Medium / High risk groups with calibration data.
        """)
    with c4:
        st.markdown("""
        **④ Action Guide**
        Get concrete retention actions (call, campaign, cross-sell) matched to each risk driver.
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Portfolio snapshot")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers",  "10,127")
    m2.metric("Historical Churn", "16.1%", help="Percentage of customers who left in the dataset")
    m3.metric("Best Model AUC",   "0.991", help="Area under ROC curve — 1.0 = perfect, 0.5 = random")
    m4.metric("Best Recall",      "88%",   help="Of customers who actually churn, the best model (XGBoost + CTGAN) correctly flags 88%")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Model performance comparison")
        st.markdown("""
        <div class="callout">
        Three models were built. The radar chart shows how each performs across five metrics.
        <b>XGBoost + CTGAN (gold)</b> gives the best balance of <b>Recall</b> and <b>F1</b> —
        the metrics that matter most for catching churning customers early. Optuna optimisation
        slightly raised general accuracy but traded off Recall, so the CTGAN model was kept
        as the final model.
        </div>
        """, unsafe_allow_html=True)

        df_c = pd.DataFrame({
            'Model'     : ['Logistic Regression', 'XGBoost + CTGAN', 'XGBoost + Optuna'],
            'Accuracy'  : [0.90, 0.97, 0.97],
            'Precision' : [0.78, 0.92, 0.94],
            'Recall'    : [0.54, 0.88, 0.83],
            'F1-Score'  : [0.64, 0.90, 0.89],
            'AUC'       : [0.917, 0.991, 0.989],
        })
        metrics = ['Accuracy','Precision','Recall','F1-Score','AUC']
        # CTGAN (best) = gold accent, Optuna = green, LR = blue
        colors  = ['#4472C4','#D4AF37','#2ECC71']
        fig = go.Figure()
        for i, m in enumerate(df_c['Model']):
            fig.add_trace(go.Scatterpolar(
                r=df_c[metrics].iloc[i].values, theta=metrics,
                fill='toself', name=m,
                line=dict(color=colors[i], width=2), opacity=0.75,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.5,1]), bgcolor='#F9FAFB'),
            paper_bgcolor='white', height=380,
            margin=dict(l=40,r=40,t=20,b=20),
            legend=dict(orientation='h', yanchor='bottom', y=-0.18),
            font=dict(family='Inter', color='#0A1929'),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_r:
        st.markdown("### Dataset class distribution")
        st.markdown("""
        <div class="callout">
        Only 16% of customers actually churn. This imbalance is why a naive model
        would just predict "stays" every time and still appear 84% accurate —
        which is useless in practice. We solved this with <b>CTGAN</b>.
        </div>
        """, unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=['Retained (84%)','Churned (16%)'],
            values=[8500, 1627], hole=0.55,
            marker=dict(colors=['#0A1929','#D4AF37']),
            textfont=dict(size=14),
        ))
        fig_pie.update_layout(
            paper_bgcolor='white', height=280,
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.08),
            annotations=[dict(text='10,127<br>customers', x=.5, y=.5,
                               font_size=15, showarrow=False)],
            font=dict(family='Inter', color='#0A1929'),
        )
        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Full metrics table")
    st.dataframe(df_c, use_container_width=True, hide_index=True)
    st.caption("Recall = share of actual churners we correctly flagged. F1 = balance of precision and recall. Best model: XGBoost + CTGAN.")


# ══════════════════════════════════════════════════════
# PAGE 2 — SINGLE CUSTOMER
# ══════════════════════════════════════════════════════
elif page == "👤 Single Customer":
    st.title("Single Customer Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout-gold">
    Fill in the customer's profile below and click <b>Run Risk Assessment</b>.
    You'll see a churn probability, a visual risk gauge, an AI explanation of
    the key drivers, and a tailored retention action.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Customer profile")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔄 Activity & Transactions**")
        total_trans_ct  = st.slider("Number of transactions in last 12 months", 10, 139, 60,
            help="How many times the customer used their card or made a transfer. Low activity is a strong churn signal.")
        total_trans_amt = st.slider("Total amount transacted ($)", 510, 18484, 4404, step=100,
            help="Total dollar value of all transactions in the past year.")
        total_rel_count = st.slider("Number of bank products held", 1, 6, 3,
            help="Products include: credit card, savings account, loan, mortgage, etc. More products = stronger relationship.")
        months_inactive = st.slider("Months of inactivity in last 12 months", 0, 6, 1,
            help="How many months had zero account activity. 3+ months is a warning sign.")
        contacts_count  = st.slider("Number of contacts with customer service", 0, 6, 2,
            help="How many times the customer called or messaged support. Very high contact can signal dissatisfaction.")
        total_rev_bal   = st.slider("Total revolving balance ($)", 0, 2517, 1163, step=50,
            help="The outstanding credit card balance carried month-to-month. Higher balance = more engaged with the card.")

    with col2:
        st.markdown("**👤 Customer Demographics**")
        age    = st.slider("Age", 26, 73, 46)
        deps   = st.slider("Number of dependents (family members)", 0, 5, 2)
        months = st.slider("Months as a customer (tenure)", 13, 56, 36,
            help="How long this person has been with the bank.")
        gender = st.selectbox("Gender", ["F", "M"])
        income = st.selectbox("Annual income range", [
            "Less than $40K", "$40K - $60K", "$60K - $80K",
            "$80K - $120K", "$120K +", "Unknown"],
            help="Self-reported income band.")
        card   = st.selectbox("Card category", ["Blue","Silver","Gold","Platinum"],
            help="Blue = standard entry-level card. Silver/Gold/Platinum = premium tiers with higher spending limits and benefits. Note: premium card holders show higher churn risk in our data.")

    st.markdown("###")
    run = st.button("🔮 Run Risk Assessment", use_container_width=True)

    if run:
        inputs = {
            'Customer_Age'             : age,
            'Dependent_count'          : deps,
            'Months_on_book'           : months,
            'Total_Relationship_Count' : total_rel_count,
            'Months_Inactive_12_mon'   : months_inactive,
            'Contacts_Count_12_mon'    : contacts_count,
            'Total_Revolving_Bal'      : total_rev_bal,
            'Total_Trans_Amt'          : total_trans_amt,
            'Total_Trans_Ct'           : total_trans_ct,
        }
        if gender == 'M':
            inputs['Gender_M'] = 1
        if income != "$120K +":
            k = f"Income_Category_{income}"
            if k in artifacts['feature_names']: inputs[k] = 1
        if card in ['Silver','Gold','Platinum']:
            k = f"Card_Category_{card}"
            if k in artifacts['feature_names']: inputs[k] = 1

        row = build_row(inputs)
        lr_p, xg_p, op_p = predict_trio(row)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Risk Assessment Results")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge(lr_p, "Logistic Regression"), use_container_width=True, theme=None)
            st.caption("Interpretable baseline model")
        with g2:
            st.plotly_chart(gauge(xg_p, "XGBoost + CTGAN ⭐"), use_container_width=True, theme=None)
            st.caption("Best-in-class model")
        with g3:
            st.plotly_chart(gauge(op_p, "XGBoost + Optuna"), use_container_width=True, theme=None)
            st.caption("Hyperparameter-tuned variant")

        st.markdown("**Risk level (best model):** " + badge(xg_p), unsafe_allow_html=True)

        # ── SHAP explanation (explains the best model: XGBoost + CTGAN) ──
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Why did the model give this score?")

        ex_left, ex_right = st.columns([1, 1])
        with ex_left:
            shap_vals = explainer.shap_values(row)
            sv        = pd.Series(shap_vals[0], index=row.columns)
            sv_top    = sv.abs().sort_values(ascending=False).head(8)

            st.markdown("""
            <div class="callout">
            The chart on the right shows which features pushed this customer's
            risk score <span style="color:#DC2626;font-weight:600;">up (red / right)</span> or
            <span style="color:#3B82F6;font-weight:600;">down (blue / left)</span>.
            The longer the bar, the bigger the impact.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Top factors for this customer:**")
            for feat in sv_top.index:
                direction = "⬆ increases" if sv[feat] > 0 else "⬇ reduces"
                color     = "#DC2626" if sv[feat] > 0 else "#059669"
                st.markdown(
                    f"<span style='color:{color};font-weight:600;'>• {feat}</span> "
                    f"— <em>{direction} churn risk</em>",
                    unsafe_allow_html=True
                )

        with ex_right:
            fig_shap, ax = plt.subplots(figsize=(7, 4))

            # --- ARKA PLANI BEYAZA ZORLA ---
            fig_shap.patch.set_facecolor('white')
            ax.set_facecolor('white')

            colors_shap = ['#DC2626' if v > 0 else '#3B82F6' for v in sv.loc[sv_top.index]]
            ax.barh(sv_top.index[::-1], sv.loc[sv_top.index][::-1], color=colors_shap[::-1])
            ax.axvline(0, color='#374151', linewidth=0.8, linestyle='--')

            # --- YAZILARI KOYU RENGE ZORLA ---
            ax.set_xlabel("SHAP value (impact on churn probability)", color='#0A1929')
            ax.set_title("Feature Contributions", fontsize=11, fontweight='bold', color='#0A1929')
            ax.tick_params(labelsize=9, colors='#0A1929')
            ax.spines['bottom'].set_color('#0A1929')
            ax.spines['left'].set_color('#0A1929')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

        # ── Suggested action (based on best model: XGBoost + CTGAN) ──
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Recommended Action")

        actions = []
        if total_trans_ct < 40:
            actions.append("**Send a cashback / reward campaign** — re-activate card usage with category-based incentives. Target: +20% transaction count within 60 days.")
        if months_inactive >= 3:
            actions.append("**Schedule a relationship-manager outreach call** — personal contact within 7 days. Goal: identify the dormant cause (life event, dissatisfaction, competitor offer).")
        if total_rel_count <= 2:
            actions.append("**Make a cross-sell offer** — bundle savings account, credit card upgrade, or investment product. Customers with 3+ products are significantly less likely to leave.")
        if contacts_count >= 4:
            actions.append("**Review recent service interactions** — high contact count suggests unresolved friction. Escalate to a senior advisor for a satisfaction review.")
        if not actions:
            actions.append("**Maintain standard engagement** — this customer appears satisfied. Include in quarterly relationship-health check.")

        box_class = "action-box-urgent" if xg_p >= 0.6 else "action-box"
        action_html = f'<div class="{box_class}">'
        if xg_p >= 0.6:
            action_html += "<b>🚨 HIGH PRIORITY — Act within 7 days</b><br><br>"
        elif xg_p >= 0.3:
            action_html += "<b>⚠️ MEDIUM PRIORITY — Include in next retention cycle</b><br><br>"
        else:
            action_html += "<b>✅ LOW PRIORITY — Standard monitoring</b><br><br>"
        for a in actions:
            action_html += f"• {a}<br><br>"
        action_html += "</div>"
        st.markdown(action_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ══════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.title("Batch Prediction")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout-gold">
    Score your <b>entire customer portfolio at once</b>. Upload a CSV file and download
    a new file with each customer's churn probability and risk level added.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step-by-step guide")
    s1, s2, s3 = st.columns(3)
    s1.markdown("**① Download the template**\nClick the button below to get a sample CSV showing the exact format required.")
    s2.markdown("**② Upload your file**\nReplace the sample rows with your own customers (same column headers).")
    s3.markdown("**③ Download results**\nThe app will add `Churn_Probability` and `Risk_Level` columns.")

    # Template download using X_test
    try:
        template_df = pd.read_csv('X_test.csv').head(10)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download sample template (10 rows)",
            csv_template, "churn_template.csv", "text/csv",
        )
    except FileNotFoundError:
        st.info("Template file not found in repo. Use any CSV with the 29 model features.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your CSV here — or click to browse",
        type="csv",
        help="Must contain the same 29 feature columns as the template."
    )

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**Preview** — {len(df_up):,} customers loaded")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("⚡ Score All Customers", use_container_width=True):
            with st.spinner(f"Scoring {len(df_up):,} customers…"):
                # Best model = XGBoost + CTGAN
                proba = xgb_model.predict_proba(df_up)[:, 1]
                pred  = (proba >= 0.5).astype(int)
                result = df_up.copy()
                result['Churn_Probability_%'] = (proba * 100).round(2)
                result['Predicted_Churn']     = pred
                result['Risk_Level']          = [rlabel(p) for p in proba]

            st.success(f"✅ {len(result):,} customers scored.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Low Risk",    int((result['Risk_Level']=='Low').sum()))
            m2.metric("Medium Risk", int((result['Risk_Level']=='Medium').sum()))
            m3.metric("High Risk",   int((result['Risk_Level']=='High').sum()))
            m4.metric("Avg Churn Prob", f"{result['Churn_Probability_%'].mean():.1f}%")

            st.markdown("**Top 20 highest-risk customers:**")
            st.dataframe(
                result.sort_values('Churn_Probability_%', ascending=False)
                      [['Churn_Probability_%','Predicted_Churn','Risk_Level']].head(20),
                use_container_width=True
            )

            dl = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download full scored file",
                dl, "churn_scored.csv", "text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════
# PAGE 4 — RISK SEGMENTATION
# ══════════════════════════════════════════════════════
elif page == "🎯 Risk Segmentation":
    st.title("Risk Segmentation")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout-gold">
    This page automatically divides customers into three risk groups —
    <b>Low, Medium, and High</b> — so your team can prioritise who to contact first.
    It also shows how accurate these groupings are against real historical outcomes.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_seg():
        X  = pd.read_csv('X_test.csv')
        y  = pd.read_csv('y_test.csv').squeeze()
        # Best model = XGBoost + CTGAN
        p  = xgb_model.predict_proba(X)[:, 1]
        return X, y, p

    try:
        _, y_t, prob = get_seg()
        seg = pd.DataFrame({
            'Probability'  : prob,
            'Actual_Churn' : y_t.values,
            'Risk_Level'   : [rlabel(p) for p in prob],
        })

        m1, m2, m3 = st.columns(3)
        low_n = int((seg['Risk_Level']=='Low').sum())
        mid_n = int((seg['Risk_Level']=='Medium').sum())
        hi_n  = int((seg['Risk_Level']=='High').sum())
        m1.metric("Low Risk customers",    f"{low_n:,}",
                  help="Churn probability < 30% — maintain standard service")
        m2.metric("Medium Risk customers", f"{mid_n:,}",
                  help="Churn probability 30–60% — include in next retention wave")
        m3.metric("High Risk customers",   f"{hi_n:,}",
                  help="Churn probability > 60% — prioritise for immediate outreach")

        tab_biz, tab_data = st.tabs(["📋 For Relationship Managers", "📈 For Data & Analytics Team"])

        with tab_biz:
            st.markdown("#### What does this mean for your team?")
            st.markdown("""
            The bar chart below shows how many customers fall into each risk category.
            **Focus your retention budget on the red (High Risk) group first** — these are
            the customers most likely to leave in the coming months.
            """)
            counts = seg['Risk_Level'].value_counts().reindex(['Low','Medium','High'])
            fig_b = go.Figure(go.Bar(
                x=counts.index, y=counts.values,
                marker_color=['#10B981','#F59E0B','#DC2626'],
                text=[f"{v:,}" for v in counts.values],
                textposition='outside', textfont=dict(color='#0A1929', size=14),
            ))
            fig_b.update_layout(
                paper_bgcolor='white', plot_bgcolor='#F9FAFB',
                yaxis=dict(title='Number of customers', color='#374151'),
                xaxis=dict(color='#374151'),
                height=380, margin=dict(l=40,r=20,t=30,b=40),
                showlegend=False, font=dict(family='Inter', color='#374151'),
            )
            st.plotly_chart(fig_b, use_container_width=True, theme=None)

            st.markdown("""
            | Risk Group | Recommended Action | Timeline |
            |---|---|---|
            | 🔴 High Risk | Personal outreach call or priority retention offer | Within 7 days |
            | 🟡 Medium Risk | Include in automated campaign / newsletter | Within 30 days |
            | 🟢 Low Risk | Standard service — quarterly health check | Ongoing |
            """)

        with tab_data:
            st.markdown("#### Probability distribution and model calibration")
            st.markdown("""
            The histogram shows how model confidence is distributed across the test set.
            The calibration chart confirms that customers assigned to higher risk groups
            actually have higher real-world churn rates — validating the model's reliability.
            """)
            fig_h = px.histogram(
                seg, x='Probability', nbins=40, color='Risk_Level',
                color_discrete_map={'Low':'#10B981','Medium':'#F59E0B','High':'#DC2626'},
                labels={'Probability':'Predicted churn probability', 'Risk_Level':'Risk group'},
            )
            fig_h.update_layout(
                paper_bgcolor='white', plot_bgcolor='#F9FAFB',
                height=340, bargap=0.05,
                margin=dict(l=40,r=20,t=20,b=40),
                font=dict(family='Inter', color='#374151'),
            )
            st.plotly_chart(fig_h, use_container_width=True, theme=None)

            cal = seg.groupby('Risk_Level')['Actual_Churn'].mean().reindex(['Low','Medium','High'])
            fig_c = go.Figure(go.Bar(
                x=cal.index, y=(cal.values*100).round(1),
                marker_color=['#10B981','#F59E0B','#DC2626'],
                text=[f"{v:.1f}%" for v in (cal.values*100)],
                textposition='outside', textfont=dict(color='#0A1929', size=13),
            ))
            fig_c.update_layout(
                title="Actual churn rate per risk group (model calibration)",
                paper_bgcolor='white', plot_bgcolor='#F9FAFB',
                yaxis=dict(title='Actual churn rate (%)', range=[0,100], color='#374151'),
                xaxis=dict(color='#374151'),
                height=340, margin=dict(l=40,r=20,t=50,b=40),
                font=dict(family='Inter', color='#374151'),
            )
            st.plotly_chart(fig_c, use_container_width=True, theme=None)
            st.caption("Calibration plot: higher predicted risk correctly maps to higher real churn rates, confirming model reliability.")

    except FileNotFoundError:
        st.warning("X_test.csv / y_test.csv not found. Upload them alongside app.py in the repository.")


# ══════════════════════════════════════════════════════
# PAGE 5 — ACTION GUIDE
# ══════════════════════════════════════════════════════
elif page == "💡 Action Guide":
    st.title("Retention Action Guide")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout-gold">
    Based on our SHAP analysis, these are the features that most strongly predict churn.
    Each one has a matched retention tactic your team can deploy today.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key churn drivers (SHAP importance)")
    drivers = pd.DataFrame({
        'Feature'    : ['Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal',
                         'Months_Inactive_12_mon','Total_Relationship_Count',
                         'Total_Ct_Chng_Q4_Q1','Contacts_Count_12_mon'],
        'Importance' : [3.92, 1.92, 1.05, 0.93, 0.84, 0.73, 0.68],
        'Plain name' : ['Transaction count','Transaction volume','Credit balance used',
                         'Months inactive','Products held','Activity trend Q4/Q1',
                         'Support contacts'],
    })

    fig_d = px.bar(
        drivers.sort_values('Importance'),
        x='Importance', y='Plain name', orientation='h',
        color='Importance',
        color_continuous_scale=[[0,'#93C5FD'],[1,'#0A1929']],
        text='Importance',
    )
    fig_d.update_traces(texttemplate='%{text:.2f}', textposition='outside',
                        textfont=dict(color='#0A1929'))
    fig_d.update_layout(
        paper_bgcolor='white', plot_bgcolor='#F9FAFB',
        coloraxis_showscale=False, height=360,
        margin=dict(l=20,r=60,t=20,b=40),
        xaxis=dict(title='Mean |SHAP| value', color='#374151'),
        yaxis=dict(color='#374151'),
        font=dict(family='Inter', color='#374151'),
    )
    st.plotly_chart(fig_d, use_container_width=True, theme=None)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Matched retention tactics")

    actions = [
        ("🔴", "Low transaction count (< 40/year)",
         "Cashback or reward campaign",
         "Incentivise card activity with category-based rewards (e.g. 2x points on grocery spend). Target: +20% transactions within 60 days."),
        ("🔴", "3+ months of inactivity",
         "Relationship manager outreach call",
         "Personal call within 7 days. Identify root cause: life event, dissatisfaction, or better competitor offer. Offer a tailored solution."),
        ("🟡", "Only 1–2 products held",
         "Cross-sell bundle offer",
         "Offer a second product (savings, investment, or card upgrade) with a limited-time fee waiver. Customers with 3+ products churn ~50% less."),
        ("🟡", "4+ customer service contacts",
         "Complaint pattern review",
         "High contact count signals unresolved friction. Pull interaction history, escalate to senior advisor, and issue a goodwill gesture if warranted."),
        ("🟡", "Declining Q4 vs Q1 activity",
         "Behavioural re-engagement campaign",
         "Personalised email with offers tied to past spend categories. Show the customer what benefits they are leaving on the table."),
        ("🟢", "Low revolving balance",
         "Credit-line review or balance-transfer offer",
         "Offer a higher limit, a promotional APR, or a balance-transfer deal to increase engagement with the revolving credit product."),
    ]

    for priority, trigger, tactic, detail in actions:
        bg = "#FEF2F2" if priority == "🔴" else ("#FFFBEB" if priority == "🟡" else "#F0FDF4")
        bd = "#DC2626" if priority == "🔴" else ("#F59E0B" if priority == "🟡" else "#22C55E")
        st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {bd}; border-radius:0 8px 8px 0;
                    padding:16px 20px; margin-bottom:14px;">
          <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;
                      color:#6B7280; margin-bottom:4px;">TRIGGER {priority}</div>
          <div style="font-weight:600; color:#0A1929; font-size:1rem; margin-bottom:4px;">{trigger}</div>
          <div style="font-weight:700; color:{bd}; margin-bottom:6px;">→ {tactic}</div>
          <div style="color:#374151; font-size:0.93rem;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### Overview

    This tool is the applied output of a capstone project for the **Probability** course
    at Eskişehir Osmangazi University. It combines classical statistical inference with
    modern machine learning and explainable AI to predict bank customer churn.

    ---

    ### Methodology at a glance

    | Step | Technique | Purpose |
    |---|---|---|
    | Exploratory analysis | Correlation heatmap, chi-square tests | Understand which features relate to churn |
    | Class balancing | CTGAN (Conditional Tabular GAN) | Fix 84/16 imbalance in training data |
    | Baseline model | Logistic Regression (statsmodels) | Interpretable model with p-values and Odds Ratios |
    | High-performance model | XGBoost | Tree-ensemble for maximum predictive power |
    | Explainability | SHAP (TreeExplainer) | Make XGBoost predictions transparent |
    | Optimisation | Optuna (Bayesian search) | Fine-tune XGBoost hyperparameters |

    ---

    ### Research findings

    Three hypotheses were tested at the p < 0.05 significance level:

    | Hypothesis | Verdict | Key evidence |
    |---|---|---|
    | H1 — Fewer transactions → higher churn | ✅ Confirmed | OR = 0.06, p < 0.001 |
    | H2 — Fewer products → higher churn | ✅ Confirmed | OR = 0.47, p < 0.001 |
    | H3 — Lower income → higher churn | ❌ Rejected | High-income group ($120K+) actually churns the most |

    H3's rejection is a valuable insight: wealthier customers have more banking alternatives and are more likely to switch for a better offer.

    ---

    ### Final model results

    The best model is **XGBoost + CTGAN** — it gives the strongest Recall and F1.
    Optuna optimisation raised general accuracy slightly but traded off Recall, so the
    CTGAN model was kept as the final model.

    | | Logistic Regression | XGBoost + CTGAN | XGBoost + Optuna |
    |---|---|---|---|
    | Accuracy | 0.90 | 0.97 | 0.97 |
    | Recall | 0.54 | **0.88** | 0.83 |
    | F1-Score | 0.64 | **0.90** | 0.89 |
    | AUC | 0.917 | **0.991** | 0.989 |

    ---

    ### Dataset

    [Credit Card Customers — Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
    — 10,127 customers, 19 modelling features (after preprocessing).

    ---

    ### Project team

    Ece Gül Yüksel - Selin İnce - Didem Hışırcı

    **Instructor:** Sinem Bozkurt Keser
    **Course:** Probability — Eskişehir Osmangazi University
    """)
