"""
Bank Customer Churn Risk Manager
Streamlit application built on top of LR + XGBoost (+ CTGAN + Optuna) models.

Pages
-----
1. Overview Dashboard
2. Single Customer Analysis (with SHAP explanation)
3. Batch Prediction (CSV upload)
4. Risk Segmentation
5. Action Recommendations
6. About
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

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Bank Churn Risk Manager",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — refined fintech aesthetic (deep navy + gold)
st.markdown("""
<style>
    /* Main font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif;
        color: #0A1929;
        letter-spacing: -0.5px;
    }
    h1 { font-weight: 700; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1929 0%, #142943 100%);
    }
    [data-testid="stSidebar"] * {
        color: #F5F5F0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #F5F5F0 !important;
        padding: 8px 0;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0A1929;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6B7280;
    }

    /* Buttons */
    .stButton > button {
        background: #0A1929;
        color: #F5F5F0;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #D4AF37;
        color: #0A1929;
        transform: translateY(-1px);
    }
    .stButton > button:focus {
        background: #D4AF37;
        color: #0A1929;
        box-shadow: none;
    }

    /* Risk badges */
    .risk-low {
        background: #D1FAE5;
        color: #065F46;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
        border-left: 4px solid #10B981;
    }
    .risk-medium {
        background: #FEF3C7;
        color: #92400E;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
        border-left: 4px solid #F59E0B;
    }
    .risk-high {
        background: #FEE2E2;
        color: #991B1B;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 700;
        display: inline-block;
        border-left: 4px solid #DC2626;
    }

    /* Headline accent line */
    .accent-line {
        width: 60px;
        height: 3px;
        background: #D4AF37;
        margin: 8px 0 24px 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS & ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    xgb_model  = joblib.load('xgb_model.pkl')
    xgb_optuna = joblib.load('xgb_optuna.pkl')
    scaler     = joblib.load('scaler.pkl')
    artifacts  = joblib.load('artifacts.pkl')
    lr_beta    = np.load('lr_beta.npy')
    return xgb_model, xgb_optuna, scaler, artifacts, lr_beta


@st.cache_resource
def load_shap_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)


try:
    xgb_model, xgb_optuna, scaler, artifacts, lr_beta = load_artifacts()
    explainer = load_shap_explainer(xgb_model)
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def scale_value(raw, col_name):
    """Apply StandardScaler transformation manually."""
    idx = artifacts['sayisal_sutunlar'].index(col_name)
    mean = artifacts['scaler_mean'][idx]
    scale = artifacts['scaler_scale'][idx]
    return (raw - mean) / scale


def build_customer_row(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame compatible with the trained models.
    Starts from feature_means (default profile) and overrides with user inputs.
    """
    feature_names = artifacts['feature_names']
    means         = artifacts['feature_means']
    numeric_cols  = artifacts['sayisal_sutunlar']

    row = pd.DataFrame([[means[c] for c in feature_names]], columns=feature_names)

    for col, val in inputs.items():
        if col in numeric_cols:
            row[col] = scale_value(val, col)
        elif col in feature_names:
            row[col] = val
    return row


def predict_all_models(row_df: pd.DataFrame):
    """Return churn probabilities for all three models."""
    lr_input = np.concatenate([[1.0], row_df.astype(float).values[0]])
    lr_prob  = float(1 / (1 + np.exp(-np.dot(lr_beta, lr_input))))

    xgb_prob = float(xgb_model.predict_proba(row_df)[0][1])
    opt_prob = float(xgb_optuna.predict_proba(row_df)[0][1])

    return lr_prob, xgb_prob, opt_prob


def risk_badge(prob: float) -> str:
    """Return HTML-styled risk badge."""
    if prob < 0.30:
        return f'<div class="risk-low">LOW RISK &nbsp;&nbsp; {prob*100:.1f}%</div>'
    elif prob < 0.60:
        return f'<div class="risk-medium">MEDIUM RISK &nbsp;&nbsp; {prob*100:.1f}%</div>'
    else:
        return f'<div class="risk-high">HIGH RISK &nbsp;&nbsp; {prob*100:.1f}%</div>'


def risk_label(prob: float) -> str:
    if prob < 0.30:  return "Low"
    elif prob < 0.60: return "Medium"
    else:             return "High"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("# 🏦 CHURN MANAGER")
    st.markdown("<div style='color:#D4AF37; letter-spacing:2px; font-size:0.8rem;'>RISK INTELLIGENCE PANEL</div>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview Dashboard",
         "👤 Single Customer Analysis",
         "📁 Batch Prediction",
         "🎯 Risk Segmentation",
         "💡 Action Recommendations",
         "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Models in production")
    st.markdown("✓ Logistic Regression")
    st.markdown("✓ XGBoost + CTGAN")
    st.markdown("✓ XGBoost + Optuna")
    st.markdown("---")
    st.caption("Built with Streamlit · Ece Gül Yüksel · Selin İnce · Didem Hışırcı")


# ============================================================
# MODEL LOAD ERROR HANDLING
# ============================================================
if not MODELS_LOADED:
    st.error(f"❌ Model files could not be loaded: {LOAD_ERROR}")
    st.info("Please ensure the following files are in the same directory as app.py:\n"
            "- xgb_model.pkl\n- xgb_optuna.pkl\n- scaler.pkl\n- artifacts.pkl\n- lr_beta.npy")
    st.stop()


# ============================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================
if page == "📊 Overview Dashboard":
    st.title("Overview Dashboard")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("Strategic snapshot of customer churn risk across the portfolio.")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", "10,127")
    c2.metric("Current Churn Rate", "16.1%")
    c3.metric("Best Model AUC", "0.991")
    c4.metric("Recall (best model)", "89%")

    st.markdown("###  ")

    # Model performance comparison
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Model Performance Comparison")
        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

        df_comparison = pd.DataFrame({
            'Model'     : ['Logistic Regression', 'XGBoost + CTGAN', 'XGBoost + Optuna'],
            'Accuracy'  : [0.90, 0.97, 0.97],
            'Precision' : [0.78, 0.92, 0.92],
            'Recall'    : [0.54, 0.86, 0.89],
            'F1-Score'  : [0.64, 0.89, 0.90],
            'AUC'       : [0.917, 0.991, 0.990],
        })

        # Plotly comparison
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        colors  = ['#4472C4', '#2ECC71', '#D4AF37']
        for i, m in enumerate(df_comparison['Model']):
            fig.add_trace(go.Scatterpolar(
                r=df_comparison[metrics].iloc[i].values,
                theta=metrics,
                fill='toself',
                name=m,
                line=dict(color=colors[i], width=2),
                opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.5, 1]),
                bgcolor='#FAFAF7',
            ),
            paper_bgcolor='white',
            font=dict(family='Inter', color='#0A1929'),
            height=420,
            margin=dict(l=40, r=40, t=20, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Class Distribution")
        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

        fig_pie = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[8500, 1627],
            hole=0.55,
            marker=dict(colors=['#0A1929', '#D4AF37']),
            textfont=dict(size=14, color='white'),
        )])
        fig_pie.update_layout(
            paper_bgcolor='white',
            font=dict(family='Inter', color='#0A1929'),
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.05),
            annotations=[dict(text='10,127<br>customers', x=0.5, y=0.5,
                             font_size=16, showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("###  ")
    st.subheader("Detailed Metrics Table")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2: SINGLE CUSTOMER ANALYSIS
# ============================================================
elif page == "👤 Single Customer Analysis":
    st.title("Single Customer Analysis")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("Enter customer attributes and receive an instant churn risk assessment "
                "with **SHAP explanations** highlighting the most influential factors.")

    # Two-column input layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🔄 Behavioral Profile")
        total_trans_ct  = st.slider("Total transactions (last 12 months)",
                                     10, 139, 60, help="Higher values strongly reduce churn risk")
        total_trans_amt = st.slider("Total transaction amount ($)",
                                     510, 18484, 4404, step=100)
        total_rel_count = st.slider("Products held with the bank",
                                     1, 6, 3)
        months_inactive = st.slider("Months inactive (last 12)",
                                     0, 6, 1, help="More inactivity → higher churn risk")
        contacts_count  = st.slider("Customer-service contacts (last 12 months)",
                                     0, 6, 2)
        total_rev_bal   = st.slider("Total revolving balance ($)",
                                     0, 2517, 1163, step=50)

    with col2:
        st.markdown("##### 👤 Demographics")
        age              = st.slider("Customer age", 26, 73, 46)
        dependents       = st.slider("Dependent count", 0, 5, 2)
        months_on_book   = st.slider("Months on book", 13, 56, 36)
        gender           = st.selectbox("Gender", ["F", "M"])
        income           = st.selectbox("Income category",
                                         ["Less than $40K", "$40K - $60K", "$60K - $80K",
                                          "$80K - $120K", "$120K +", "Unknown"])
        card             = st.selectbox("Card category",
                                         ["Blue", "Silver", "Gold", "Platinum"])

    st.markdown("###  ")

    # Predict button
    if st.button("🔮 Run Risk Assessment", use_container_width=True):
        # Build customer row
        inputs = {
            'Customer_Age'              : age,
            'Dependent_count'           : dependents,
            'Months_on_book'            : months_on_book,
            'Total_Relationship_Count'  : total_rel_count,
            'Months_Inactive_12_mon'    : months_inactive,
            'Contacts_Count_12_mon'     : contacts_count,
            'Total_Revolving_Bal'       : total_rev_bal,
            'Total_Trans_Amt'           : total_trans_amt,
            'Total_Trans_Ct'            : total_trans_ct,
        }
        # One-hot fields
        if gender == 'M':
            inputs['Gender_M'] = 1
        if income != "$120K +":
            col = f"Income_Category_{income}"
            if col in artifacts['feature_names']:
                inputs[col] = 1
        if card in ['Silver', 'Gold', 'Platinum']:
            col = f"Card_Category_{card}"
            if col in artifacts['feature_names']:
                inputs[col] = 1

        row_df = build_customer_row(inputs)
        lr_p, xgb_p, opt_p = predict_all_models(row_df)

        # Display predictions
        st.markdown("### Prediction Results")
        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Logistic Regression**")
            st.markdown(risk_badge(lr_p), unsafe_allow_html=True)
            st.caption("Interpretable baseline")
        with rc2:
            st.markdown("**XGBoost + CTGAN**")
            st.markdown(risk_badge(xgb_p), unsafe_allow_html=True)
            st.caption("Balanced training")
        with rc3:
            st.markdown("**XGBoost + Optuna ⭐**")
            st.markdown(risk_badge(opt_p), unsafe_allow_html=True)
            st.caption("Best-in-class model")

        # SHAP explanation
        st.markdown("###  ")
        st.markdown("### Why this prediction?")
        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
        st.markdown("Each bar shows how a specific feature pushed the prediction "
                    "*toward* (red) or *away from* (blue) churn.")

        shap_vals = explainer.shap_values(row_df)

        fig, ax = plt.subplots(figsize=(11, 5))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_vals[0],
            feature_names=row_df.columns,
            max_display=12,
            show=False,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tailored action
        st.markdown("###  ")
        st.markdown("### Suggested action")
        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
        prob_use = opt_p
        if prob_use >= 0.60:
            st.error("🚨 **HIGH-PRIORITY INTERVENTION RECOMMENDED**")
            actions = []
            if total_trans_ct < 40:
                actions.append("• Send a **cashback / reward campaign** to re-activate transactions")
            if months_inactive >= 3:
                actions.append("• Schedule a **relationship-manager outreach call** within 7 days")
            if total_rel_count <= 2:
                actions.append("• Offer a **cross-sell promotion** (savings, credit card, investment)")
            if not actions:
                actions.append("• Initiate **personalized retention review** with senior advisor")
            for a in actions:
                st.markdown(a)
        elif prob_use >= 0.30:
            st.warning("⚠️ **Medium risk** — monitor for the next 30 days; include in periodic engagement campaigns.")
        else:
            st.success("✅ **Low risk** — customer appears engaged. Continue standard service.")


# ============================================================
# PAGE 3: BATCH PREDICTION
# ============================================================
elif page == "📁 Batch Prediction":
    st.title("Batch Prediction")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV of preprocessed customer records to receive churn "
                "probabilities and risk levels for the entire portfolio.")

    st.info("📋 **CSV format expected:** Same 29 features used in model training. "
            "You can download `X_test.csv` from the repository as a reference template.")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown("##### Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Total records: {len(df):,}")

        if st.button("⚡ Run Batch Predictions", use_container_width=True):
            with st.spinner("Scoring customers..."):
                # Predict with best model
                proba = xgb_optuna.predict_proba(df)[:, 1]
                pred  = (proba >= 0.5).astype(int)

                result = df.copy()
                result['Churn_Probability'] = (proba * 100).round(2)
                result['Predicted_Churn']   = pred
                result['Risk_Level']        = [risk_label(p) for p in proba]

            st.success(f"✅ Scored {len(result):,} customers successfully.")

            # Summary
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Low Risk",    int((result['Risk_Level'] == 'Low').sum()))
            sc2.metric("Medium Risk", int((result['Risk_Level'] == 'Medium').sum()))
            sc3.metric("High Risk",   int((result['Risk_Level'] == 'High').sum()))
            sc4.metric("Avg Churn Prob", f"{result['Churn_Probability'].mean():.1f}%")

            st.markdown("##### Results preview")
            display_cols = ['Churn_Probability', 'Predicted_Churn', 'Risk_Level']
            st.dataframe(result[display_cols].head(20), use_container_width=True)

            # Download
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download full results (CSV)",
                csv,
                "churn_predictions.csv",
                "text/csv",
                use_container_width=True,
            )


# ============================================================
# PAGE 4: RISK SEGMENTATION
# ============================================================
elif page == "🎯 Risk Segmentation":
    st.title("Risk Segmentation")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("Strategic partition of the test-set portfolio by predicted churn risk.")

    # Load test data
    @st.cache_data
    def load_test_data():
        X = pd.read_csv('X_test.csv')
        y = pd.read_csv('y_test.csv').squeeze()
        proba = xgb_optuna.predict_proba(X)[:, 1]
        return X, y, proba

    try:
        X_test, y_test, proba = load_test_data()

        seg = pd.DataFrame({
            'Probability': proba,
            'Actual_Churn': y_test.values,
            'Risk_Level': [risk_label(p) for p in proba],
        })

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Low Risk",    int((seg['Risk_Level']=='Low').sum()),
                  f"{(seg['Risk_Level']=='Low').mean()*100:.1f}%")
        c2.metric("Medium Risk", int((seg['Risk_Level']=='Medium').sum()),
                  f"{(seg['Risk_Level']=='Medium').mean()*100:.1f}%")
        c3.metric("High Risk",   int((seg['Risk_Level']=='High').sum()),
                  f"{(seg['Risk_Level']=='High').mean()*100:.1f}%")

        st.markdown("###  ")

        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.subheader("Distribution of Predicted Probabilities")
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                seg, x='Probability', nbins=40, color='Risk_Level',
                color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#DC2626'},
            )
            fig_hist.update_layout(
                paper_bgcolor='white', plot_bgcolor='#FAFAF7',
                font=dict(family='Inter', color='#0A1929'),
                bargap=0.05,
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            st.subheader("Segment Breakdown")
            st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
            counts = seg['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High'])
            fig_bar = go.Figure(data=[go.Bar(
                x=counts.index,
                y=counts.values,
                marker=dict(color=['#10B981', '#F59E0B', '#DC2626']),
                text=counts.values,
                textposition='outside',
            )])
            fig_bar.update_layout(
                paper_bgcolor='white', plot_bgcolor='#FAFAF7',
                font=dict(family='Inter', color='#0A1929'),
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
                showlegend=False,
                yaxis=dict(title='Customer count'),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("###  ")
        st.subheader("Model Calibration: Risk Level vs Actual Churn")
        st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
        cal = seg.groupby('Risk_Level')['Actual_Churn'].mean().reindex(['Low', 'Medium', 'High'])
        cal_df = pd.DataFrame({
            'Risk_Level': cal.index,
            'Actual_Churn_Rate': (cal.values * 100).round(1)
        })
        fig_cal = px.bar(
            cal_df, x='Risk_Level', y='Actual_Churn_Rate',
            color='Risk_Level',
            color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#DC2626'},
            text='Actual_Churn_Rate',
        )
        fig_cal.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_cal.update_layout(
            paper_bgcolor='white', plot_bgcolor='#FAFAF7',
            font=dict(family='Inter', color='#0A1929'),
            height=350,
            yaxis=dict(title='Actual churn rate (%)', range=[0, 100]),
            xaxis=dict(title=''),
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_cal, use_container_width=True)
        st.caption("Higher predicted risk segments show progressively higher actual churn rates — confirming model calibration.")

    except FileNotFoundError:
        st.warning("Test set files (`X_test.csv` and `y_test.csv`) not found. Upload them alongside the app.")


# ============================================================
# PAGE 5: ACTION RECOMMENDATIONS
# ============================================================
elif page == "💡 Action Recommendations":
    st.title("Action Recommendations")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown("Rule-based retention strategies derived from SHAP analysis findings.")

    st.markdown("###  ")
    st.subheader("📉 Top Drivers of Churn (SHAP analysis)")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    drivers = pd.DataFrame({
        'Feature': ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                    'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count'],
        'Impact':  [2.44, 1.95, 1.17, 0.93, 0.84, 0.73, 0.68],
        'Direction': ['Higher amount → ↑ risk', 'Lower count → ↑ risk',
                      'Lower balance → ↑ risk', 'More inactivity → ↑ risk',
                      'More contacts → ↑ risk', 'Lower ratio → ↑ risk',
                      'Fewer products → ↑ risk'],
    })

    fig_imp = px.bar(
        drivers.sort_values('Impact'), x='Impact', y='Feature',
        orientation='h',
        color='Impact',
        color_continuous_scale=[[0, '#D4AF37'], [1, '#0A1929']],
    )
    fig_imp.update_layout(
        paper_bgcolor='white', plot_bgcolor='#FAFAF7',
        font=dict(family='Inter', color='#0A1929'),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("###  ")
    st.subheader("🎯 Recommended Actions by Trigger")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    actions = [
        ("Low transaction count (< 40 / year)",
         "🎁 **Cashback / reward campaign** — incentivize transactions with category-based rewards. Target: +20% transactions in 60 days."),
        ("3+ months of inactivity",
         "📞 **Relationship-manager outreach** — personal call within 7 days. Use case: identify dormant causes (life event, dissatisfaction)."),
        ("Holding ≤ 2 products",
         "💼 **Cross-sell promotion** — bundle offer for savings, credit card, or investment account. Reduce single-product churn risk by ~50%."),
        ("Increased customer-service contacts",
         "🛎️ **Customer experience review** — investigate complaint patterns. High contact count often signals unresolved friction."),
        ("Declining Q4 vs Q1 activity",
         "📊 **Behavioral re-engagement campaign** — personalized email with targeted offers based on past spend categories."),
        ("Low revolving balance utilization",
         "💳 **Credit-line review** — offer limit increase, balance-transfer promotion, or rewards on revolving usage."),
    ]
    for trigger, action in actions:
        st.markdown(f"**Trigger:** {trigger}")
        st.markdown(action)
        st.markdown("---")


# ============================================================
# PAGE 6: ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.title("About This Project")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview

    This dashboard is the productized output of a data-science capstone in
    **Probability & Statistics**. It combines classical statistical inference
    (Logistic Regression with p-values, Odds Ratios, hypothesis testing) with
    modern machine learning (XGBoost + CTGAN + Optuna) and explainable AI (SHAP).

    ### Dataset
    - **Source:** Kaggle — Credit Card Customers
    - **Size:** 10,127 customers × 21 features
    - **Target:** Attrition_Flag (0 = Retained, 1 = Churned)
    - **Class distribution:** 84% Retained / 16% Churned (imbalanced)

    ### Methodology Pipeline
    1. Exploratory Data Analysis (distributions, correlations, chi-square tests)
    2. Outlier analysis, encoding, scaling
    3. Stratified 80/20 train-test split
    4. Logistic Regression (statsmodels) — interpretable baseline
    5. CTGAN — synthetic minority-class generation (class balancing)
    6. XGBoost — tree-ensemble model on balanced data
    7. SHAP — explainability layer
    8. Optuna — Bayesian hyperparameter optimization

    ### Research Hypotheses
    | Hypothesis | Result |
    |---|---|
    | H1 — Lower transaction count → higher churn | ✅ Confirmed |
    | H2 — Fewer bank products → higher churn | ✅ Confirmed |
    | H3 — Lower income → higher churn | ❌ Rejected (high-income customers churn more) |

    ### Final Model Performance
    | Metric | LR | XGBoost+CTGAN | XGBoost+Optuna |
    |---|---|---|---|
    | Accuracy | 0.90 | 0.97 | 0.97 |
    | Precision | 0.78 | 0.92 | 0.92 |
    | Recall | 0.54 | 0.86 | **0.89** |
    | F1 | 0.64 | 0.89 | **0.90** |
    | AUC | 0.917 | 0.991 | 0.990 |

    ### Team
    Ece Gül Yüksel · Selin İnce · Didem Hışırcı

    ### Course
    Probability & Statistics — Eskişehir Osmangazi University
    """)
