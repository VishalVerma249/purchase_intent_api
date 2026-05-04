import os
import pickle

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# 1. Setup & Model Loading
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
le_month = model_dict["le_month"]
le_visitor = model_dict["le_visitor"]
feature_columns = model_dict["feature_columns"]

# ──────────────────────────────────────────────
# 2. Page Config & Header
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Purchase Intention Predictor",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* ── Global ── */
        .main {background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);}
        section[data-testid="stSidebar"] {background: #0d0d1a;}
        h1, h2, h3, h4 {color: #e0e0ff !important;}

        /* ── Accent bar ── */
        .accent-bar {
            height: 4px;
            border-radius: 2px;
            background: linear-gradient(90deg, #6c63ff, #3fc1c9, #f5367a);
            margin-bottom: 1.5rem;
        }

        /* ── Cards ── */
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            text-align: center;
            backdrop-filter: blur(8px);
        }
        .metric-card h2 {
            margin: 0; font-size: 2.2rem; font-weight: 700;
        }
        .metric-card p {
            margin: 0.3rem 0 0; font-size: 0.85rem;
            color: #aaa; text-transform: uppercase; letter-spacing: 1px;
        }

        /* ── Result boxes ── */
        .result-box {
            border-radius: 14px; padding: 2rem; text-align: center;
            margin-top: 1.2rem; font-size: 1.1rem;
        }
        .result-high {
            background: linear-gradient(135deg, #0d3b25, #145a38);
            border: 1px solid #2ecc71;
            color: #a7f3d0;
        }
        .result-high h2 {color: #2ecc71 !important; margin-bottom: 0.3rem;}
        .result-low {
            background: linear-gradient(135deg, #3b2a08, #4a3510);
            border: 1px solid #f39c12;
            color: #fde68a;
        }
        .result-low h2 {color: #f39c12 !important; margin-bottom: 0.3rem;}

        /* ── Tab styling ── */
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 10px 10px 0 0;
            padding: 0.6rem 1.6rem;
            color: #ccc;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .stTabs [aria-selected="true"] {
            background: rgba(108,99,255,0.15);
            color: #fff;
            border-bottom: 2px solid #6c63ff;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, #6c63ff, #3fc1c9);
            color: #fff; border: none;
            border-radius: 10px; padding: 0.6rem 2rem;
            font-weight: 600; transition: 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(108,99,255,0.4);
        }
        .stDownloadButton > button {
            background: linear-gradient(135deg, #6c63ff, #3fc1c9);
            color: #fff; border: none;
            border-radius: 10px; padding: 0.6rem 2rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🛒 E-Commerce Purchase Intention Predictor")
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.caption("Admin Dashboard · Powered by XGBoost")

# ──────────────────────────────────────────────
# 3. Tabs
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑‍💻  Live User Simulator", "📊  Bulk Analysis (CSV)"])

# ╔══════════════════════════════════════════════╗
# ║  TAB 1 – LIVE USER SIMULATOR                ║
# ╚══════════════════════════════════════════════╝
with tab1:
    st.markdown("#### Simulate a Single Visitor Session")
    st.markdown(
        "Fill in the visitor's browsing behaviour below and click **Predict Intent** "
        "to estimate purchase probability."
    )
    st.markdown("---")

    # ── Row 1: Page-level metrics ──
    col1, col2, col3 = st.columns(3)
    with col1:
        administrative = st.number_input(
            "Administrative (pages)", min_value=0, value=0, step=1,
            help="Number of administrative pages visited.",
        )
        administrative_duration = st.number_input(
            "Administrative Duration (s)", min_value=0.0, value=0.0, step=1.0,
            help="Total seconds spent on administrative pages.",
        )
    with col2:
        informational = st.number_input(
            "Informational (pages)", min_value=0, value=0, step=1,
            help="Number of informational pages visited.",
        )
        informational_duration = st.number_input(
            "Informational Duration (s)", min_value=0.0, value=0.0, step=1.0,
            help="Total seconds spent on informational pages.",
        )
    with col3:
        product_related = st.number_input(
            "Product Related (pages)", min_value=0, value=1, step=1,
            help="Number of product-related pages visited.",
        )
        product_related_duration = st.number_input(
            "Product Related Duration (s)", min_value=0.0, value=0.0, step=1.0,
            help="Total seconds spent on product-related pages.",
        )

    st.markdown("---")

    # ── Row 2: Rate & value metrics ──
    col4, col5, col6 = st.columns(3)
    with col4:
        bounce_rates = st.number_input(
            "Bounce Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
            format="%.4f",
            help="Average bounce rate of the pages visited.",
        )
        exit_rates = st.number_input(
            "Exit Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
            format="%.4f",
            help="Average exit rate of the pages visited.",
        )
    with col5:
        page_values = st.number_input(
            "Page Values", min_value=0.0, value=0.0, step=0.5,
            help="Average page value of the pages visited.",
        )
        special_day = st.number_input(
            "Special Day", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
            help="Closeness to a special day (e.g., Valentine's, Mother's Day).",
        )
    with col6:
        month = st.selectbox("Month", options=list(le_month.classes_))
        visitor_type = st.selectbox("Visitor Type", options=list(le_visitor.classes_))

    st.markdown("---")

    # ── Row 3: Technical / categorical ──
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        operating_systems = st.number_input(
            "Operating System", min_value=1, value=1, step=1,
            help="OS identifier (1-8).",
        )
    with col8:
        browser = st.number_input(
            "Browser", min_value=1, value=1, step=1,
            help="Browser identifier (1-13).",
        )
    with col9:
        region = st.number_input(
            "Region", min_value=1, value=1, step=1,
            help="Geographic region identifier (1-9).",
        )
    with col10:
        traffic_type = st.number_input(
            "Traffic Type", min_value=1, value=1, step=1,
            help="Traffic source identifier (1-20).",
        )

    weekend = st.selectbox("Weekend", options=["No", "Yes"])

    st.markdown("---")

    # ── Prediction ──
    if st.button("🔮  Predict Intent", use_container_width=True):
        weekend_val = 1 if weekend == "Yes" else 0
        month_enc = le_month.transform([month])[0]
        visitor_enc = le_visitor.transform([visitor_type])[0]

        input_data = pd.DataFrame(
            [
                {
                    "Administrative": administrative,
                    "Administrative_Duration": administrative_duration,
                    "Informational": informational,
                    "Informational_Duration": informational_duration,
                    "ProductRelated": product_related,
                    "ProductRelated_Duration": product_related_duration,
                    "BounceRates": bounce_rates,
                    "ExitRates": exit_rates,
                    "PageValues": page_values,
                    "SpecialDay": special_day,
                    "Month": month_enc,
                    "OperatingSystems": operating_systems,
                    "Browser": browser,
                    "Region": region,
                    "TrafficType": traffic_type,
                    "VisitorType": visitor_enc,
                    "Weekend": weekend_val,
                }
            ]
        )

        # Ensure column order matches training
        input_data = input_data[feature_columns]

        proba = model.predict_proba(input_data)[0][1]
        pct = proba * 100

        if proba > 0.50:
            st.markdown(
                f"""
                <div class="result-box result-high">
                    <h2>🟢 High Purchase Intent</h2>
                    <p style="font-size:2rem;font-weight:700;margin:0.4rem 0;">{pct:.1f}%</p>
                    <p>This visitor is <strong>very likely</strong> to complete a purchase.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-box result-low">
                    <h2>🟡 Low Intent · Window Shopper</h2>
                    <p style="font-size:2rem;font-weight:700;margin:0.4rem 0;">{pct:.1f}%</p>
                    <p>This visitor is <strong>unlikely</strong> to purchase. Consider retargeting.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ╔══════════════════════════════════════════════╗
# ║  TAB 2 – BULK ANALYSIS (CSV UPLOAD)         ║
# ╚══════════════════════════════════════════════╝
with tab2:
    st.markdown("#### Upload a Customer Dataset for Bulk Prediction")
    st.markdown(
        "Upload a CSV that matches the training data schema. The tool will score "
        "every row and surface high-intent users for your marketing team."
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drop your CSV here", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.markdown("##### 📄 Preview of Uploaded Data")
        st.dataframe(raw_df.head(10), use_container_width=True)

        # Summary cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card"><h2>{len(raw_df):,}</h2>'
                f"<p>Total Rows</p></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><h2>{len(raw_df.columns)}</h2>'
                f"<p>Columns</p></div>",
                unsafe_allow_html=True,
            )
        with c3:
            missing = int(raw_df.isnull().sum().sum())
            st.markdown(
                f'<div class="metric-card"><h2>{missing}</h2>'
                f"<p>Missing Values</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        if st.button("🚀  Run Bulk Prediction", use_container_width=True):
            df = raw_df.copy()

            # Preprocess exactly like training
            if "Weekend" in df.columns:
                df["Weekend"] = df["Weekend"].astype(int)
            if "Revenue" in df.columns:
                df.drop("Revenue", axis=1, inplace=True)
            if "Month" in df.columns:
                df["Month"] = le_month.transform(df["Month"])
            if "VisitorType" in df.columns:
                df["VisitorType"] = le_visitor.transform(df["VisitorType"])

            # Ensure column order
            df = df[feature_columns]

            probas = model.predict_proba(df)[:, 1]
            raw_df["Purchase_Probability (%)"] = (probas * 100).round(2)

            # ── High-intent filter ──
            high_intent_df = raw_df[raw_df["Purchase_Probability (%)"] > 50.0].copy()

            st.markdown("---")

            # Result cards
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f'<div class="metric-card"><h2>{len(raw_df):,}</h2>'
                    f"<p>Total Scored</p></div>",
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="metric-card"><h2 style="color:#2ecc71">'
                    f"{len(high_intent_df):,}</h2>"
                    f"<p>High-Intent Users (&gt;50 %)</p></div>",
                    unsafe_allow_html=True,
                )
            with r3:
                conv_rate = (
                    (len(high_intent_df) / len(raw_df) * 100) if len(raw_df) else 0
                )
                st.markdown(
                    f'<div class="metric-card"><h2 style="color:#3fc1c9">'
                    f"{conv_rate:.1f}%</h2>"
                    f"<p>High-Intent Rate</p></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("##### 🎯 High-Intent Users (Probability > 50%)")

            if high_intent_df.empty:
                st.info("No users exceeded the 50 % threshold in this dataset.")
            else:
                st.dataframe(
                    high_intent_df.sort_values(
                        "Purchase_Probability (%)", ascending=False
                    ),
                    use_container_width=True,
                )

                csv_bytes = high_intent_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download High-Intent Users CSV",
                    data=csv_bytes,
                    file_name="high_intent_users.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
