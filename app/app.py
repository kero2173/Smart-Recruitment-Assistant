"""
Smart Recruitment Assistant — Candidate Predictor
--------------------------------------------------
A small Streamlit interface around the Task 5/6/7 pipeline:
loads best_model.pkl + preprocessing.pkl (already fitted, no retraining)
and predicts whether a candidate is likely to be looking for a job change.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="🎯",
    layout="centered",
)

# ----------------------------------------------------------------------
# Dark, high-contrast styling on top of the base Streamlit dark theme
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1b1430 0%, #0e1117 55%);
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }
    div[data-testid="stForm"] {
        background: #15182199;
        border: 1px solid #2a2e3d;
        border-radius: 16px;
        padding: 1.6rem 1.6rem 0.8rem 1.6rem;
    }
    .section-label {
        color: #a78bfa;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0.6rem 0 0.4rem 0;
    }
    .result-card {
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.4rem;
        border: 1px solid #2a2e3d;
    }
    .result-positive {
        background: linear-gradient(135deg, #3b1f2b, #1a1220);
        border-color: #f43f5e55;
    }
    .result-negative {
        background: linear-gradient(135deg, #133327, #101c22);
        border-color: #22c55e55;
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 800;
    }
    .result-sub {
        color: #b8bcc8;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }
    .prob-bar-bg {
        background: #262a38;
        border-radius: 999px;
        height: 10px;
        margin-top: 0.9rem;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load the already-fitted model + preprocessing (produced in Task 6 / 7)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    prep = joblib.load("preprocessing.pkl")
    return model, prep


model, prep = load_artifacts()

# ----------------------------------------------------------------------
# Same preprocessing logic as Task 7's predict_candidate(), reused as-is
# ----------------------------------------------------------------------
def predict_candidate(raw_candidate: dict, model=model, prep=prep):
    cand = pd.DataFrame([raw_candidate])

    # 1) Missing values (same rule as Task 2) — not expected here since the
    #    form always supplies a value, kept for robustness with partial input
    for col in prep["high_missing"]:
        if col in cand.columns:
            cand[col] = cand[col].fillna("Unknown")
    for col in prep["low_missing"]:
        if col in cand.columns:
            cand[col] = cand[col].fillna(prep["low_missing_modes"][col])

    # 2) Clean categorical strings
    for col in prep["high_missing"] + prep["low_missing"]:
        if col in cand.columns:
            cand[col] = cand[col].astype(str).str.strip()

    # 3) Map experience / last_new_job to numeric (Task 4 mapping)
    cand["experience"] = cand["experience"].map(prep["exp_map"])
    cand["last_new_job"] = cand["last_new_job"].map(prep["job_map"])

    # 4) log1p transform on training_hours (Task 4)
    cand["training_hours"] = np.log1p(cand["training_hours"])

    # 5) Manual one-hot encoding against the exact training columns
    #    (avoids the drop_first=True single-row bug found in Task 7)
    for col in prep["categorical_cols"]:
        prefix = col + "_"
        dummy_cols_for_this_feature = [
            c for c in prep["training_columns"] if c.startswith(prefix)
        ]
        for dummy_col in dummy_cols_for_this_feature:
            category_value = dummy_col[len(prefix):]
            cand[dummy_col] = (cand[col].astype(str) == category_value).astype(int)
        cand = cand.drop(columns=[col])

    # 6) Align to the exact 34 training columns, in order
    cand = cand.reindex(columns=prep["training_columns"], fill_value=0)

    # 7) Scale numeric columns with the already-fitted scaler (transform only)
    cand[prep["numeric_cols"]] = prep["scaler"].transform(cand[prep["numeric_cols"]])

    pred = int(model.predict(cand)[0])
    proba = float(model.predict_proba(cand)[0][1])  # P(target = 1)
    label = "Looking for a job change" if pred == 1 else "Not looking for a job change"

    return {"prediction": pred, "label": label, "probability": proba}


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.markdown('<div class="app-title">Smart Recruitment Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter a candidate\'s profile to get a job-change-risk '
    "prediction from the trained model (Random Forest, F1 = 0.53).</div>",
    unsafe_allow_html=True,
)

with st.form("candidate_form"):
    st.markdown('<div class="section-label">Background</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        education_level = st.selectbox(
            "Education level", ["Graduate", "Masters", "Phd", "High School", "Primary School"]
        )
    with c2:
        major_discipline = st.selectbox(
            "Major discipline",
            ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"],
        )
        enrolled_university = st.selectbox(
            "University enrollment", ["no_enrollment", "Full time course", "Part time course"]
        )

    st.markdown('<div class="section-label">Experience</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        relevent_experience = st.selectbox(
            "Relevant experience", ["Has relevent experience", "No relevent experience"]
        )
        experience_options = ["<1"] + [str(i) for i in range(1, 21)] + [">20"]
        experience = st.selectbox("Years of experience", experience_options, index=6)
    with c4:
        last_new_job = st.selectbox("Years since last job change", ["never", "1", "2", "3", "4", ">4"])
        training_hours = st.number_input("Training hours completed", min_value=0, max_value=500, value=65)

    st.markdown('<div class="section-label">Current company</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        company_size = st.selectbox(
            "Company size",
            ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"],
        )
    with c6:
        company_type = st.selectbox(
            "Company type",
            ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other", "Unknown"],
        )

    st.markdown('<div class="section-label">City</div>', unsafe_allow_html=True)
    city_development_index = st.slider(
        "City development index", min_value=0.4, max_value=0.95, value=0.75, step=0.01,
        help="Higher = more developed city. This is the single strongest predictor in the EDA.",
    )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    gender_value = "Unknown" if gender == "Prefer not to say" else gender
    raw_candidate = {
        "city_development_index": city_development_index,
        "gender": gender_value,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours": training_hours,
    }

    result = predict_candidate(raw_candidate)
    prob_pct = result["probability"] * 100

    if result["prediction"] == 1:
        card_class = "result-positive"
        headline = "⚠️ Likely looking for a job change"
        bar_color = "#f43f5e"
    else:
        card_class = "result-negative"
        headline = "✅ Likely to stay"
        bar_color = "#22c55e"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-title">{headline}</div>
            <div class="result-sub">Model confidence that this candidate is looking for a job change: {prob_pct:.1f}%</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{prob_pct:.1f}%; background:{bar_color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show raw input sent to the model"):
        st.json(raw_candidate)

st.markdown(
    '<div style="color:#5b5f70; font-size:0.78rem; margin-top:2rem;">'
    "Target meaning — 0: not looking for a job change · 1: looking for a job change. "
    "This is a decision-support signal, not an automated hiring decision."
    "</div>",
    unsafe_allow_html=True,
)
