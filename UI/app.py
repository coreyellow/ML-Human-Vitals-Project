import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from scipy.spatial.distance import cdist
from dotenv import load_dotenv
from groq import Groq

# Page config
st.set_page_config(
    page_title="Human Vitals",
    layout="wide",
)

# Load CSS
css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Paths
ROOT = Path(__file__).parent.parent

# Load API key
load_dotenv(ROOT / ".env")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load artefacts
@st.cache_resource
def load_artefacts():
    scaler_male   = joblib.load(ROOT / "Clustering_data/scaler_male.joblib")
    scaler_female = joblib.load(ROOT / "Clustering_data/scaler_female.joblib")
    labels_male   = joblib.load(ROOT / "Clustering_data/male_clustering_labels_ward_20.joblib")
    labels_female = joblib.load(ROOT / "Clustering_data/female_clustering_labels_ward_20.joblib")

    df_male   = pd.read_csv(ROOT / "Datasets/male_dataset.csv")
    df_female = pd.read_csv(ROOT / "Datasets/female_dataset.csv")

    FEATURE_COLS = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Age",
        "Derived_HRV", "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP",
    ]

    male_features   = df_male[FEATURE_COLS].values
    female_features = df_female[FEATURE_COLS].values

    def build_centers(features, scaler, labels):
        scaled = scaler.transform(pd.DataFrame(features[: len(labels)], columns=FEATURE_COLS))
        centers = {}
        for cid in np.unique(labels):
            mask = labels == cid
            centers[cid] = scaled[mask].mean(axis=0)
        return centers

    male_centers   = build_centers(male_features,   scaler_male,   labels_male)
    female_centers = build_centers(female_features, scaler_female, labels_female)

    male_sizes   = pd.Series(labels_male).value_counts().to_dict()
    female_sizes = pd.Series(labels_female).value_counts().to_dict()

    def cluster_means(df, labels):
        sub = df[FEATURE_COLS].iloc[: len(labels)].copy()
        sub["_cluster"] = labels
        return sub.groupby("_cluster")[FEATURE_COLS].agg(["mean", "std", "min", "max"])

    male_means   = cluster_means(df_male,   labels_male)
    female_means = cluster_means(df_female, labels_female)

    return (
        scaler_male, scaler_female,
        male_centers, female_centers,
        male_sizes,   female_sizes,
        male_means,   female_means,
    )


(
    scaler_male, scaler_female,
    male_centers, female_centers,
    male_sizes,   female_sizes,
    male_means,   female_means,
) = load_artefacts()


# Helpers
FEATURE_COLS = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Age",
    "Derived_HRV", "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP",
]

def predict_cluster(feature_vec, scaler, centers):
    input_df      = pd.DataFrame([feature_vec], columns=FEATURE_COLS)
    scaled        = scaler.transform(input_df)
    center_ids    = list(centers.keys())
    center_matrix = np.array([centers[c] for c in center_ids])
    dists         = cdist(scaled, center_matrix, metric="euclidean")[0]
    best_idx      = int(np.argmin(dists))
    return center_ids[best_idx], dists[best_idx]


def get_health_assessment(patient_data: dict) -> str:
    cs = patient_data["cluster_stats"]  # dict keyed by feature -> {mean, std, min, max}

    def fmt(feat, unit="", normal=""):
        m  = cs[feat]["mean"]
        sd = cs[feat]["std"]
        lo = cs[feat]["min"]
        hi = cs[feat]["max"]
        line = f"  {feat}: {m:.2f} ± {sd:.2f} {unit}  [range: {lo:.2f}–{hi:.2f}]"
        if normal:
            line += f"  (normal: {normal})"
        return line

    prompt = f"""
You are a clinical health analyst reviewing a patient cluster's vital sign statistics.
Your job is to give a clear verdict on whether patients in this cluster are likely healthy or should seek medical help.

Follow this exact structure in your response:

1. Start with one of these two opening sentences, whichever applies:
   - "Your vitals look healthy. You do not need to seek medical attention at this time."
   - "Based on your vitals, you should seek medical attention."

2. Then explain why in plain English. If something is wrong, name exactly which vital is abnormal, what the normal range is, what the cluster average reads, and what condition or disease it could indicate. If everything is fine, briefly confirm which vitals are normal and why that is a good sign.

3. End with one clear sentence of advice — either to maintain their current lifestyle, or to see a doctor as soon as possible.

Rules:
- Write in second person ("your", "you").
- Do not use bullet points, headers, or lists. Write in paragraphs only.
- Do not mention AI, models, algorithms, or clusters.
- Keep the total response under 150 words.

Cluster Profile (Gender: {patient_data['gender']}, Cluster {patient_data['cluster_id']} of {patient_data['total_clusters']}, {patient_data['cluster_size']} patients):
{fmt('Heart Rate',             'bpm',        '60–100')}
{fmt('Respiratory Rate',       'breaths/min','12–20')}
{fmt('Body Temperature',       'C',          '36.1–37.2')}
{fmt('Oxygen Saturation',      '%',          '95–100')}
{fmt('Age',                    'years',      '')}
{fmt('Derived_HRV',            '',           '0.01–0.20')}
{fmt('Derived_Pulse_Pressure', 'mmHg',       '40–60')}
{fmt('Derived_BMI',            '',           '18.5–24.9')}
{fmt('Derived_MAP',            'mmHg',       '70–100')}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a5f 0%, #1e4d8c 60%, #1a6fa8 100%);
            border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
            box-shadow: 0 4px 24px rgba(30,58,95,0.25);">
    <div style="font-size: 1.75rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
        Human Vitals
    </div>
    <div style="font-size: 0.9rem; color: #93c5fd; margin-top: 0.3rem;">
        Patient vital signs analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Gender
st.markdown("**Patient Gender**")
gender = st.radio("Patient Gender", ["Male", "Female"], horizontal=True, label_visibility="collapsed")

st.divider()

# Inputs
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("Core Vitals")
    heart_rate       = st.number_input("Heart Rate (bpm)",               value=75,   step=1,   min_value=0)
    respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", value=16,   step=1,   min_value=0)
    body_temp        = st.number_input("Body Temperature (C)",           value=36.6, step=0.1, min_value=0.0, format="%.1f")
    oxygen_sat       = st.number_input("Oxygen Saturation (%)",          value=98,   step=1,   min_value=0,   max_value=100)
    age              = st.number_input("Age (years)",                    value=45,   step=1,   min_value=0)

with col2:
    st.subheader("Blood Pressure & BMI")
    systolic_bp  = st.number_input("Systolic BP (mmHg)",  value=120,  step=1,   min_value=0)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", value=80,   step=1,   min_value=0)
    bmi          = st.number_input("BMI (kg/m2)",         value=24.0, step=0.1, min_value=0.0, format="%.1f")

with col3:
    st.subheader("Heart Rate Variability")
    hrv = st.number_input("HRV", value=0.05, step=0.001, min_value=0.0, format="%.3f")

st.divider()

# Derived metrics
pulse_pressure = systolic_bp - diastolic_bp
map_value      = (systolic_bp + 2 * diastolic_bp) / 3

st.markdown("##### Derived Metrics")
d1, d2, d3 = st.columns(3)
d1.metric("Pulse Pressure (mmHg)",       f"{pulse_pressure}")
d2.metric("Mean Arterial Pressure (mmHg)", f"{map_value:.1f}")
d3.metric("BMI",                          f"{bmi:.1f}")

st.divider()

# Predict
if st.button("Run Analysis", type="primary", use_container_width=True):
    feature_vec = np.array([
        heart_rate, respiratory_rate, body_temp, oxygen_sat, age,
        hrv, pulse_pressure, bmi, map_value,
    ], dtype=float)

    if gender == "Male":
        scaler, centers, sizes, means = scaler_male,   male_centers,   male_sizes,   male_means
    else:
        scaler, centers, sizes, means = scaler_female, female_centers, female_sizes, female_means

    cluster_id, dist = predict_cluster(feature_vec, scaler, centers)

    # Health assessment
    st.divider()
    st.markdown("##### Health Assessment")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your-groq-api-key-here":
        st.warning("Add your Groq API key to the .env file to enable the health assessment.")
    else:
        with st.spinner("Loading..."):
            try:
                assessment = get_health_assessment({
                    "gender":         gender,
                    "cluster_id":     cluster_id,
                    "total_clusters": len(centers),
                    "cluster_size":   sizes.get(cluster_id, 0),
                    "cluster_stats":  {
                        feat: {
                            "mean": means.loc[cluster_id, (feat, "mean")],
                            "std":  means.loc[cluster_id, (feat, "std")],
                            "min":  means.loc[cluster_id, (feat, "min")],
                            "max":  means.loc[cluster_id, (feat, "max")],
                        }
                        for feat in FEATURE_COLS
                    },
                })
                st.markdown(f"""
                <div style="background: #ffffff; border: 1.5px solid #e2e8f0;
                            border-left: 5px solid #3b82f6; border-radius: 12px;
                            padding: 1.25rem 1.5rem; font-size: 0.97rem;
                            line-height: 1.75; color: #1e293b;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    {assessment}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not generate assessment: {e}")
