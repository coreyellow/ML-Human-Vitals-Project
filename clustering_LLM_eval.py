import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import time

def summarize_clusters_with_llm(
    pca, scaler, labels, pca_csv_path, feature_cols,
    gender, linkage, clusters, groq_client
):
    print("\n" + "="*70)
    print(f"{gender.upper()} CLUSTER STATISTICS + LLM")
    print("="*70)

    df_pca = pd.read_csv(pca_csv_path)

    # Safety check
    if len(labels) != len(df_pca):
        raise ValueError("Mismatch between labels and PCA rows")

    # Reconstruct
    X_scaled = pca.inverse_transform(df_pca.values)
    X_original = scaler.inverse_transform(X_scaled)

    df_original = pd.DataFrame(X_original, columns=feature_cols)
    df_original["cluster"] = labels

    grouped = df_original.groupby("cluster")

    output_path = f"Clustering_data/LLM_{linkage}_{clusters}_{gender}.txt"

    # Clear file first
    with open(output_path, "w") as f:
        f.write(f"LLM Cluster Analysis ({linkage}, k={clusters})\n\n")

    # Helper formatter
    def fmt(cs, feat, unit="", normal=""):
        m  = cs[feat]["mean"]
        sd = cs[feat]["std"]
        lo = cs[feat]["min"]
        hi = cs[feat]["max"]

        line = f"  {feat}: {m:.2f} +- {sd:.2f} {unit}  [range: {lo:.2f}-{hi:.2f}]"
        if normal:
            line += f"  (normal: {normal})"
        return line

    for cluster_id, group in grouped:
        print(f"\nProcessing Cluster {cluster_id} ({gender})...")

        # Build stats dict
        stats = group[feature_cols].agg(["mean", "min", "max", "std"])
        cs = {
            feat: {
                "mean": stats.loc["mean", feat],
                "min":  stats.loc["min", feat],
                "max":  stats.loc["max", feat],
                "std":  stats.loc["std", feat],
            }
            for feat in feature_cols
        }

        patient_data = {
            "gender": gender,
            "cluster_id": cluster_id,
            "total_clusters": clusters,
            "cluster_size": len(group),
        }

        # Build prompt
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
            {fmt(cs, 'Heart Rate',             'bpm',        '60-100')}
            {fmt(cs, 'Respiratory Rate',       'breaths/min','12-20')}
            {fmt(cs, 'Body Temperature',       'C',          '36.1-37.2')}
            {fmt(cs, 'Oxygen Saturation',      '%',          '95-100')}
            {fmt(cs, 'Age',                    'years',      '')}
            {fmt(cs, 'Derived_HRV',            '',           '0.01-0.20')}
            {fmt(cs, 'Derived_Pulse_Pressure', 'mmHg',       '40-60')}
            {fmt(cs, 'Derived_BMI',            '',           '18.5-24.9')}
            {fmt(cs, 'Derived_MAP',            'mmHg',       '70-100')}
        """

        # Call Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        # Delay between API calls
        time.sleep(1)

        reply = response.choices[0].message.content.strip()

        # Save to file
        with open(output_path, "a") as f:
            f.write(f"--- Gender: {gender} | Cluster {cluster_id} ---\n")
            f.write(f"Cluster Size: {patient_data['cluster_size']}\n")
            f.write(fmt(cs, 'Heart Rate', 'bpm', '60–100') + "\n")
            f.write(fmt(cs, 'Respiratory Rate', 'breaths/min', '12–20') + "\n")
            f.write(fmt(cs, 'Body Temperature', 'C', '36.1–37.2') + "\n")
            f.write(fmt(cs, 'Oxygen Saturation', '%', '95–100') + "\n")
            f.write(fmt(cs, 'Age', 'years', '') + "\n")
            f.write(fmt(cs, 'Derived_HRV', '', '0.01–0.20') + "\n")
            f.write(fmt(cs, 'Derived_Pulse_Pressure', 'mmHg', '40–60') + "\n")
            f.write(fmt(cs, 'Derived_BMI', '', '18.5–24.9') + "\n")
            f.write(fmt(cs, 'Derived_MAP', 'mmHg', '70–100') + "\n\n")
            f.write(reply + "\n\n")

    print(f"\nSaved LLM results → {output_path}")

if __name__ == "__main__":
    # Which clustering hyperparameters to evaluate
    linkage = "ward"
    clusters = 50
    
    scaler_male = joblib.load("Clustering_data/scaler_male.pkl")
    scaler_female = joblib.load("Clustering_data/scaler_female.pkl")
    
    pca_male = joblib.load("Clustering_data/pca_male.pkl")
    pca_female = joblib.load("Clustering_data/pca_female.pkl")
    
    labels_male = joblib.load(f"Clustering_data/male_clustering_labels_{linkage}_{clusters}.joblib")
    labels_female = joblib.load(f"Clustering_data/female_clustering_labels_{linkage}_{clusters}.joblib")

    FEATURE_COLS = [
        "Heart Rate", "Respiratory Rate", "Body Temperature",
        "Oxygen Saturation", "Age",
        "Derived_HRV", "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP",
    ]
    
    # Load API key
    ROOT = Path(__file__).parent
    load_dotenv(ROOT / ".env")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    summarize_clusters_with_llm(
        pca_male, scaler_male, labels_male,
        "PCA_output/HVS_PCA_male.csv",
        FEATURE_COLS, "male",
        linkage, clusters,
        groq_client
    )
    
    summarize_clusters_with_llm(
        pca_female, scaler_female, labels_female,
        "PCA_output/HVS_PCA_female.csv",
        FEATURE_COLS, "female",
        linkage, clusters,
        groq_client
    )
    
    
    
    