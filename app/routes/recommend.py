import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template, current_app

recommend = Blueprint("recommend", __name__)


def load_df():
    return pd.read_csv(current_app.config["DATA_PATH"])


@recommend.route("/recommend")
def recommend_page():
    return render_template("recommend.html")


# -- Drug Clustering (K-Means) ----------------------------------
@recommend.route("/api/recommend/cluster/<drugname>")
def api_cluster(drugname):
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = load_df()
        drugname = drugname.upper()

        if drugname not in df["drugname"].str.upper().values:
            return jsonify({"error": f"Drug not found: {drugname}"}), 404

        top_reactions = df["pt"].value_counts().head(50).index.tolist()
        top_drugs = df["drugname"].str.upper().value_counts().head(200).index.tolist()

        if drugname not in top_drugs:
            top_drugs = [drugname] + top_drugs[:199]

        df["drugname_upper"] = df["drugname"].str.upper()
        matrix = {}
        for drug in top_drugs:
            drug_df = df[df["drugname_upper"] == drug]
            vec = []
            for reac in top_reactions:
                count = len(drug_df[drug_df["pt"] == reac])
                total = len(drug_df) if len(drug_df) > 0 else 1
                vec.append(count / total)
            matrix[drug] = vec

        drug_list = list(matrix.keys())
        X = np.array([matrix[d] for d in drug_list])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        drug_idx = drug_list.index(drugname)
        drug_cluster = int(labels[drug_idx])

        same_cluster = [
            drug_list[i]
            for i in range(len(drug_list))
            if labels[i] == drug_cluster and drug_list[i] != drugname
        ]

        drug_vec = X_scaled[drug_idx]
        similarities = []
        for drug in same_cluster:
            idx = drug_list.index(drug)
            other_vec = X_scaled[idx]
            norm = np.linalg.norm(drug_vec) * np.linalg.norm(other_vec)
            sim = float(np.dot(drug_vec, other_vec) / norm) if norm > 0 else 0
            report_count = int(df[df["drugname_upper"] == drug].shape[0])
            similarities.append(
                {
                    "drug": drug,
                    "similarity": round(sim, 3),
                    "report_count": report_count,
                    "cluster": drug_cluster,
                }
            )

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_similar = similarities[:10]

        drug_df = df[df["drugname_upper"] == drugname]
        top_reac = drug_df["pt"].value_counts().head(5).to_dict()

        return jsonify(
            {
                "drug": drugname,
                "cluster": drug_cluster,
                "total_clusters": n_clusters,
                "similar_drugs": top_similar,
                "drug_top_reactions": [
                    {"reaction": k, "count": v} for k, v in top_reac.items()
                ],
                "report_count": int(len(drug_df)),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- Co-medication Analysis ------------------------------------
@recommend.route("/api/recommend/comedication/<drugname>")
def api_comedication(drugname):
    """동일 환자에서 함께 사용된 약물 Top 10 + 병용 시 주요 부작용"""
    try:
        df = load_df()
        drugname = drugname.upper()

        drug_ids = set(df[df["drugname"].str.upper() == drugname]["primaryid"])
        if len(drug_ids) == 0:
            return jsonify({"error": f"Drug not found: {drugname}"}), 404

        df_cases = df[df["primaryid"].isin(drug_ids)]
        co_drugs = (
            df_cases[df_cases["drugname"].str.upper() != drugname]["drugname"]
            .str.upper()
            .value_counts()
            .head(10)
        )

        serious_outcomes = {"DE", "HO", "LT"}
        result = []

        for co_drug, co_count in co_drugs.items():
            co_ids = set(df[df["drugname"].str.upper() == co_drug]["primaryid"])
            both_ids = drug_ids & co_ids

            if len(both_ids) == 0:
                continue

            df_both = df[df["primaryid"].isin(both_ids)]
            serious = df_both[df_both["outc_cod"].isin(serious_outcomes)][
                "primaryid"
            ].nunique()
            serious_rate = round(serious / len(both_ids) * 100, 1)

            top_reac = (
                df_both["pt"]
                .value_counts()
                .head(5)
                .reset_index()
                .rename(columns={"pt": "reaction", "count": "count"})
                .to_dict(orient="records")
            )

            co_rate = round(len(both_ids) / len(drug_ids) * 100, 1)

            result.append(
                {
                    "co_drug": co_drug,
                    "co_count": int(co_count),
                    "both_cases": int(len(both_ids)),
                    "co_rate": co_rate,
                    "serious_rate": serious_rate,
                    "top_reactions": top_reac,
                }
            )

        return jsonify(
            {
                "drug": drugname,
                "total_cases": int(len(drug_ids)),
                "co_medications": result,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
