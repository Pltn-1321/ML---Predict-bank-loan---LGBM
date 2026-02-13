"""Dashboard Streamlit de monitoring - Credit Scoring Platform."""

import json
import sys
import time
from pathlib import Path

# Ajouter la racine du projet au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring.drift import compute_drift_report, simulate_drift

# --- Configuration ---
st.set_page_config(
    page_title="HC Credit Risk | Monitoring",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="collapsed",
)

ARTIFACTS_DIR = Path("artifacts")
PREDICTIONS_LOG = Path("monitoring/predictions_log.csv")
REFERENCE_DATA = Path("data/test_preprocessed.csv")

# --- Theme premium bank ---
COLORS = {
    "primary": "#1B2A4A",
    "secondary": "#2C4A7C",
    "accent": "#C9A96E",
    "success": "#1D6A4B",
    "danger": "#8B2D2D",
    "approved": "#1D6A4B",
    "refused": "#8B2D2D",
    "chart_ref": "#1D6A4B",
    "chart_prod": "#8B2D2D",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background-color: #F4F1EC;
        font-family: 'Inter', sans-serif;
    }}

    .main-header {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
        padding: 2rem 2.5rem;
        border-radius: 0 0 16px 16px;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
    }}

    .main-header h1 {{
        color: white !important;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .main-header p {{
        color: {COLORS["accent"]};
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }}

    .section-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLORS["primary"]};
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS["accent"]};
    }}

    .status-badge {{
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    .status-ok {{
        background: rgba(29,106,75,0.12);
        color: {COLORS["success"]};
    }}

    .status-alert {{
        background: rgba(139,45,45,0.12);
        color: {COLORS["danger"]};
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(27,42,74,0.08);
        border: 1px solid rgba(27,42,74,0.06);
    }}

    div[data-testid="stMetric"] label {{
        color: #8B95A5 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500 !important;
    }}

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {COLORS["primary"]} !important;
        font-weight: 700 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: white;
        border-radius: 12px;
        padding: 0.3rem;
        box-shadow: 0 1px 3px rgba(27,42,74,0.08);
        gap: 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.85rem;
        color: {COLORS["primary"]};
        padding: 0.6rem 1.2rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLORS["primary"]} !important;
        color: white !important;
    }}

    .result-card {{
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(27,42,74,0.1);
        border: 1px solid rgba(27,42,74,0.06);
        text-align: center;
        margin: 1rem 0;
    }}

    .result-approved {{
        border-left: 5px solid {COLORS["success"]};
    }}

    .result-refused {{
        border-left: 5px solid {COLORS["danger"]};
    }}

    .result-card .decision {{
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    .result-card .proba {{
        font-size: 1rem;
        color: #8B95A5;
        margin-top: 0.3rem;
    }}

    .info-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(27,42,74,0.06);
        font-size: 0.9rem;
    }}

    .info-row .info-label {{
        color: #8B95A5;
        font-weight: 500;
    }}

    .info-row .info-value {{
        color: {COLORS["primary"]};
        font-weight: 600;
    }}

    .doc-section {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(27,42,74,0.08);
        border: 1px solid rgba(27,42,74,0.06);
        color: {COLORS["primary"]};
        line-height: 1.7;
    }}

    .doc-section h3 {{
        color: {COLORS["primary"]};
        font-weight: 700;
        margin-top: 0;
        border-bottom: 2px solid {COLORS["accent"]};
        padding-bottom: 0.4rem;
    }}

    .doc-section p, .doc-section li {{
        color: #3D4F6F;
        font-size: 0.92rem;
    }}

    .doc-section code {{
        background: #EDE9E0;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }}

    .stSidebar {{
        background: white;
        border-right: 1px solid rgba(27,42,74,0.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>Home Credit Risk Platform</h1>
        <p>Credit Scoring Model &mdash; Monitoring & Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load metadata ---
with open(ARTIFACTS_DIR / "model_metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata.get("optimal_threshold", 0.494)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["primary"], size=13),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(27,42,74,0.06)", zerolinecolor="rgba(27,42,74,0.1)"),
    yaxis=dict(gridcolor="rgba(27,42,74,0.06)", zerolinecolor="rgba(27,42,74,0.1)"),
)


@st.cache_resource
def load_model():
    """Charge le modele de scoring (cache Streamlit)."""
    import joblib

    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, feature_names


# === TABS ===
tab_predict, tab_scores, tab_perf, tab_drift, tab_model, tab_doc = st.tabs(
    ["Prediction", "Scores & Decisions", "Performance API", "Data Drift", "Modele", "Documentation"]
)

# ============================================================
# TAB : Prediction client
# ============================================================
with tab_predict:
    st.markdown(
        '<div class="section-title">Scoring Client</div>',
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown(
            "Entrez un **identifiant client** pour charger ses features depuis les donnees "
            "de test, ou saisissez des features manuellement."
        )

        input_mode = st.radio(
            "Mode de saisie",
            ["Par identifiant client", "Manuel (features)"],
            horizontal=True,
        )

        client_features = {}

        if input_mode == "Par identifiant client":
            client_id = st.number_input("SK_ID_CURR", min_value=0, value=100001, step=1)
            if REFERENCE_DATA.exists():
                ref_full = pd.read_csv(REFERENCE_DATA, nrows=5000)
                if "SK_ID_CURR" in ref_full.columns:
                    client_row = ref_full[ref_full["SK_ID_CURR"] == client_id]
                    if len(client_row) > 0:
                        client_features = (
                            client_row.drop("SK_ID_CURR", axis=1).iloc[0].to_dict()
                        )
                        st.success(
                            f"Client {client_id} trouve — {len(client_features)} features chargees"
                        )
                    else:
                        st.warning(
                            f"Client {client_id} introuvable dans les 5 000 premiers clients de test."
                        )
            else:
                st.warning("Fichier test_preprocessed.csv non disponible.")
        else:
            st.markdown("Saisissez les features au format `NOM: valeur` (une par ligne) :")
            manual_input = st.text_area(
                "Features (une par ligne)",
                value="AMT_CREDIT: 0.5\nAMT_ANNUITY: -0.3\nEXT_SOURCE_2: 0.7",
                height=150,
            )
            for line in manual_input.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    try:
                        client_features[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
            if client_features:
                st.info(f"{len(client_features)} features saisies")

        predict_btn = st.button("Lancer le scoring", type="primary", use_container_width=True)

    with col_result:
        if predict_btn and client_features:
            model, feature_names = load_model()

            start = time.perf_counter()
            df = pd.DataFrame([client_features])
            df = df.reindex(columns=feature_names, fill_value=0)
            proba = float(model.predict_proba(df)[:, 1][0])
            elapsed_ms = (time.perf_counter() - start) * 1000

            prediction = int(proba >= THRESHOLD)
            decision = "REFUSED" if prediction == 1 else "APPROVED"
            color = COLORS["danger"] if prediction == 1 else COLORS["success"]
            css_class = "result-refused" if prediction == 1 else "result-approved"

            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div class="decision" style="color: {color};">{decision}</div>
                    <div class="proba">Probabilite de defaut : <strong>{proba:.4f}</strong></div>
                    <div class="proba">Seuil : {THRESHOLD:.3f} &nbsp;|&nbsp; Inference : {elapsed_ms:.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba,
                number=dict(
                    font=dict(size=36, color=COLORS["primary"]),
                    valueformat=".4f",
                ),
                gauge=dict(
                    axis=dict(range=[0, 1], tickfont=dict(size=12)),
                    bar=dict(color=color, thickness=0.3),
                    bgcolor="white",
                    steps=[
                        dict(range=[0, THRESHOLD], color="rgba(29,106,75,0.1)"),
                        dict(range=[THRESHOLD, 1], color="rgba(139,45,45,0.1)"),
                    ],
                    threshold=dict(
                        line=dict(color=COLORS["accent"], width=3),
                        thickness=0.8,
                        value=THRESHOLD,
                    ),
                ),
            ))
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=30, r=30, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_gauge, width="stretch")

        elif predict_btn:
            st.warning("Aucune feature chargee. Verifiez l'identifiant ou la saisie.")

# ============================================================
# TAB : Distribution des scores
# ============================================================
with tab_scores:
    if PREDICTIONS_LOG.exists():
        logs = pd.read_csv(PREDICTIONS_LOG)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predictions", f"{len(logs):,}")
        col2.metric("Taux de refus", f"{logs['prediction'].mean():.1%}")
        col3.metric("Score moyen", f"{logs['probability'].mean():.3f}")
        col4.metric("Seuil", f"{THRESHOLD:.3f}")

        st.markdown(
            '<div class="section-title">Distribution des Scores</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        approved = logs[logs["prediction"] == 0]["probability"]
        refused = logs[logs["prediction"] == 1]["probability"]

        fig.add_trace(go.Histogram(
            x=approved, nbinsx=40, name="Approved",
            marker_color=COLORS["approved"], opacity=0.75,
        ))
        fig.add_trace(go.Histogram(
            x=refused, nbinsx=40, name="Refused",
            marker_color=COLORS["refused"], opacity=0.75,
        ))
        fig.add_vline(
            x=THRESHOLD, line_dash="dash", line_color=COLORS["accent"], line_width=2,
            annotation_text=f"  Seuil ({THRESHOLD:.3f})",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["accent"], size=12),
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Probabilite de defaut",
            yaxis_title="Nombre de predictions",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div class="section-title">Repartition</div>',
                unsafe_allow_html=True,
            )
            n_approved = int((logs["prediction"] == 0).sum())
            n_refused = int((logs["prediction"] == 1).sum())
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Approved", "Refused"],
                values=[n_approved, n_refused],
                marker=dict(colors=[COLORS["approved"], COLORS["refused"]]),
                hole=0.55,
                textinfo="percent+label",
                textfont=dict(size=14, color="white"),
                insidetextorientation="horizontal",
            )])
            fig_pie.update_layout(
                showlegend=False,
                height=350,
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            )
            st.plotly_chart(fig_pie, width="stretch")

        with col_b:
            st.markdown(
                '<div class="section-title">Volume temporel</div>',
                unsafe_allow_html=True,
            )
            logs["timestamp"] = pd.to_datetime(logs["timestamp"])
            logs_hourly = (
                logs.set_index("timestamp")
                .resample("h")["prediction"]
                .agg(["count", "mean"])
            )
            if len(logs_hourly) > 1:
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=logs_hourly.index, y=logs_hourly["count"],
                    mode="lines+markers", name="Volume",
                    line=dict(color=COLORS["secondary"], width=2),
                    marker=dict(size=5),
                ))
                fig_time.update_layout(
                    xaxis_title="",
                    yaxis_title="Predictions / heure",
                    height=350,
                    showlegend=False,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_time, width="stretch")
            else:
                st.info("Donnees temporelles insuffisantes.")
    else:
        st.info("En attente de predictions. Lancez l'API et envoyez des requetes.")

# ============================================================
# TAB : Performance API
# ============================================================
with tab_perf:
    if PREDICTIONS_LOG.exists():
        logs = pd.read_csv(PREDICTIONS_LOG)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latence moyenne", f"{logs['inference_time_ms'].mean():.1f} ms")
        col2.metric("Latence P50", f"{logs['inference_time_ms'].median():.1f} ms")
        col3.metric("Latence P95", f"{logs['inference_time_ms'].quantile(0.95):.1f} ms")
        col4.metric("Latence max", f"{logs['inference_time_ms'].max():.1f} ms")

        st.markdown(
            '<div class="section-title">Distribution de la Latence</div>',
            unsafe_allow_html=True,
        )

        fig_lat = go.Figure()
        fig_lat.add_trace(go.Histogram(
            x=logs["inference_time_ms"], nbinsx=40,
            marker_color=COLORS["secondary"], opacity=0.8,
        ))
        p95 = logs["inference_time_ms"].quantile(0.95)
        fig_lat.add_vline(
            x=p95, line_dash="dash", line_color=COLORS["danger"],
            annotation_text=f"  P95 ({p95:.1f} ms)",
            annotation_position="top right",
            annotation_font=dict(size=12),
        )
        fig_lat.update_layout(
            xaxis_title="Temps d'inference (ms)",
            yaxis_title="Nombre de requetes",
            showlegend=False,
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_lat, width="stretch")

        st.markdown(
            '<div class="section-title">Latence dans le temps</div>',
            unsafe_allow_html=True,
        )

        logs["timestamp"] = pd.to_datetime(logs["timestamp"])
        mean_lat = logs["inference_time_ms"].mean()
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=logs["timestamp"], y=logs["inference_time_ms"],
            mode="markers",
            marker=dict(color=COLORS["secondary"], size=4, opacity=0.4),
        ))
        fig_ts.add_hline(
            y=mean_lat, line_dash="dash", line_color=COLORS["accent"],
            annotation_text=f"  Moyenne : {mean_lat:.1f} ms",
            annotation_position="top left",
            annotation_font=dict(size=12, color=COLORS["accent"]),
        )
        fig_ts.update_layout(
            xaxis_title="",
            yaxis_title="Latence (ms)",
            showlegend=False,
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_ts, width="stretch")
    else:
        st.info("En attente de donnees de performance.")

# ============================================================
# TAB : Data Drift
# ============================================================
with tab_drift:
    if REFERENCE_DATA.exists():
        st.sidebar.markdown("### Simulation du Drift")

        drift_type = st.sidebar.selectbox(
            "Type de drift",
            ["none", "gradual", "sudden", "feature_shift"],
            index=1,
            format_func=lambda x: {
                "none": "Aucun drift",
                "gradual": "Drift graduel",
                "sudden": "Drift soudain",
                "feature_shift": "Shift de features",
            }[x],
        )
        intensity = st.sidebar.slider("Intensite", 0.0, 1.0, 0.3, 0.05)
        n_samples = st.sidebar.slider("Echantillons", 100, 5000, 1000, 100)

        ref_data = pd.read_csv(REFERENCE_DATA, nrows=n_samples)
        if "SK_ID_CURR" in ref_data.columns:
            ref_data = ref_data.drop("SK_ID_CURR", axis=1)

        prod_data = simulate_drift(ref_data, drift_type=drift_type, intensity=intensity)
        drift_report = compute_drift_report(ref_data, prod_data, top_n=20)

        n_drifted = int(drift_report["drift_detected"].sum())
        n_total = len(drift_report)
        drift_pct = n_drifted / n_total * 100 if n_total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Features analysees", n_total)
        col2.metric("Features en drift", n_drifted)
        col3.metric("Taux de drift", f"{drift_pct:.0f}%")

        if drift_pct > 30:
            st.markdown(
                '<span class="status-badge status-alert">ALERTE — Drift significatif detecte</span>',
                unsafe_allow_html=True,
            )
        elif drift_pct > 0:
            st.markdown(
                '<span class="status-badge status-ok">Drift modere — Surveillance recommandee</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-ok">Aucun drift detecte</span>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title">Top Features par Drift (KS Statistic)</div>',
            unsafe_allow_html=True,
        )

        # Tronquer les noms longs pour la lisibilite du graphe
        display_report = drift_report.copy()
        display_report["feature_short"] = display_report["feature"].apply(
            lambda x: x[:28] + "..." if len(x) > 28 else x
        )

        fig_ks = go.Figure()
        fig_ks.add_trace(go.Bar(
            y=display_report["feature_short"],
            x=display_report["ks_statistic"],
            orientation="h",
            marker_color=[
                COLORS["danger"] if d else COLORS["secondary"]
                for d in display_report["drift_detected"]
            ],
            opacity=0.85,
            text=[f"{v:.3f}" for v in display_report["ks_statistic"]],
            textposition="outside",
            textfont=dict(size=11),
            hovertext=display_report["feature"],
        ))
        fig_ks.update_layout(
            yaxis=dict(
                autorange="reversed",
                gridcolor="rgba(27,42,74,0.06)",
                tickfont=dict(size=11),
            ),
            xaxis_title="KS Statistic",
            height=max(400, n_total * 28),
            showlegend=False,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
        )
        st.plotly_chart(fig_ks, width="stretch")

        # Distributions comparees top 3
        top_drifted = drift_report[drift_report["drift_detected"]].head(3)
        if len(top_drifted) > 0:
            st.markdown(
                '<div class="section-title">Distributions Comparees (Reference vs Production)</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(top_drifted), 3))
            for idx, (_, row) in enumerate(top_drifted.iterrows()):
                feat = row["feature"]
                with cols[idx]:
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Histogram(
                        x=ref_data[feat], name="Reference",
                        opacity=0.6, marker_color=COLORS["chart_ref"], nbinsx=30,
                    ))
                    fig_comp.add_trace(go.Histogram(
                        x=prod_data[feat], name="Production",
                        opacity=0.6, marker_color=COLORS["chart_prod"], nbinsx=30,
                    ))
                    short_name = feat[:20] + "..." if len(feat) > 20 else feat
                    fig_comp.update_layout(
                        title=dict(
                            text=(
                                f"<b>{short_name}</b><br>"
                                f"<span style='font-size:11px;color:#8B95A5;'>"
                                f"KS = {row['ks_statistic']:.3f}</span>"
                            ),
                            font=dict(size=13),
                        ),
                        barmode="overlay",
                        height=300,
                        showlegend=idx == 0,
                        legend=dict(orientation="h", y=1.2, font=dict(size=11)),
                        margin=dict(l=30, r=10, t=70, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", size=11),
                    )
                    st.plotly_chart(fig_comp, width="stretch")
    else:
        st.info("Fichier de reference non disponible.")

# ============================================================
# TAB : Info Modele
# ============================================================
with tab_model:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-title">Performance du Modele</div>',
            unsafe_allow_html=True,
        )
        info_perf = [
            ("Modele", metadata.get("best_model_name", "N/A")),
            ("Seuil optimal", f"{metadata.get('optimal_threshold', 0):.4f}"),
            ("Business Cost (optimal)", f"{metadata.get('business_cost_optimal', 0):.4f}"),
            ("Nombre de features", str(metadata.get("n_features", "N/A"))),
        ]
        for label, value in info_perf:
            st.markdown(
                f'<div class="info-row">'
                f'<span class="info-label">{label}</span>'
                f'<span class="info-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            '<div class="section-title">Parametres Metier</div>',
            unsafe_allow_html=True,
        )
        dist = metadata.get("class_distribution", {})
        total = sum(int(v) for v in dist.values()) if dist else 0
        default_rate = int(dist.get("1", 0)) / total if total > 0 else 0

        info_biz = [
            ("Cout Faux Negatif (FN)", f"{metadata.get('cost_fn', 'N/A')}x"),
            ("Cout Faux Positif (FP)", f"{metadata.get('cost_fp', 'N/A')}x"),
            ("Echantillon d'entrainement", f"{metadata.get('n_train_samples', 0):,} clients"),
            ("Taux de defaut historique", f"{default_rate:.2%}"),
        ]
        for label, value in info_biz:
            st.markdown(
                f'<div class="info-row">'
                f'<span class="info-label">{label}</span>'
                f'<span class="info-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    with st.expander("Configuration complete (JSON)"):
        st.json(metadata)

# ============================================================
# TAB : Documentation
# ============================================================
with tab_doc:
    st.markdown("""
    <div class="doc-section">
        <h3>A propos de cette application</h3>
        <p>
            Cette plateforme est le <strong>dashboard de monitoring</strong> du modele de scoring
            credit developpe pour <strong>Home Credit</strong>. Elle permet aux equipes metier et
            data science de suivre en temps reel le comportement du modele en production.
        </p>
        <p>
            Le modele predit la <strong>probabilite de defaut de paiement</strong> d'un client
            a partir de ses donnees financieres et comportementales. Si cette probabilite depasse
            le seuil optimal (<strong>0.494</strong>), le credit est refuse.
        </p>
    </div>

    <div class="doc-section">
        <h3>Comment utiliser le dashboard</h3>
        <ul>
            <li><strong>Prediction</strong> : Testez le modele en temps reel. Entrez un identifiant
            client existant ou saisissez des features manuellement pour obtenir une decision
            instantanee (APPROVED / REFUSED) avec la probabilite de defaut et une jauge visuelle.</li>
            <li><strong>Scores & Decisions</strong> : Visualisez la distribution des probabilites
            predites, le taux de refus global, et le volume de predictions dans le temps.</li>
            <li><strong>Performance API</strong> : Surveillez la latence de l'API (temps de reponse
            moyen, P50, P95, max) pour detecter d'eventuelles degradations.</li>
            <li><strong>Data Drift</strong> : Simulez et detectez le data drift en comparant les
            donnees de production aux donnees de reference. Utilisez la sidebar pour configurer
            le type de drift, l'intensite et le nombre d'echantillons.</li>
            <li><strong>Modele</strong> : Consultez les parametres du modele, le seuil de decision,
            les couts metier (FN/FP) et la configuration complete.</li>
        </ul>
    </div>

    <div class="doc-section">
        <h3>Qu'est-ce que le Data Drift ?</h3>
        <p>
            Le <strong>data drift</strong> (derive des donnees) se produit lorsque la distribution
            statistique des donnees en production <strong>diverge significativement</strong> de celle
            des donnees d'entrainement. C'est un phenomene naturel et inevitable en machine learning.
        </p>
        <p><strong>Causes courantes :</strong></p>
        <ul>
            <li><strong>Changement de population</strong> : nouveaux segments de clients, evolution
            demographique, expansion geographique.</li>
            <li><strong>Evolution economique</strong> : crise financiere, inflation, changement de
            politique de credit.</li>
            <li><strong>Probleme technique</strong> : bug dans le pipeline de donnees, changement de
            format, feature manquante.</li>
            <li><strong>Saisonnalite</strong> : comportements differents selon les periodes de l'annee.</li>
        </ul>
        <p><strong>Pourquoi c'est critique :</strong></p>
        <p>
            Un modele entraine sur des donnees de 2024 peut perdre en performance si les clients de
            2025 ont un profil different. Sans monitoring, cette degradation passe inapercue et le
            modele peut prendre des decisions inadaptees (accorder des credits risques ou refuser
            des bons clients).
        </p>
    </div>

    <div class="doc-section">
        <h3>Comment on detecte le drift</h3>
        <p>
            Nous utilisons le <strong>test de Kolmogorov-Smirnov (KS)</strong> pour chaque feature.
            Ce test statistique compare deux distributions et retourne :
        </p>
        <ul>
            <li><strong>KS Statistic</strong> : mesure la distance maximale entre les deux
            distributions cumulatives. Plus la valeur est elevee, plus le drift est important.</li>
            <li><strong>p-value</strong> : probabilite que les deux echantillons proviennent de la
            meme distribution. Si <code>p-value &lt; 0.05</code>, on considere que le drift est
            statistiquement significatif.</li>
        </ul>
        <p>
            L'onglet Data Drift permet de <strong>simuler</strong> differents types de drift
            (graduel, soudain, shift de features) pour comprendre comment le monitoring reagit.
            En production, ces metriques seraient calculees automatiquement sur les vraies donnees
            entrantes.
        </p>
    </div>

    <div class="doc-section">
        <h3>Pourquoi monitorer un modele ML</h3>
        <p>
            Deployer un modele ne suffit pas. En production, il faut surveiller en continu :
        </p>
        <ul>
            <li><strong>La qualite des predictions</strong> : le modele continue-t-il a bien predire ?</li>
            <li><strong>La stabilite des donnees</strong> : les features en entree restent-elles coherentes
            avec les donnees d'entrainement ?</li>
            <li><strong>La performance technique</strong> : l'API repond-elle dans des delais acceptables ?</li>
            <li><strong>Le volume d'utilisation</strong> : y a-t-il des pics anormaux ou des baisses
            d'activite ?</li>
        </ul>
        <p>
            Ce monitoring permet de detecter rapidement les problemes, de declencher un
            <strong>re-entrainement</strong> du modele si necessaire, et de maintenir la confiance
            des equipes metier dans le systeme de scoring.
        </p>
    </div>

    <div class="doc-section">
        <h3>Architecture technique</h3>
        <ul>
            <li><strong>Modele</strong> : LightGBM (Gradient Boosting) entraine sur 307k clients,
            419 features, avec fonction de cout metier asymetrique (FN = 10x, FP = 1x).</li>
            <li><strong>API</strong> : FastAPI (Python) servant les predictions via
            <code>POST /predict</code>.</li>
            <li><strong>Monitoring</strong> : Streamlit (ce dashboard) consommant les logs de
            predictions et les donnees de reference.</li>
            <li><strong>Detection du drift</strong> : Test KS (Kolmogorov-Smirnov) via SciPy,
            avec simulation de drift pour demonstration.</li>
            <li><strong>CI/CD</strong> : GitHub Actions (test, build Docker, deploy Render).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
