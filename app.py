import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency at runtime
    shap = None

from src.ufc_predictor.config import (
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REPORT_PATH,
    TARGET_COLUMN,
)
from src.ufc_predictor.features import align_feature_columns, build_inference_frame
from src.ufc_predictor.train import main as train_pipeline


st.set_page_config(
    page_title="Who Is More Likely To Win?",
    page_icon="🥊",
    layout="wide",
)

st.markdown(
    """
<style>
    .red-card  { background:#1a0000; border:1px solid #c0392b; border-radius:10px; padding:16px; }
    .blue-card { background:#00001a; border:1px solid #2980b9; border-radius:10px; padding:16px; }
    .vs-badge  { font-size:2rem; font-weight:900; color:#f0b429; text-align:center; padding-top:28px; }
    .result-box { border-radius:12px; padding:24px; text-align:center; margin-top:16px; }
    .winner-red  { background:linear-gradient(135deg,#7b0000,#c0392b); }
    .winner-blue { background:linear-gradient(135deg,#00007b,#2980b9); }
    [data-testid="metric-container"] { background:#1a1a2e; border-radius:8px; padding:10px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    if not MODEL_PATH.exists():
        with st.spinner("Preparing the model for the first time. Please wait..."):
            train_pipeline()

    if not MODEL_PATH.exists():
        return None, None, None

    bundle = joblib.load(MODEL_PATH)
    processed = (
        pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["fight_date"])
        if PROCESSED_DATA_PATH.exists()
        else None
    )
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    return bundle, processed, report


@st.cache_resource(show_spinner=False)
def build_shap_explainer(_model, feature_columns, background_df):
    if shap is None or background_df is None or len(background_df) == 0:
        return None

    background = background_df[feature_columns].copy()

    def predict_red_probability(data):
        frame = pd.DataFrame(data, columns=feature_columns)
        return _model.predict_proba(frame)[:, 1]

    return shap.Explainer(
        predict_red_probability,
        background,
        feature_names=feature_columns,
        algorithm="permutation",
    )


@st.cache_data(show_spinner=False)
def load_fighter_catalog():
    if not RAW_DATA_PATH.exists():
        return [], {}

    raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["fight_date"])

    red_view = raw_df[
        [
            "fight_date",
            "fighter_red",
            "red_age",
            "red_height_cm",
            "red_reach_cm",
            "red_wins",
            "red_losses",
            "red_sig_str_acc",
            "red_takedown_acc",
            "red_stance",
        ]
    ].rename(
        columns={
            "fighter_red": "fighter_name",
            "red_age": "age",
            "red_height_cm": "height_cm",
            "red_reach_cm": "reach_cm",
            "red_wins": "wins",
            "red_losses": "losses",
            "red_sig_str_acc": "sig_str_acc",
            "red_takedown_acc": "takedown_acc",
            "red_stance": "stance",
        }
    )

    blue_view = raw_df[
        [
            "fight_date",
            "fighter_blue",
            "blue_age",
            "blue_height_cm",
            "blue_reach_cm",
            "blue_wins",
            "blue_losses",
            "blue_sig_str_acc",
            "blue_takedown_acc",
            "blue_stance",
        ]
    ].rename(
        columns={
            "fighter_blue": "fighter_name",
            "blue_age": "age",
            "blue_height_cm": "height_cm",
            "blue_reach_cm": "reach_cm",
            "blue_wins": "wins",
            "blue_losses": "losses",
            "blue_sig_str_acc": "sig_str_acc",
            "blue_takedown_acc": "takedown_acc",
            "blue_stance": "stance",
        }
    )

    fighter_history = pd.concat([red_view, blue_view], ignore_index=True)
    fighter_history = fighter_history.dropna(subset=["fighter_name"]).sort_values("fight_date")
    latest_profile = fighter_history.groupby("fighter_name", as_index=False).tail(1)
    latest_profile = latest_profile.sort_values("fighter_name").reset_index(drop=True)

    fighter_names = latest_profile["fighter_name"].tolist()
    fighter_profiles = latest_profile.set_index("fighter_name").to_dict(orient="index")
    return fighter_names, fighter_profiles


def pretty_feature_name(feature_name: str) -> str:
    labels = {
        "age_diff": "Age difference",
        "height_cm_diff": "Height difference",
        "reach_cm_diff": "Reach difference",
        "wins_diff": "Wins difference",
        "losses_diff": "Losses difference",
        "sig_str_acc_diff": "Striking accuracy difference",
        "takedown_acc_diff": "Takedown accuracy difference",
        "win_rate_diff": "Win rate difference",
        "same_stance": "Same stance",
        "fight_month": "Fight month",
        "fight_year": "Fight year",
        "red_win_rate": "Red corner win rate",
        "blue_win_rate": "Blue corner win rate",
    }
    return labels.get(feature_name, feature_name.replace("_", " ").title())


def resolve_fighter_inputs(selected_name: str, prefix: str, defaults: dict):
    custom_mode = selected_name == "Fill manually"
    fighter_name = (
        st.text_input("Name", defaults["name"], key=f"name_{prefix}")
        if custom_mode
        else selected_name
    )
    age = st.number_input("Age", 18, 50, int(round(defaults["age"])), key=f"age_{prefix}")
    height_cm = st.number_input("Height (cm)", 150, 220, int(round(defaults["height_cm"])), key=f"ht_{prefix}")
    reach_cm = st.number_input(
        "Reach (cm)", 150, 240, int(round(defaults["reach_cm"])), key=f"rc_{prefix}"
    )
    wins = st.number_input("Wins", 0, 60, int(round(defaults["wins"])), key=f"w_{prefix}")
    losses = st.number_input("Losses", 0, 40, int(round(defaults["losses"])), key=f"l_{prefix}")
    sig_str_acc = st.slider(
        "Striking accuracy", 0.0, 1.0, float(defaults["sig_str_acc"]), 0.01, key=f"sig_{prefix}"
    )
    takedown_acc = st.slider(
        "Takedown accuracy", 0.0, 1.0, float(defaults["takedown_acc"]), 0.01, key=f"td_{prefix}"
    )
    stance_options = ["Orthodox", "Southpaw", "Switch", "Open Stance", "Unknown"]
    default_stance = defaults["stance"] if defaults["stance"] in stance_options else "Unknown"
    stance = st.selectbox("Stance", stance_options, index=stance_options.index(default_stance), key=f"st_{prefix}")

    return {
        "fighter_name": fighter_name,
        "age": age,
        "height_cm": height_cm,
        "reach_cm": reach_cm,
        "wins": wins,
        "losses": losses,
        "sig_str_acc": sig_str_acc,
        "takedown_acc": takedown_acc,
        "stance": stance,
    }


bundle, processed_df, report_text = load_artifacts()
fighter_names, fighter_profiles = load_fighter_catalog()

st.title("🥊 Who Is More Likely To Win?")
st.caption("Compare both athletes and see who the model favors before the fight.")

if bundle is None:
    st.error(
        "The model could not be prepared automatically. "
        "Run `python -m src.ufc_predictor.train` in the terminal."
    )
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
metrics = bundle["metrics"]
baseline_metrics = bundle.get("baseline_metrics", {})
shap_background = bundle.get("shap_background")
shap_explainer = build_shap_explainer(model, feature_columns, shap_background)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Model accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("📈 ROC AUC", f"{metrics['roc_auc']:.3f}")
c3.metric("📉 Log Loss", f"{metrics['log_loss']:.3f}")
c4.metric("⚙️ Features", len(feature_columns))

st.divider()

tab1, tab2, tab3 = st.tabs(["🥊 Fight simulator", "📊 Processed data", "📋 Model report"])

with tab1:
    st.subheader("Build the matchup")
    st.caption("Choose real fighters from the dataset or use `Fill manually`.")

    col_red, col_vs, col_blue = st.columns([5, 1, 5])
    fighter_options = ["Fill manually"] + fighter_names

    with col_red:
        st.markdown('<div class="red-card">', unsafe_allow_html=True)
        st.markdown("### 🔴 Red corner fighter")
        selected_red = st.selectbox("Fighter", fighter_options, index=0, key="fighter_select_red")
        red_defaults = fighter_profiles.get(
            selected_red,
            {
                "name": "Red corner fighter",
                "age": 29,
                "height_cm": 180,
                "reach_cm": 185,
                "wins": 12,
                "losses": 3,
                "sig_str_acc": 0.48,
                "takedown_acc": 0.35,
                "stance": "Orthodox",
            },
        )
        if selected_red != "Fill manually":
            red_defaults = {"name": selected_red, **red_defaults}
        red_inputs = resolve_fighter_inputs(selected_red, "r", red_defaults)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_vs:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)

    with col_blue:
        st.markdown('<div class="blue-card">', unsafe_allow_html=True)
        st.markdown("### 🔵 Blue corner fighter")
        selected_blue = st.selectbox("Fighter", fighter_options, index=0, key="fighter_select_blue")
        blue_defaults = fighter_profiles.get(
            selected_blue,
            {
                "name": "Blue corner fighter",
                "age": 28,
                "height_cm": 178,
                "reach_cm": 182,
                "wins": 10,
                "losses": 4,
                "sig_str_acc": 0.45,
                "takedown_acc": 0.31,
                "stance": "Orthodox",
            },
        )
        if selected_blue != "Fill manually":
            blue_defaults = {"name": selected_blue, **blue_defaults}
        blue_inputs = resolve_fighter_inputs(selected_blue, "b", blue_defaults)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    predict_btn = st.button("⚡ Get prediction", type="primary", use_container_width=True)

    if predict_btn:
        fighter_red = red_inputs["fighter_name"]
        fighter_blue = blue_inputs["fighter_name"]

        if fighter_red.strip() == fighter_blue.strip():
            st.warning("Both sides have the same name. Please check before continuing.")
        else:
            with st.spinner("Calculating prediction..."):
                inference_df = build_inference_frame(
                    fight_date=pd.Timestamp.today().normalize(),
                    fighter_red=fighter_red,
                    fighter_blue=fighter_blue,
                    red_age=red_inputs["age"],
                    blue_age=blue_inputs["age"],
                    red_height_cm=red_inputs["height_cm"],
                    blue_height_cm=blue_inputs["height_cm"],
                    red_reach_cm=red_inputs["reach_cm"],
                    blue_reach_cm=blue_inputs["reach_cm"],
                    red_wins=red_inputs["wins"],
                    blue_wins=blue_inputs["wins"],
                    red_losses=red_inputs["losses"],
                    blue_losses=blue_inputs["losses"],
                    red_sig_str_acc=red_inputs["sig_str_acc"],
                    blue_sig_str_acc=blue_inputs["sig_str_acc"],
                    red_takedown_acc=red_inputs["takedown_acc"],
                    blue_takedown_acc=blue_inputs["takedown_acc"],
                    red_stance=red_inputs["stance"],
                    blue_stance=blue_inputs["stance"],
                )
                X = align_feature_columns(inference_df, feature_columns)
                red_prob = float(model.predict_proba(X)[0, 1])
                blue_prob = 1.0 - red_prob
                red_wins_flag = red_prob >= 0.5
                winner = fighter_red if red_wins_flag else fighter_blue
                win_prob = red_prob if red_wins_flag else blue_prob
                css_class = "winner-red" if red_wins_flag else "winner-blue"
                emoji = "🔴" if red_wins_flag else "🔵"

            st.markdown(
                f'<div class="result-box {css_class}">'
                f"<h2>{emoji} Model pick: {winner}</h2>"
                f'<p style="font-size:1.2rem">Estimated win chance: <strong>{win_prob:.1%}</strong></p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption("Estimated probability split between both sides")

            red_w = max(int(red_prob * 100), 1)
            blue_w = max(int(blue_prob * 100), 1)
            bar1, bar2 = st.columns([red_w, blue_w])
            with bar1:
                st.markdown(
                    (
                        "<div style='background:#c0392b;border-radius:6px 0 0 6px;"
                        "padding:8px;text-align:center;color:white;font-weight:bold'>"
                        f"🔴 {red_prob:.1%}</div>"
                    ),
                    unsafe_allow_html=True,
                )
            with bar2:
                st.markdown(
                    (
                        "<div style='background:#2980b9;border-radius:0 6px 6px 0;"
                        "padding:8px;text-align:center;color:white;font-weight:bold'>"
                        f"🔵 {blue_prob:.1%}</div>"
                    ),
                    unsafe_allow_html=True,
                )

            with st.expander("🔍 Fighter comparison"):
                attrs = {
                    "Age": (red_inputs["age"], blue_inputs["age"], False),
                    "Height (cm)": (red_inputs["height_cm"], blue_inputs["height_cm"], True),
                    "Reach (cm)": (red_inputs["reach_cm"], blue_inputs["reach_cm"], True),
                    "Wins": (red_inputs["wins"], blue_inputs["wins"], True),
                    "Losses": (red_inputs["losses"], blue_inputs["losses"], False),
                    "Striking accuracy": (red_inputs["sig_str_acc"], blue_inputs["sig_str_acc"], True),
                    "Takedown accuracy": (red_inputs["takedown_acc"], blue_inputs["takedown_acc"], True),
                }
                rows = []
                for label, (rv, bv, higher_is_better) in attrs.items():
                    diff = rv - bv
                    if diff == 0:
                        adv = "-"
                    elif (diff > 0) == higher_is_better:
                        adv = "🔴"
                    else:
                        adv = "🔵"
                    rows.append(
                        {
                            "Attribute": label,
                            "🔴 Red corner": rv,
                            "🔵 Blue corner": bv,
                            "Difference": f"{diff:+.3g}",
                            "Edge": adv,
                        }
                    )
                st.dataframe(pd.DataFrame(rows).set_index("Attribute"), use_container_width=True)

            with st.expander("🧠 Why the model leaned this way"):
                st.caption("SHAP values show which attributes increased or decreased the red corner win probability.")
                if shap_explainer is None:
                    st.info(
                        "SHAP is not available in the current environment yet. "
                        "Install dependencies again with `pip install -r requirements.txt`."
                    )
                else:
                    shap_values = shap_explainer(X, max_evals=2 * len(feature_columns) + 1)
                    contributions = pd.DataFrame(
                        {
                            "feature": feature_columns,
                            "shap_value": shap_values.values[0],
                            "feature_value": X.iloc[0].values,
                        }
                    )
                    contributions["label"] = contributions["feature"].map(pretty_feature_name)
                    contributions["impact"] = np.where(
                        contributions["shap_value"] >= 0,
                        "Pushes toward red corner",
                        "Pushes toward blue corner",
                    )
                    top_contributions = contributions.reindex(
                        contributions["shap_value"].abs().sort_values(ascending=False).index
                    ).head(10)

                    takeaway_lines = []
                    for _, row in top_contributions.head(3).iterrows():
                        direction = "red corner" if row["shap_value"] >= 0 else "blue corner"
                        takeaway_lines.append(
                            f"- **{row['label']}** pushed the prediction toward the **{direction}**."
                        )

                    st.markdown("**Main drivers of this prediction**")
                    for line in takeaway_lines:
                        st.markdown(line)

                    st.dataframe(
                        top_contributions[
                            ["label", "feature_value", "shap_value", "impact"]
                        ].rename(
                            columns={
                                "label": "Feature",
                                "feature_value": "Value used",
                                "shap_value": "SHAP contribution",
                            }
                        ).reset_index(drop=True),
                        use_container_width=True,
                    )

                    plot_df = top_contributions.iloc[::-1]
                    colors = ["#c0392b" if value >= 0 else "#2980b9" for value in plot_df["shap_value"]]
                    fig, ax = plt.subplots(figsize=(8, 4.8))
                    ax.barh(plot_df["label"], plot_df["shap_value"], color=colors)
                    ax.axvline(0, color="#bbbbbb", linewidth=1)
                    ax.set_title("Top SHAP contributions")
                    ax.set_xlabel("Impact on red corner win probability")
                    ax.set_ylabel("")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("Processed data")
    if processed_df is None:
        st.info("No processed dataset has been generated yet.")
    else:
        total = len(processed_df)
        red_pct = processed_df[TARGET_COLUMN].mean() if TARGET_COLUMN in processed_df.columns else None

        m1, m2, m3 = st.columns(3)
        m1.metric("Total fights", f"{total:,}")
        if red_pct is not None:
            m2.metric("Red corner wins", f"{red_pct:.1%}")
            m3.metric("Blue corner wins", f"{1 - red_pct:.1%}")

        st.dataframe(processed_df.head(50), use_container_width=True)

        if TARGET_COLUMN in processed_df.columns:
            st.markdown("**Result distribution**")
            outcome_counts = (
                processed_df[TARGET_COLUMN]
                .value_counts()
                .rename(index={0: "Blue wins", 1: "Red wins"})
            )
            st.bar_chart(outcome_counts)

with tab3:
    st.subheader("Model report")
    if baseline_metrics:
        st.markdown("**Model comparison**")
        comparison_rows = [
            {
                "Model": "HistGradientBoostingClassifier",
                "Accuracy": metrics["accuracy"],
                "ROC AUC": metrics["roc_auc"],
                "Log loss": metrics["log_loss"],
            }
        ]
        for baseline_name, baseline_result in baseline_metrics.items():
            comparison_rows.append(
                {
                    "Model": baseline_name.replace("_", " ").title(),
                    "Accuracy": baseline_result["accuracy"],
                    "ROC AUC": baseline_result["roc_auc"],
                    "Log loss": baseline_result["log_loss"],
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if report_text:
        st.code(report_text, language="text")
        st.download_button(
            "⬇️ Download report",
            data=report_text,
            file_name="training_report.txt",
            mime="text/plain",
        )
    else:
        st.info("The report is not available yet.")
