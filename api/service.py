from __future__ import annotations

import joblib
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency at runtime
    shap = None

from src.ufc_predictor.config import MODEL_PATH
from src.ufc_predictor.features import align_feature_columns, build_inference_frame


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


class PredictionService:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model artifact not found. Run `python -m src.ufc_predictor.train` first."
            )

        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.metrics: dict = bundle.get("metrics", {})
        self.shap_background = bundle.get("shap_background")
        self.shap_explainer = self._build_shap_explainer()

    def _build_shap_explainer(self):
        if shap is None or self.shap_background is None or len(self.shap_background) == 0:
            return None

        background = self.shap_background[self.feature_columns].copy()

        def predict_red_probability(data):
            frame = pd.DataFrame(data, columns=self.feature_columns)
            return self.model.predict_proba(frame)[:, 1]

        return shap.Explainer(
            predict_red_probability,
            background,
            feature_names=self.feature_columns,
            algorithm="permutation",
        )

    def predict(self, red_corner: dict, blue_corner: dict, include_explanations: bool = True) -> dict:
        inference_df = build_inference_frame(
            fight_date=pd.Timestamp.today().normalize(),
            fighter_red=red_corner["name"],
            fighter_blue=blue_corner["name"],
            red_age=red_corner["age"],
            blue_age=blue_corner["age"],
            red_height_cm=red_corner["height_cm"],
            blue_height_cm=blue_corner["height_cm"],
            red_reach_cm=red_corner["reach_cm"],
            blue_reach_cm=blue_corner["reach_cm"],
            red_wins=red_corner["wins"],
            blue_wins=blue_corner["wins"],
            red_losses=red_corner["losses"],
            blue_losses=blue_corner["losses"],
            red_sig_str_acc=red_corner["sig_str_acc"],
            blue_sig_str_acc=blue_corner["sig_str_acc"],
            red_takedown_acc=red_corner["takedown_acc"],
            blue_takedown_acc=blue_corner["takedown_acc"],
            red_stance=red_corner["stance"],
            blue_stance=blue_corner["stance"],
        )
        X = align_feature_columns(inference_df, self.feature_columns)
        red_prob = float(self.model.predict_proba(X)[0, 1])
        blue_prob = 1.0 - red_prob
        predicted_winner = "red" if red_prob >= 0.5 else "blue"
        predicted_name = red_corner["name"] if predicted_winner == "red" else blue_corner["name"]

        response = {
            "predicted_winner": predicted_winner,
            "predicted_fighter_name": predicted_name,
            "red_win_probability": red_prob,
            "blue_win_probability": blue_prob,
            "top_explanations": [],
        }

        if include_explanations and self.shap_explainer is not None:
            shap_values = self.shap_explainer(X, max_evals=2 * len(self.feature_columns) + 1)
            contributions = pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "shap_value": shap_values.values[0],
                    "feature_value": X.iloc[0].values,
                }
            )
            contributions["feature"] = contributions["feature"].map(pretty_feature_name)
            contributions["impact"] = contributions["shap_value"].apply(
                lambda value: "Pushes toward red corner"
                if value >= 0
                else "Pushes toward blue corner"
            )
            top_contributions = contributions.reindex(
                contributions["shap_value"].abs().sort_values(ascending=False).index
            ).head(5)
            response["top_explanations"] = top_contributions.to_dict(orient="records")

        return response
