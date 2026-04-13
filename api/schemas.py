from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


StanceType = Literal["Orthodox", "Southpaw", "Switch", "Open Stance", "Unknown"]


class FighterInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    age: float = Field(..., ge=18, le=60)
    height_cm: float = Field(..., ge=140, le=230)
    reach_cm: float = Field(..., ge=140, le=250)
    wins: int = Field(..., ge=0, le=80)
    losses: int = Field(..., ge=0, le=60)
    sig_str_acc: float = Field(..., ge=0.0, le=1.0)
    takedown_acc: float = Field(..., ge=0.0, le=1.0)
    stance: StanceType = "Unknown"


class PredictionRequest(BaseModel):
    red_corner: FighterInput
    blue_corner: FighterInput
    include_explanations: bool = True


class ExplanationItem(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    impact: str


class PredictionResponse(BaseModel):
    predicted_winner: Literal["red", "blue"]
    predicted_fighter_name: str
    red_win_probability: float
    blue_win_probability: float
    top_explanations: list[ExplanationItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    feature_count: int
