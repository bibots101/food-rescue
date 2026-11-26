"""Project configuration: config/default.yaml loaded into typed objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class LLMConfig(BaseModel):
    model: str = "qwen3.5:9b-q4_K_M"
    embedding_model: str = "bge-m3"
    host: str = "http://127.0.0.1:11434"  # "localhost" costs 2 s per new connection on Windows (IPv6 first)
    mode: Literal["live", "record", "replay"] = "record"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    think: bool = False
    max_retries: int = 2
    cache_dir: str = "data/llm_cache"


class GeoConfig(BaseModel):
    road_factor: float = 1.3
    speed_kmh: float = 25.0


class RoutingConfig(BaseModel):
    service_minutes: int = 5
    time_limit_s: float = 2.0
    pickup_delay_cost: int = 1  # cost of one minute of pickup delay, relative to one minute of driving


class MatchingConfig(BaseModel):
    w_need: float = 0.5  # how much the charity needs this food (its own estimate)
    w_distance: float = 0.3  # closer charities: less driving, fresher food
    w_fairness: float = 0.2  # charities that got less so far get a boost
    max_distance_km: float = 15.0  # distance at which closeness reaches 0


class SimulationConfig(BaseModel):
    step_minutes: int = 1
    cfp_timeout_minutes: int = 3
    # How the dispatcher reads drivers' problem reports: "cautious" (always a cancellation), "llm", or
    # "oracle" (true durations: what perfect reading would give). features.llm_dispatcher: true selects "llm".
    dispatcher_policy: Literal["cautious", "llm", "oracle"] = "cautious"
    # Drivers accept or decline each delivery (features.driver_confirmation): how long a request
    # waits before it goes to another driver, and who answers ("auto": simulated drivers, after 1-2 min;
    # "manual": the presenter, in the Driver screen).
    offer_timeout_minutes: int = 5
    driver_policy: Literal["auto", "manual"] = "auto"
    # A charity that wins a donation confirms it before it is delivered: "auto" (its agent decides from its
    # hidden preferences) or "manual" (a person accepts or declines in the Charity screen). An award not
    # answered within `confirm_timeout_minutes` goes to the next bidder.
    charity_policy: Literal["auto", "manual"] = "auto"
    confirm_timeout_minutes: int = 10


class FeatureFlags(BaseModel):
    replanning: bool = True  # ablation switch: False = a cancelled driver's food is lost
    contract_net: bool = True  # ablation switch: False = nearest open charity, one call, no negotiation
    llm_intake: bool = False
    rag_safety: bool = False
    learning: bool = False
    llm_dispatcher: bool = False
    driver_confirmation: bool = False  # off in simulations, on in the platform


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)


def load_config(path: Path | None = None) -> Config:
    """Read the YAML config. The LLM_MODE environment variable overrides `llm.mode`."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        config = Config.model_validate(yaml.safe_load(fh) or {})
    if mode := os.environ.get("LLM_MODE"):
        config.llm = LLMConfig.model_validate({**config.llm.model_dump(), "mode": mode})
    return config


def project_path(relative: str | Path) -> Path:
    """Resolve a path from the config relative to the project folder."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
