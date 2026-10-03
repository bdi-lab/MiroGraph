import json
import os
import re
from typing import Any, Dict, List, Tuple

from mirograph_config import AGENTS


FeatureSchema = Dict[str, Any]
FeatureNamesByAgent = Dict[str, Tuple[str, ...]]

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def load_feature_schema(path: str) -> FeatureSchema:
    if not path:
        raise ValueError("Feature schema path is required. Run feature_designer.py first.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature schema not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate_feature_schema(schema)
    return schema


def save_feature_schema(path: str, schema: FeatureSchema) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    validate_feature_schema(schema)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def validate_feature_schema(schema: FeatureSchema) -> None:
    if not isinstance(schema, dict):
        raise ValueError("Feature schema must be a JSON object.")
    agents = schema.get("agents")
    if not isinstance(agents, dict):
        raise ValueError("Feature schema must contain an 'agents' object.")
    missing = [agent for agent in AGENTS if agent not in agents]
    if missing:
        raise ValueError(f"Feature schema missing agents: {missing}")
    for agent in AGENTS:
        features = agents[agent].get("features") if isinstance(agents[agent], dict) else None
        if not isinstance(features, list) or not features:
            raise ValueError(f"Feature schema for {agent} must contain a non-empty features list.")
        names: List[str] = []
        for feature in features:
            if not isinstance(feature, dict):
                raise ValueError(f"Feature entry for {agent} must be an object.")
            name = feature.get("name")
            if not isinstance(name, str) or not SNAKE_CASE_RE.match(name):
                raise ValueError(f"Invalid feature name for {agent}: {name!r}")
            if name in names:
                raise ValueError(f"Duplicate feature name for {agent}: {name}")
            names.append(name)


def agent_features_from_schema(schema: FeatureSchema) -> FeatureNamesByAgent:
    validate_feature_schema(schema)
    return {
        agent: tuple(feature["name"] for feature in schema["agents"][agent]["features"])
        for agent in AGENTS
    }


def feature_descriptions(schema: FeatureSchema, agent: str) -> str:
    validate_feature_schema(schema)
    lines = []
    for feature in schema["agents"][agent]["features"]:
        name = feature["name"]
        description = str(feature.get("description", "")).strip()
        direction = str(feature.get("direction", "")).strip()
        value_range = str(feature.get("value_range", "")).strip()
        details = description
        if direction:
            details = f"{details} Direction: {direction}" if details else f"Direction: {direction}"
        if value_range:
            details = f"{details} Value range: {value_range}" if details else f"Value range: {value_range}"
        lines.append(f"- {name}: {details}")
    return "\n".join(lines)


def feature_names(agent_features: FeatureNamesByAgent) -> List[str]:
    return [
        f"{agent}_{field}"
        for agent in AGENTS
        for field in agent_features[agent]
    ]
