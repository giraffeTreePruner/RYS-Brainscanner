"""Pydantic models for the LL-MRI output JSON schema."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ScanMetadata(BaseModel):
    model_name: str
    model_type: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    total_params_base: str
    backend: str
    device: str
    scan_start_utc: str
    scan_end_utc: Optional[str] = None
    scan_duration_seconds: Optional[float] = None
    total_configs: int
    completed_configs: int
    pubmedqa_dataset: str
    pubmedqa_dataset_size: int
    eq_dataset: str
    eq_dataset_size: int
    max_new_tokens: int


class BaselineResult(BaseModel):
    config: list[int]
    pubmedqa_score: float
    eq_score: float
    combined_score: float


class ConfigResult(BaseModel):
    config: list[int]
    pubmedqa_score: float
    eq_score: float
    combined_score: float
    pubmedqa_delta: float
    eq_delta: float
    combined_delta: float
    duplicated_layers: list[int]
    num_duplicated: int
    layer_path: list[int]
    total_layers_in_path: int
    param_increase_pct: float


class Rankings(BaseModel):
    top_combined: list[list[int]]
    top_pubmedqa: list[list[int]]
    top_eq: list[list[int]]


class HeatmapMatrix(BaseModel):
    description: str
    data: list[list[Optional[float]]]


class HeatmapMatrices(BaseModel):
    pubmedqa_delta: HeatmapMatrix
    eq_delta: HeatmapMatrix
    combined_delta: HeatmapMatrix


class ScanResults(BaseModel):
    llmri_version: str = "1.0.0"
    scan_metadata: ScanMetadata
    baseline: Optional[BaselineResult] = None
    results: list[ConfigResult] = Field(default_factory=list)
    rankings: Optional[Rankings] = None
    heatmap_matrices: Optional[HeatmapMatrices] = None
