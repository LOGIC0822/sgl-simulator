"""ML-trained per-iter latency predictor.

Loads a joblib pickle of a sklearn-compatible regressor and predicts forward latency
from batch composition features. Train one with `train_latency_model.py`.

sim_config.json usage:
    "predictor": {
        "name": "ml",
        "database_path": "/path/to/latency_model.pkl"
    }
"""

import os

import joblib
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class MLTimePredictor(InferTimePredictor):
    """Per-iter latency predictor backed by an offline-trained sklearn regressor.

    Features (18 dim) extracted from ScheduleBatch:
        batch_size, sum/max/min(extend), sum/max/min(past),
        sum(extend*past), sum(extend^2), sum(past^2),
        sum_attn_flops (= sum(e*(p+e/2))),
        sum(extend × max_past), log1p(sum_past), log1p(sum_attn_flops),
        batch_size × sum_extend, max_past - min_past,
        is_decode, is_prefill
    """

    # This ordered list is the ABI between offline training and simulation.
    # The concrete regressor algorithm is intentionally unrestricted as long
    # as it exposes sklearn-compatible predict([[18 features]]) -> [seconds].

    FEATURE_NAMES = list(ML_FEATURE_NAMES)

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        latency_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        database_path = os.path.expandvars(os.path.expanduser(database_path))
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}. "
                "Train one with `train_latency_model.py` first."
            )

        bundle = joblib.load(database_path)
        if (
            not isinstance(bundle, dict)
            or "model" not in bundle
            or "features" not in bundle
        ):
            raise ValueError(
                "MLTimePredictor requires a joblib bundle containing both "
                "'model' and ordered 'features' metadata"
            )
        self._model = bundle["model"]
        saved_features = list(bundle["features"])

        if saved_features != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature contract mismatch: "
                f"saved={saved_features}, expected={self.FEATURE_NAMES}. "
                "Retrain or export the model with the exact 18-feature ABI."
            )
        if not callable(getattr(self._model, "predict", None)):
            raise TypeError(
                "MLTimePredictor model must expose a callable predict() method"
            )

        self._features = saved_features
        self._call_count = 0
        self._latency_scale = float(latency_scale)
        logger.info(
            "MLTimePredictor loaded from %s (model=%s, n_features=%d, latency_scale=%.4f)",
            database_path,
            type(self._model).__name__,
            len(self._features),
            self._latency_scale,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        feats = extract_ml_features(exts, pasts)

        self._call_count += 1
        return float(self._model.predict([feats])[0]) * self._latency_scale
