import numpy as np
import pytest
import torch

from src.frozen_final_evaluation import (
    CHECKPOINT_SCHEMA,
    score_trials,
    validate_checkpoint_payload,
)
from src.frozen_handoff_cache import FrozenVerificationProtocol
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


def test_score_trials_uses_resolved_cache_indices():
    embeddings = torch.zeros((3, 192), dtype=torch.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 1.0
    embeddings[2, 1] = 1.0
    protocol = FrozenVerificationProtocol(
        enroll_indices=np.asarray([0, 0], dtype=np.int32),
        test_indices=np.asarray([1, 2], dtype=np.int32),
        targets=np.asarray([1, 0], dtype=np.int8),
        sha256="demo",
        trial_ids=("same", "different"),
    )
    scores = score_trials(embeddings, protocol)
    assert scores.tolist() == pytest.approx([1.0, 0.0])


def test_checkpoint_schema_guard_matches_current_training_pipeline():
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "embedding_model_state_dict": {},
        "mean_var_norm_state_dict": {},
        "runtime_config": {
            "model": {
                "source": SpeechBrainECAPAFrontend.SOURCE,
                "embedding_dim": 192,
            }
        },
        "cursor": {},
    }
    validate_checkpoint_payload(checkpoint)
    checkpoint["schema"] = "generic_ecapa_aam_training_v3"
    with pytest.raises(ValueError, match="frozen-handoff"):
        validate_checkpoint_payload(checkpoint)
