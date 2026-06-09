from __future__ import annotations

import torch

from facepred.models import FacePredLoss, FacePredWorldModel
from facepred.models.losses import masked_binary_cross_entropy, sequence_cross_entropy


def tiny_model_config() -> dict:
    return {
        "encoders": {
            "audio_prosody": {"input_dim": 8, "hidden_dim": 12, "output_dim": 6},
            "vad": {"input_dim": 3, "hidden_dim": 6, "output_dim": 4},
            "quality": {"input_dim": 4, "hidden_dim": 6, "output_dim": 4},
        },
        "fusion": {
            "type": "cross_attention",
            "embed_dim": 16,
            "num_heads": 4,
            "dropout": 0.0,
            "reliability": "learned",
        },
        "rssm": {
            "gru_hidden": 12,
            "latent_type": "categorical",
            "categorical_classes": 4,
            "categorical_dims": 3,
            "gaussian_dim": 8,
        },
        "prediction_horizons_ms": [200, 1000],
        "heads": {
            "yield": {"enabled": True},
            "turn_taking": {"num_classes": 4, "hidden_dim": 16},
            "end_of_turn": {"num_buckets": 5, "hidden_dim": 16},
            "dialog_act": {"num_classes": 6, "hidden_dim": 16},
            "affect": {"num_emotions": 7, "hidden_dim": 16},
        },
    }


def test_world_model_forward_and_loss_accept_timestep_targets() -> None:
    torch.manual_seed(0)
    model = FacePredWorldModel.from_config(tiny_model_config())
    features = model.synthetic_features(batch_size=2, steps=4)
    outputs = model(features)

    assert outputs["turn_taking_logits"].shape == (2, 4, 2, 4)
    assert outputs["yield_logits"].shape == (2, 4, 2)
    assert outputs["yield_probs"].shape == (2, 4, 2)
    assert outputs["end_of_turn_logits"].shape == (2, 4, 2, 5)
    assert outputs["dialog_act_logits"].shape == (2, 4, 2, 6)
    assert outputs["valence_arousal"].shape == (2, 4, 2, 2)
    assert outputs["reliability"].shape == (2, 4, 3)

    targets = {
        "turn_taking": torch.randint(0, 4, (2, 4)),
        "yield": torch.randint(0, 2, (2, 4, 2)),
        "end_of_turn": torch.randint(0, 5, (2, 4)),
        "dialog_act": torch.randint(0, 6, (2, 4)),
        "emotion": torch.randint(0, 7, (2, 4)),
        "valence_arousal": torch.randn(2, 4, 2).clamp(-1, 1),
    }
    loss = FacePredLoss()(outputs, targets)

    assert torch.isfinite(loss.total)
    assert loss.total.item() > 0
    assert "rssm_dynamics" in loss.components


def test_world_model_predict_step_runs_eval_mode() -> None:
    model = FacePredWorldModel.from_config(tiny_model_config())
    outputs = model.predict_step(model.synthetic_features(batch_size=1, steps=2))

    assert outputs["rssm_state"].shape[:2] == (1, 2)
    assert model.training is False


def test_weighted_focal_cross_entropy_rewards_minority_correction() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 0, 1])
    weights = torch.tensor([0.5, 1.5])

    loss = sequence_cross_entropy(logits, targets, class_weights=weights, focal_gamma=1.5)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[-1, 1] < 0


def test_binary_yield_loss_respects_ignore_and_horizon_mask() -> None:
    logits = torch.tensor([[0.0, 0.0], [10.0, -10.0]], requires_grad=True)
    targets = torch.tensor([[1, 0], [-100, 1]])
    mask = torch.tensor([[True, True], [True, False]])

    loss = masked_binary_cross_entropy(logits, targets, mask=mask, pos_weight=torch.tensor([2.0, 2.0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad[1].abs().sum() == 0


def test_deterministic_rssm_has_no_stochastic_state() -> None:
    config = tiny_model_config()
    config["rssm"]["use_stochastic"] = False
    model = FacePredWorldModel.from_config(config)

    outputs = model(model.synthetic_features(batch_size=1, steps=3))

    assert outputs["rssm"].stochastic.shape[-1] == 0


def test_legacy_model_config_omits_yield_head() -> None:
    config = tiny_model_config()
    config["heads"].pop("yield")
    model = FacePredWorldModel.from_config(config)

    outputs = model(model.synthetic_features(batch_size=1, steps=3))

    assert "yield_logits" not in outputs
    assert "yield_probs" not in outputs
