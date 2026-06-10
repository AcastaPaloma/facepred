from __future__ import annotations

import pytest
import torch

from facepred.data import build_vap_state_targets


def test_vap_state_targets_encode_future_activity_for_both_speakers() -> None:
    activity = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 0],
        ]
    )

    targets, valid = build_vap_state_targets(activity, bin_steps=[2, 2])

    assert targets[0] == 9
    assert valid[:2].all()
    assert targets[-4:].eq(-100).all()


def test_vap_state_targets_require_two_speakers() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_vap_state_targets(torch.zeros(5, 1), bin_steps=[2])
