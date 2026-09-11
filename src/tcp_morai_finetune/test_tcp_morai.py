from __future__ import annotations

import unittest

import numpy as np
import torch

from tcp_morai_finetune.data import (
    TCP_COMMAND_AVOID,
    _morai_xy_to_tcp,
    _route_command,
    _route_target,
    _signal_window_mask,
)
from tcp_morai_finetune.model import TCPMorai


class TCPMoraiTest(unittest.TestCase):
    def test_explicit_avoid_command_preserves_official_tcp_width(self) -> None:
        command = np.zeros(6, dtype=np.float32)
        command[TCP_COMMAND_AVOID] = 1.0
        self.assertEqual(command.shape, (6,))
        self.assertEqual(int(command.argmax()), 4)

    def test_model_contract_and_official_parameter_count(self) -> None:
        model = TCPMorai()
        output = model(
            torch.zeros(2, 3, 256, 256),
            torch.zeros(2, 9),
            torch.zeros(2, 2),
        )
        self.assertEqual(tuple(output["waypoints"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["speed"].shape), (2, 1))
        self.assertEqual(tuple(output["action_logits"].shape), (2, 3))
        self.assertEqual(model.parameter_counts()["total"], 26_025_092)

    def test_route_adapter(self) -> None:
        route = np.zeros((64, 4), dtype=np.float32)
        route[:, 0] = np.arange(64, dtype=np.float32) - 2.0
        route[:, 2] = 1.0
        target = _route_target(route, 10.0)
        self.assertAlmostEqual(float(target[0]), 10.0, delta=1.1)
        command = _route_command(route)
        self.assertEqual(int(command.argmax()), 3)
        self.assertAlmostEqual(float(command.sum()), 1.0)

    def test_full_policy_accepts_native_morai_aspect_ratio(self) -> None:
        model = TCPMorai().eval()
        with torch.no_grad():
            output = model.forward_original(
                torch.zeros(1, 3, 360, 640),
                torch.zeros(1, 9),
                torch.zeros(1, 2),
            )
        self.assertEqual(tuple(output["waypoints"].shape), (1, 4, 2))
        self.assertEqual(tuple(output["action_alpha"].shape), (1, 2))

    def test_tcp_coordinate_adapter(self) -> None:
        morai = np.asarray([[10.0, 2.0], [20.0, -3.0]], dtype=np.float32)
        tcp = _morai_xy_to_tcp(morai)
        np.testing.assert_allclose(tcp, [[-2.0, -10.0], [3.0, -20.0]])

    def test_signal_window_marks_approach_and_exit(self) -> None:
        pose = np.stack((np.arange(61, dtype=np.float64), np.zeros(61)), axis=1)
        signal = np.asarray([[40.0, 0.0]], dtype=np.float64)
        mask = _signal_window_mask(pose, signal)
        self.assertTrue(mask[10])
        self.assertTrue(mask[50])
        self.assertFalse(mask[9])
        self.assertFalse(mask[51])

    def test_state_only_phase_freezes_tcp_encoder(self) -> None:
        model = TCPMorai()
        model.set_training_phase("state_only")
        model.train()
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("state_head.") for name in trainable))
        self.assertFalse(model.perception.training)
        self.assertTrue(model.state_head.training)


if __name__ == "__main__":
    unittest.main()
