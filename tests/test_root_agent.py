from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from CONVERTER import SpirePPOAgent, build_bridge_payload, vectorize_gamestate
from Train import (
    TrainerConfig,
    PPOTrainer,
    SharedPolicy,
    choose_automatic_action,
    load_checkpoint,
    save_checkpoint,
)


class RootAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = {
            "decision_id": "d1",
            "phase": "combat",
            "player": {
                "hp": 70,
                "max_hp": 80,
                "block": 4,
                "energy": 3,
                "gold": 99,
                "hand": [
                    {"card_id": "strike#0", "card_type": "Attack", "cost": 1, "playable": True},
                    {"card_id": "defend#1", "card_type": "Skill", "cost": 1, "playable": True},
                ],
            },
            "enemies": [{"enemy_id": "e1", "hp": 20, "max_hp": 30, "is_alive": True}],
        }
        self.actions = [
            {
                "action_id": "a-strike",
                "type": "play_card",
                "params": {"card_id": "strike#0"},
                "target_constraints": ["e1"],
            },
            {
                "action_id": "a-defend",
                "type": "play_card",
                "params": {"card_id": "defend#1"},
            },
            {"action_id": "a-end", "type": "end_turn", "params": {}},
        ]

    def test_mask_and_target_payload(self) -> None:
        state = vectorize_gamestate(self.snapshot, self.actions)
        self.assertEqual(
            state["action_mask"].tolist(), [True, True] + [False] * 8 + [True]
        )
        payload = build_bridge_payload(0, self.actions, self.snapshot)
        self.assertEqual(payload["action_id"], "a-strike")
        self.assertEqual(payload["params"]["target_id"], "e1")

    def test_restart_action_selection(self) -> None:
        menu = {"phase": "menu", "metadata": {"window_kind": "main_menu"}}
        actions = [
            {"action_id": "continue", "type": "continue_run", "params": {}},
            {"action_id": "new", "type": "start_new_run", "params": {}},
        ]
        self.assertEqual(choose_automatic_action(menu, actions)["type"], "start_new_run")

        setup = {"phase": "menu", "metadata": {"window_kind": "new_run_setup"}}
        setup_actions = [
            {"action_id": "start", "type": "confirm_start_run", "params": {}}
        ]
        self.assertEqual(
            choose_automatic_action(setup, setup_actions)["type"], "confirm_start_run"
        )

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            config = TrainerConfig(checkpoint=path, minibatch_size=2)
            model = SpirePPOAgent()
            shared = SharedPolicy(model, torch.device("cpu"))
            trainer = PPOTrainer(shared, config)
            save_checkpoint(trainer, path)

            restored = SpirePPOAgent()
            optimizer = torch.optim.Adam(restored.parameters(), lr=config.learning_rate)
            updates, steps, episodes = load_checkpoint(
                restored, optimizer, path, torch.device("cpu")
            )
            self.assertEqual((updates, steps, episodes), (0, 0, 0))
            for expected, actual in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
