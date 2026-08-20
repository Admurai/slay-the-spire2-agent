from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.distributions import Categorical

MAX_HAND_SIZE = 10
MAX_ENEMIES = 5
NUM_ACTIONS = MAX_HAND_SIZE + 1

CARD_TYPE_MAP = {
    "attack": 1.0,
    "skill": 2.0,
    "power": 3.0,
    "status": 4.0,
    "curse": 5.0,
}


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _number(value)))


def _scaled(value: Any, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return _clamp(_number(value, default) / denominator)


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _identifier_values(item: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "card_id",
        "instance_card_id",
        "canonical_card_id",
        "enemy_id",
        "instance_enemy_id",
        "id",
    ):
        value = item.get(key)
        if value is not None:
            values.add(str(value))
    return values


def _card_type_value(card: dict[str, Any]) -> float:
    raw_type = card.get("card_type") or card.get("type") or ""
    return CARD_TYPE_MAP.get(str(raw_type).strip().lower(), 0.0) / 5.0


def _action_params(action: dict[str, Any]) -> dict[str, Any]:
    params = action.get("params")
    return params if isinstance(params, dict) else {}


def _action_card_index(action: dict[str, Any], hand: list[dict[str, Any]]) -> int | None:
    params = _action_params(action)
    for key in ("hand_index", "card_index", "slot", "index"):
        raw_index = params.get(key)
        if isinstance(raw_index, int) and 0 <= raw_index < min(len(hand), MAX_HAND_SIZE):
            return raw_index
        if isinstance(raw_index, str) and raw_index.isdigit():
            index = int(raw_index)
            if 0 <= index < min(len(hand), MAX_HAND_SIZE):
                return index

    action_ids = _identifier_values(params)
    if not action_ids:
        return None
    for index, card in enumerate(hand[:MAX_HAND_SIZE]):
        if action_ids.intersection(_identifier_values(card)):
            return index
    return None


def vectorize_gamestate(snapshot: dict, legal_actions: list | None = None) -> dict[str, torch.Tensor | None]:
    """Convert a bridge snapshot into bounded tensors for the PPO policy."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    player = snapshot.get("player") if isinstance(snapshot.get("player"), dict) else {}
    enemies = _as_list(snapshot.get("enemies"))

    max_hp = max(_number(player.get("max_hp"), 1.0), 1.0)
    player_tensor = torch.tensor(
        [
            _clamp(_number(player.get("hp")) / max_hp),
            _scaled(player.get("block"), 100.0),
            _scaled(player.get("energy"), 10.0),
            _scaled(player.get("gold"), 1000.0),
        ],
        dtype=torch.float32,
    )

    hand = _as_list(player.get("hand"))
    hand_features: list[list[float]] = []
    for index in range(MAX_HAND_SIZE):
        if index >= len(hand):
            hand_features.append([0.0, 0.0, 0.0, 0.0])
            continue
        card = hand[index]
        hand_features.append(
            [
                _card_type_value(card),
                _scaled(card.get("cost_for_turn", card.get("cost", 0)), 5.0),
                1.0 if bool(card.get("upgraded")) else 0.0,
                1.0 if bool(card.get("playable")) else 0.0,
            ]
        )
    hand_tensor = torch.tensor(hand_features, dtype=torch.float32)

    enemy_features: list[list[float]] = []
    for index in range(MAX_ENEMIES):
        if index >= len(enemies):
            enemy_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        enemy = enemies[index]
        enemy_max_hp = max(_number(enemy.get("max_hp"), 1.0), 1.0)
        enemy_features.append(
            [
                1.0 if bool(enemy.get("is_alive", True)) else 0.0,
                _clamp(_number(enemy.get("hp")) / enemy_max_hp),
                _scaled(enemy.get("block"), 100.0),
                _scaled(enemy.get("intent_damage"), 50.0),
                _scaled(enemy.get("intent_hits"), 10.0),
            ]
        )
    enemy_tensor = torch.tensor(enemy_features, dtype=torch.float32)

    action_mask: torch.Tensor | None = None
    if legal_actions is not None:
        action_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        for action in legal_actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("type", "")).lower()
            if action_type == "play_card":
                index = _action_card_index(action, hand[:MAX_HAND_SIZE])
                if index is not None:
                    action_mask[index] = True
            elif action_type == "end_turn":
                action_mask[-1] = True
        if not bool(action_mask.any()):
            action_mask[-1] = True

    return {
        "player": player_tensor,
        "hand": hand_tensor,
        "enemies": enemy_tensor,
        "action_mask": action_mask,
    }


def flatten_for_network(state_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
    return torch.cat(
        [
            state_dict["player"],
            state_dict["hand"].flatten(),
            state_dict["enemies"].flatten(),
        ]
    )


class SpirePPOAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_body = nn.Sequential(
            nn.Linear(69, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(128, NUM_ACTIONS)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, flat_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if flat_state.ndim == 1:
            flat_state = flat_state.unsqueeze(0)
        hidden = self.shared_body(flat_state)
        return self.actor_head(hidden), self.critic_head(hidden).squeeze(-1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    safe_mask = mask.to(dtype=torch.bool, device=logits.device).clone()
    empty_rows = ~safe_mask.any(dim=-1)
    if bool(empty_rows.any()):
        safe_mask[empty_rows, -1] = True
    return logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)


def select_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    deterministic: bool = False,
) -> tuple[int, torch.Tensor]:
    distribution = Categorical(logits=masked_logits(logits, mask))
    action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
    return int(action.item()), distribution.log_prob(action).squeeze()


def _choose_target(params: dict[str, Any], constraints: Iterable[Any], snapshot: dict) -> str | None:
    if params.get("target_id") is not None:
        return str(params["target_id"])
    enemies = _as_list(snapshot.get("enemies"))
    allowed = {str(value) for value in constraints if value is not None}
    alive = [enemy for enemy in enemies if bool(enemy.get("is_alive", True))]
    if allowed:
        matching = [enemy for enemy in alive if allowed.intersection(_identifier_values(enemy))]
        if matching:
            alive = matching
    if not alive:
        return None
    target = alive[0]
    for key in ("enemy_id", "instance_enemy_id", "id"):
        if target.get(key) is not None:
            return str(target[key])
    return None


def _payload_for_action(action: dict[str, Any], snapshot: dict) -> dict[str, Any]:
    params = dict(_action_params(action))
    constraints = action.get("target_constraints") or []
    if constraints and "target_id" not in params:
        target_id = _choose_target(params, constraints, snapshot)
        if target_id is not None:
            params["target_id"] = target_id
    return {
        "decision_id": snapshot.get("decision_id"),
        "action_id": action.get("action_id"),
        "params": params,
    }


def build_bridge_payload(action_idx: int, legal_actions: list, snapshot: dict) -> dict:
    """Translate a policy slot into a legal bridge action."""
    hand = _as_list((snapshot.get("player") or {}).get("hand"))[:MAX_HAND_SIZE]
    actions = [action for action in legal_actions if isinstance(action, dict)]
    if 0 <= action_idx < MAX_HAND_SIZE:
        for action in actions:
            if str(action.get("type", "")).lower() == "play_card" and _action_card_index(action, hand) == action_idx:
                return _payload_for_action(action, snapshot)
    if action_idx == MAX_HAND_SIZE:
        for action in actions:
            if str(action.get("type", "")).lower() == "end_turn":
                return _payload_for_action(action, snapshot)
    fallback = next(
        (action for action in actions if str(action.get("type", "")).lower() == "end_turn"),
        None,
    ) or (actions[0] if actions else None)
    if fallback is None:
        raise ValueError(f"No legal actions available for decision {snapshot.get('decision_id')}.")
    return _payload_for_action(fallback, snapshot)
