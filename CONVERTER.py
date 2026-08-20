import torch
import torch.nn as nn
from torch.distributions import Categorical

MAX_HAND_SIZE = 10
MAX_ENEMIES = 5

# Map card types to numeric category IDs
CARD_TYPE_MAP = {"Attack": 1.0, "Skill": 2.0, "Power": 3.0, "Status": 4.0, "Curse": 5.0}

def vectorize_gamestate(snapshot: dict, legal_actions: list = None) -> dict[str, torch.Tensor]:
    player = snapshot.get("player") or {}
    enemies = snapshot.get("enemies") or []

    # 1. Player Scalars: [hp_ratio, block, energy, gold]
    max_hp = max(player.get("max_hp", 1), 1)
    player_tensor = torch.tensor([
        player.get("hp", 0) / max_hp,
        player.get("block", 0) / 100.0,
        player.get("energy", 0) / 10.0,
        player.get("gold", 0) / 1000.0
    ], dtype=torch.float32)

    # 2. Hand Cards: [MAX_HAND_SIZE, 4] -> [card_type, cost, upgraded, playable]
    hand = player.get("hand", [])
    hand_features = []
    for i in range(MAX_HAND_SIZE):
        if i < len(hand):
            card = hand[i]
            c_type = CARD_TYPE_MAP.get(card.get("card_type"), 0.0) / 5.0
            hand_features.append([
                c_type,
                card.get("cost", 0) / 5.0,
                1.0 if card.get("upgraded") else 0.0,
                1.0 if card.get("playable") else 0.0
            ])
        else:
            hand_features.append([0.0, 0.0, 0.0, 0.0])
    hand_tensor = torch.tensor(hand_features, dtype=torch.float32)

    # 3. Enemies: [MAX_ENEMIES, 5] -> [is_alive, hp_ratio, block, intent_dmg, intent_hits]
    enemy_features = []
    for i in range(MAX_ENEMIES):
        if i < len(enemies):
            e = enemies[i]
            e_max_hp = max(e.get("max_hp", 1), 1)
            enemy_features.append([
                1.0 if e.get("is_alive", True) else 0.0,
                e.get("hp", 0) / e_max_hp,
                e.get("block", 0) / 100.0,
                (e.get("intent_damage") or 0) / 50.0,
                (e.get("intent_hits") or 0) / 10.0
            ])
        else:
            enemy_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
    enemy_tensor = torch.tensor(enemy_features, dtype=torch.float32)

    # 4. Action Mask
    mask = None
    if legal_actions is not None:
        mask = torch.zeros(MAX_HAND_SIZE + 1, dtype=torch.bool)
        
        # Build map of card_id -> hand slot index
        card_id_to_idx = {
            card["card_id"]: i 
            for i, card in enumerate(hand) 
            if i < MAX_HAND_SIZE and "card_id" in card
        }

        for act in legal_actions:
            act_type = act.get("type")
            params = act.get("params", {})
            if act_type == "play_card":
                card_id = params.get("card_id")
                idx = card_id_to_idx.get(card_id, params.get("hand_index"))
                if idx is not None and idx < MAX_HAND_SIZE:
                    mask[idx] = True
            elif act_type == "end_turn":
                mask[-1] = True
                
        # Safety fallback: if no actions are legal, permit end_turn to avoid NaNs
        if not mask.any():
            mask[-1] = True

    return {
        "player": player_tensor,
        "hand": hand_tensor,
        "enemies": enemy_tensor,
        "action_mask": mask
    }

def flatten_for_network(state_dict: dict) -> torch.Tensor:
    # 4 (player) + 40 (hand: 10x4) + 25 (enemies: 5x5) = 69 total features
    flat_hand = state_dict["hand"].flatten()
    flat_enemies = state_dict["enemies"].flatten()
    return torch.cat([state_dict["player"], flat_hand, flat_enemies])

class SpirePPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 69  # Updated from 59 to 69 for card types
        num_actions = 11

        self.shared_body = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(128, num_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, flat_state: torch.Tensor):
        hidden = self.shared_body(flat_state)
        action_logits = self.actor_head(hidden)
        state_value = self.critic_head(hidden).squeeze(-1)
        return action_logits, state_value

def select_action(logits: torch.Tensor, mask: torch.Tensor):
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    dist = Categorical(logits=masked_logits)
    action_idx = dist.sample()
    return action_idx.item(), dist.log_prob(action_idx)

def build_bridge_payload(action_idx: int, legal_actions: list, snapshot: dict) -> dict:
    decision_id = snapshot.get("decision_id")

    # Handle Card Plays (0-9)
    if action_idx < 10:
        for act in legal_actions:
            if act.get("type") == "play_card" and act.get("params", {}).get("hand_index") == action_idx:
                params = dict(act.get("params", {}))
                
                if "target_constraints" in act:
                    enemies = snapshot.get("enemies", [])
                    for e in enemies:
                        if e.get("is_alive", True):
                            target = e.get("enemy_id") or e.get("instance_enemy_id") or "1"
                            params["target_id"] = str(target)
                            break
                
                return {
                    "decision_id": decision_id,
                    "action_id": act["action_id"],
                    "params": params
                }

    # Handle End Turn (10)
    elif action_idx == 10:
        for act in legal_actions:
            if act.get("type") == "end_turn":
                return {
                    "decision_id": decision_id,
                    "action_id": act["action_id"],
                    "params": act.get("params", {})
                }

    # === SAFETY FALLBACK ===
    # If the network picked an illegal card slot or unavailable action, 
    # automatically fallback to the first available legal action (usually end_turn or a valid play).
    if legal_actions:
        fallback_act = legal_actions[0]
        return {
            "decision_id": decision_id,
            "action_id": fallback_act["action_id"],
            "params": fallback_act.get("params", {})
        }

    raise ValueError(f"No legal actions available at all for decision {decision_id}.")