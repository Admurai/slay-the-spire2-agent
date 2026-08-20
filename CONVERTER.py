import torch

# Define fixed limits for tensor shapes
MAX_HAND_SIZE = 10
MAX_ENEMIES = 5

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

    # 2. Hand Cards: [MAX_HAND_SIZE, 3] -> [cost, upgraded, playable]
    hand = player.get("hand", [])
    hand_features = []
    for i in range(MAX_HAND_SIZE):
        if i < len(hand):
            card = hand[i]
            hand_features.append([
                card.get("cost", 0) / 5.0,
                1.0 if card.get("upgraded") else 0.0,
                1.0 if card.get("playable") else 0.0
            ])
        else:
            hand_features.append([0.0, 0.0, 0.0])
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

    # 4. Action Mask (if legal_actions provided)
    mask = None
    if legal_actions is not None:
        # Example: mask of length MAX_HAND_SIZE + 1 (cards + end turn)
        mask = torch.zeros(MAX_HAND_SIZE + 1, dtype=torch.bool)
        for act in legal_actions:
            if act.get("type") == "play_card":
                idx = act.get("params", {}).get("hand_index", 0)
                if idx < MAX_HAND_SIZE:
                    mask[idx] = True
            elif act.get("type") == "end_turn":
                mask[-1] = True

    return {
        "player": player_tensor,
        "hand": hand_tensor,
        "enemies": enemy_tensor,
        "action_mask": mask
    }

def flatten_for_network(state_dict):
    # 1. Squash the 2D grids into 1D lines
    flat_hand = state_dict["hand"].flatten()       # 10 x 3 = 30 numbers
    flat_enemies = state_dict["enemies"].flatten() # 5 x 5 = 25 numbers
    
    # 2. Glue them all together with the player stats
    # Sizes: 4 (player) + 30 (hand) + 25 (enemies) = 59 total numbers
    flat_state = torch.cat([
        state_dict["player"], 
        flat_hand, 
        flat_enemies
    ])
    
    return flat_state


import torch.nn as nn

class SpirePPOAgent(nn.Module):
    def __init__(self):
        super().__init__() # Required PyTorch boilerplate
        
        # The total number of inputs we calculated above (4 + 30 + 25)
        input_size = 59 
        # Total possible actions (10 card slots + 1 end turn button)
        num_actions = 11 
        
        # --- THE SHARED BODY ---
        # nn.Linear are just standard neural network layers.
        # We step down from 59 inputs -> 128 hidden -> 128 hidden
        self.shared_body = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(), # The activation function (adds non-linearity)
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # --- THE ACTOR HEAD (Policy) ---
        # Takes the 128 hidden features and outputs 11 action scores (logits)
        self.actor_head = nn.Linear(128, num_actions)
        
        # --- THE CRITIC HEAD (Value) ---
        # Takes the 128 hidden features and outputs 1 single score (Value)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, flat_state):
        # 1. Pass the flattened state through the shared body
        hidden = self.shared_body(flat_state)
        
        # 2. Get the action scores (Actor)
        action_logits = self.actor_head(hidden)
        
        # 3. Get the state value (Critic)
        state_value = self.critic_head(hidden)
        
        return action_logits, state_value