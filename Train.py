import time
import requests
import torch
from CONVERTER import (
    vectorize_gamestate, 
    flatten_for_network, 
    SpirePPOAgent, 
    select_action, 
    build_bridge_payload
)

# Bridge Mod Configuration
BRIDGE_BASE_URL = "http://127.0.0.1:17654"
POLL_INTERVAL_SEC = 0.5  # How long to wait between pings

def run_agent_loop():
    # 1. Initialize PyTorch Agent
    # In a real training scenario, you would load your saved weights here
    # e.g., agent.load_state_dict(torch.load("ppo_spire_model.pth"))
    agent = SpirePPOAgent()
    agent.eval() # Set to evaluation mode for live play
    
    # 2. Open an HTTP Session for fast, reused connections
    session = requests.Session()
    print(f"Connecting to STS2 Bridge at {BRIDGE_BASE_URL}...")
    
    # Track the last decision_id to avoid sending duplicate commands for the same turn
    last_decision_id = None

    try:
        while True:
            try:
                # Polling the gamestate and available actions
                snap_resp = session.get(f"{BRIDGE_BASE_URL}/snapshot", timeout=2)
                act_resp = session.get(f"{BRIDGE_BASE_URL}/actions", timeout=2)
                
                # Wait for the game to be fully loaded and returning valid JSON
                if snap_resp.status_code != 200 or act_resp.status_code != 200:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                    
                snapshot = snap_resp.json()
                legal_actions = act_resp.json()
                
                # Check if the game animations have finished and it's actually ready for input
                is_ready = snapshot.get("compatibility", {}).get("ready", True)
                if not is_ready:
                    time.sleep(0.1)
                    continue

                current_decision_id = snapshot.get("decision_id")
                phase = snapshot.get("phase")
                window_kind = snapshot.get("metadata", {}).get("window_kind")
                
                if phase == "combat" and window_kind == "player_turn" and current_decision_id != last_decision_id:
                    
                    if not legal_actions:
                        time.sleep(0.1)
                        continue

                    print(f"\n--- New Combat Turn Detected ({current_decision_id}) ---")
                    
                    state_dict = vectorize_gamestate(snapshot, legal_actions)
                    flat_state = flatten_for_network(state_dict)
                    
                    with torch.no_grad():
                        logits, state_value = agent(flat_state)
                        
                    action_idx, log_prob = select_action(logits, state_dict["action_mask"])
                    payload = build_bridge_payload(action_idx, legal_actions, snapshot)
                    
                    # Execute the action in-game
                    apply_resp = session.post(f"{BRIDGE_BASE_URL}/apply", json=payload, timeout=5)
                    
                    if apply_resp.status_code == 200:
                        apply_data = apply_resp.json()
                        if apply_data.get("status") == "accepted":
                            print(f"Success: {apply_data.get('message')}")
                            # Lock the decision ID so we wait for the board to update
                            last_decision_id = current_decision_id
                        else:
                            print(f"Action Rejected: {apply_data}")
                            time.sleep(0.5) # Wait before trying again
                            
                    elif apply_resp.status_code == 409:
                        apply_data = apply_resp.json()
                        print(f"Conflict 409: {apply_data.get('error_code')} - {apply_data.get('message')}")
                        if apply_data.get("error_code") == "stale_decision":
                            # The game has already moved on, lock this ID so we poll for the new one
                            last_decision_id = current_decision_id
                        else:
                            time.sleep(0.5)
                    else:
                        print(f"Apply Endpoint Failed. HTTP {apply_resp.status_code}")
                        time.sleep(0.5)
                
                
                elif phase in ["menu", "reward", "map", "event"]:# Automatically pick the first available valid option to speedrun back to combat
                                    action_idx = 0 
                                    payload = build_bridge_payload(action_idx, legal_actions, snapshot)
                                    session.post(f"{BRIDGE_BASE_URL}/apply", json=payload)

                # Rest briefly before pinging again
                time.sleep(POLL_INTERVAL_SEC)
                
            except requests.exceptions.RequestException as e:
                # The game might be closed or loading, gracefully ignore and retry
                print(f"Waiting for STS2 Bridge... ({e})", end="\r")
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\nAgent loop terminated by user.")

if __name__ == "__main__":
    run_agent_loop()