from __future__ import annotations

import argparse
import copy
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from CONVERTER import (
    NUM_ACTIONS,
    SpirePPOAgent,
    build_bridge_payload,
    flatten_for_network,
    masked_logits,
    vectorize_gamestate,
)

DEFAULT_URL = "http://127.0.0.1:17654"


@dataclass(frozen=True)
class TrainerConfig:
    base_urls: tuple[str, ...] = (DEFAULT_URL,)
    checkpoint: Path = Path("ppo_spire_model.pt")
    resume: Path | None = None
    device: str = "auto"
    rollout_steps: int = 128
    max_episodes: int = 0
    max_updates: int = 0
    poll_seconds: float = 0.2
    timeout_seconds: float = 3.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    ppo_epochs: int = 4
    minibatch_size: int = 256
    deterministic: bool = False


@dataclass
class Transition:
    state: torch.Tensor
    mask: torch.Tensor
    action: int
    old_log_prob: float
    value: float
    reward: float = 0.0
    next_value: float = 0.0
    done: bool = False


@dataclass
class Rollout:
    worker_id: int
    transitions: list[Transition]
    episode_done: bool = False
    result: str | None = None


class SharedPolicy:
    def __init__(self, model: SpirePPOAgent, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.lock = threading.RLock()

    def act(self, state: torch.Tensor, mask: torch.Tensor, deterministic: bool) -> tuple[int, float, float]:
        with self.lock, torch.no_grad():
            logits, value = self.model(state.to(self.device).unsqueeze(0))
            distribution = Categorical(logits=masked_logits(logits, mask.to(self.device)))
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def value(self, state: torch.Tensor) -> float:
        with self.lock, torch.no_grad():
            _, value = self.model(state.to(self.device).unsqueeze(0))
            return float(value.item())


class Bridge:
    def __init__(self, base_url: str, config: TrainerConfig) -> None:
        self.base_url = base_url.rstrip("/")
        self.config = config
        self.session = requests.Session()

    def window(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        snapshot_response = self.session.get(self.base_url + "/snapshot", timeout=self.config.timeout_seconds)
        actions_response = self.session.get(self.base_url + "/actions", timeout=self.config.timeout_seconds)
        snapshot_response.raise_for_status()
        actions_response.raise_for_status()
        snapshot = snapshot_response.json()
        actions = actions_response.json()
        return (snapshot if isinstance(snapshot, dict) else {}, actions if isinstance(actions, list) else [])

    def apply(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        response = self.session.post(
            self.base_url + "/apply",
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"message": response.text}
        return response.status_code, body if isinstance(body, dict) else {}


def _action(actions: list[dict[str, Any]], *names: str) -> dict[str, Any] | None:
    wanted = {name.lower() for name in names}
    return next(
        (item for item in actions if isinstance(item, dict) and str(item.get("type", "")).lower() in wanted),
        None,
    )


def automatic_action(snapshot: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Keep non-combat transitions moving until they have policy features."""
    phase = str(snapshot.get("phase", "")).lower()
    metadata = snapshot.get("metadata") or {}
    kind = str(metadata.get("window_kind", "")).lower()
    if phase == "menu":
        if kind == "new_run_setup":
            return _action(actions, "confirm_start_run", "select_character")
        return _action(actions, "start_new_run", "continue_run")
    if phase == "reward":
        return _action(actions, "skip_reward", "choose_reward")
    if phase == "map":
        return _action(actions, "choose_map_node")
    if phase == "event":
        return _action(actions, "continue_event", "choose_event_option")
    if phase == "shop":
        return _action(actions, "leave_shop")
    return None


def _hp(snapshot: dict[str, Any]) -> float:
    player = snapshot.get("player") or {}
    try:
        return float(player.get("hp", 0))
    except (TypeError, ValueError):
        return 0.0


def _enemy_hp(snapshot: dict[str, Any]) -> float:
    result = 0.0
    for enemy in snapshot.get("enemies") or []:
        if isinstance(enemy, dict) and enemy.get("is_alive", True):
            try:
                result += max(0.0, float(enemy.get("hp", 0)))
            except (TypeError, ValueError):
                pass
    return result


def _enemy_count(snapshot: dict[str, Any]) -> int:
    return sum(
        1 for enemy in snapshot.get("enemies") or []
        if isinstance(enemy, dict) and enemy.get("is_alive", True)
    )


def terminal(snapshot: dict[str, Any]) -> bool:
    metadata = snapshot.get("metadata") or {}
    return bool(
        snapshot.get("terminal")
        or str(snapshot.get("phase", "")).lower() == "terminal"
        or str(metadata.get("result", "")).lower() in {"win", "victory", "loss", "defeat"}
    )


def reward(previous: dict[str, Any], current: dict[str, Any]) -> float:
    """Bounded transition reward; no reward is assigned to an unobserved action."""
    value = -0.01 + max(-2.0, min(2.0, (_hp(current) - _hp(previous)) * 0.05))
    if previous.get("phase") == current.get("phase") == "combat":
        value += min(2.0, max(0.0, _enemy_hp(previous) - _enemy_hp(current)) * 0.05)
        value += min(2.0, float(max(0, _enemy_count(previous) - _enemy_count(current))))
    if previous.get("phase") == "combat" and current.get("phase") in {"reward", "map"}:
        value += 2.0
    if terminal(current):
        result = str((current.get("metadata") or {}).get("result", "")).lower()
        value += 10.0 if result in {"win", "victory"} else -10.0
    return max(-12.0, min(12.0, value))


def _accepted(status: int, body: dict[str, Any]) -> bool:
    return status == 200 and str(body.get("status", "")).lower() == "accepted"


def _state(snapshot: dict[str, Any], actions: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = vectorize_gamestate(snapshot, actions)
    mask = encoded["action_mask"]
    if mask is None:
        mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    return flatten_for_network(encoded).detach(), mask.detach()


def run_worker(
    worker_id: int,
    base_url: str,
    policy: SharedPolicy,
    config: TrainerConfig,
    output: queue.Queue[Rollout],
    stop: threading.Event,
    episodes: list[int],
    episodes_lock: threading.Lock,
) -> None:
    bridge = Bridge(base_url, config)
    transitions: list[Transition] = []
    pending: tuple[Transition, dict[str, Any]] | None = None
    last_decision: str | None = None
    last_terminal_decision: str | None = None
    episode_return = 0.0

    def flush(done: bool = False, result: str | None = None) -> None:
        nonlocal transitions, episode_return
        if transitions:
            output.put(Rollout(worker_id, transitions, done, result))
            transitions = []
        if done:
            with episodes_lock:
                episodes[0] += 1
            print(f"worker={worker_id} episode_return={episode_return:.3f} result={result or 'unknown'}")
            episode_return = 0.0

    while not stop.is_set():
        try:
            snapshot, actions = bridge.window()
            phase = str(snapshot.get("phase", "")).lower()
            decision_id = snapshot.get("decision_id")

            if pending is not None and decision_id != pending[1].get("decision_id"):
                transition, previous = pending
                next_state, _ = _state(snapshot, actions)
                transition.reward = reward(previous, snapshot)
                transition.next_value = policy.value(next_state)
                transition.done = terminal(snapshot)
                transitions.append(transition)
                episode_return += transition.reward
                pending = None
                if len(transitions) >= config.rollout_steps:
                    flush()

            if terminal(snapshot):
                if decision_id != last_terminal_decision:
                    flush(True, str((snapshot.get("metadata") or {}).get("result", "terminal")))
                    last_terminal_decision = decision_id
                time.sleep(0.5)
                continue

            if phase != "combat":
                action = automatic_action(snapshot, actions)
                if action is not None and decision_id != last_decision:
                    status, body = bridge.apply({
                        "decision_id": decision_id,
                        "action_id": action.get("action_id"),
                        "params": dict(action.get("params") or {}),
                    })
                    if _accepted(status, body) or body.get("error_code") == "stale_decision":
                        last_decision = decision_id
                    else:
                        print(f"worker={worker_id} noncombat_reject={status} body={body}")
                if phase == "menu":
                    last_terminal_decision = None
                time.sleep(config.poll_seconds)
                continue

            if not actions or decision_id == last_decision:
                time.sleep(config.poll_seconds)
                continue

            state, mask = _state(snapshot, actions)
            action_index, old_log_prob, value = policy.act(state, mask, config.deterministic)
            status, body = bridge.apply(build_bridge_payload(action_index, actions, snapshot))
            if _accepted(status, body):
                last_decision = decision_id
                pending = (Transition(state.cpu(), mask.cpu(), action_index, old_log_prob, value), copy.deepcopy(snapshot))
            elif body.get("error_code") == "stale_decision":
                last_decision = decision_id
            else:
                print(f"worker={worker_id} combat_reject={status} body={body}")
            time.sleep(config.poll_seconds)
        except requests.RequestException as exc:
            print(f"worker={worker_id} bridge_wait={exc}")
            time.sleep(1.0)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            print(f"worker={worker_id} state_error={type(exc).__name__}: {exc}")
            time.sleep(config.poll_seconds)

    flush()


def ppo_update(policy: SharedPolicy, optimizer: torch.optim.Optimizer, config: TrainerConfig, rollouts: list[Rollout]) -> dict[str, float]:
    transitions = [item for rollout in rollouts for item in rollout.transitions]
    if not transitions:
        return {}
    device = policy.device
    states = torch.stack([item.state for item in transitions]).to(device)
    masks = torch.stack([item.mask for item in transitions]).to(device)
    actions = torch.tensor([item.action for item in transitions], device=device)
    old_log_probs = torch.tensor([item.old_log_prob for item in transitions], device=device)
    returns: list[float] = []
    advantages: list[float] = []
    for rollout in rollouts:
        gae = 0.0
        local_advantages = [0.0] * len(rollout.transitions)
        for index in reversed(range(len(rollout.transitions))):
            item = rollout.transitions[index]
            non_terminal = 0.0 if item.done else 1.0
            delta = item.reward + config.gamma * item.next_value * non_terminal - item.value
            gae = delta + config.gamma * config.gae_lambda * non_terminal * gae
            local_advantages[index] = gae
        returns.extend(item.value + adv for item, adv in zip(rollout.transitions, local_advantages))
        advantages.extend(local_advantages)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    if len(advantages_tensor) > 1:
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

    metrics: dict[str, float] = {}
    with policy.lock:
        policy.model.train()
        batch_size = states.shape[0]
        minibatch_size = min(max(1, config.minibatch_size), batch_size)
        for _ in range(max(1, config.ppo_epochs)):
            order = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                index = order[start:start + minibatch_size]
                logits, values = policy.model(states[index])
                distribution = Categorical(logits=masked_logits(logits, masks[index]))
                new_log_probs = distribution.log_prob(actions[index])
                ratios = (new_log_probs - old_log_probs[index]).exp()
                clipped = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                actor_loss = -torch.minimum(ratios * advantages_tensor[index], clipped * advantages_tensor[index]).mean()
                value_loss = F.mse_loss(values, returns_tensor[index])
                entropy = distribution.entropy().mean()
                loss = actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
                optimizer.step()
                metrics = {"loss": float(loss.detach().cpu()), "return": float(returns_tensor.mean().detach().cpu())}
        policy.model.eval()
    return metrics


def save_checkpoint(policy: SharedPolicy, optimizer: torch.optim.Optimizer, config: TrainerConfig, updates: int, steps: int, episodes: int) -> None:
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.checkpoint.with_suffix(config.checkpoint.suffix + ".tmp")
    with policy.lock:
        torch.save({
            "model_state_dict": policy.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "updates": updates,
            "total_steps": steps,
            "total_episodes": episodes,
        }, temporary)
    os.replace(temporary, config.checkpoint)


def load_checkpoint(policy: SharedPolicy, optimizer: torch.optim.Optimizer, path: Path) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location=policy.device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        policy.model.load_state_dict(checkpoint)
        return 0, 0, 0
    policy.model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("updates", 0)), int(checkpoint.get("total_steps", 0)), int(checkpoint.get("total_episodes", 0))


def train(config: TrainerConfig) -> None:
    random.seed(7)
    torch.manual_seed(7)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else config.device if config.device != "auto" else "cpu")
    policy = SharedPolicy(SpirePPOAgent().to(device), device)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)
    updates = steps = completed_episodes = 0
    if config.resume and config.resume.exists():
        updates, steps, completed_episodes = load_checkpoint(policy, optimizer, config.resume)
        print(f"resumed={config.resume} updates={updates} steps={steps}")
    policy.model.eval()

    stop = threading.Event()
    rollouts: queue.Queue[Rollout] = queue.Queue()
    episodes = [completed_episodes]
    episodes_lock = threading.Lock()
    workers = [threading.Thread(target=run_worker, args=(index, url, policy, config, rollouts, stop, episodes, episodes_lock), daemon=True) for index, url in enumerate(config.base_urls)]
    for worker in workers:
        worker.start()

    buffered: list[Rollout] = []
    try:
        while not stop.is_set():
            try:
                buffered.append(rollouts.get(timeout=1.0))
            except queue.Empty:
                continue
            enough_steps = sum(len(item.transitions) for item in buffered) >= config.rollout_steps
            ended_episode = any(item.episode_done for item in buffered)
            if not enough_steps and not ended_episode:
                continue
            metrics = ppo_update(policy, optimizer, config, buffered)
            steps += sum(len(item.transitions) for item in buffered)
            updates += bool(metrics)
            buffered = []
            if metrics:
                print(f"update={updates} steps={steps} episodes={episodes[0]} loss={metrics['loss']:.4f} return={metrics['return']:.3f}")
            save_checkpoint(policy, optimizer, config, updates, steps, episodes[0])
            if (config.max_episodes and episodes[0] >= config.max_episodes) or (config.max_updates and updates >= config.max_updates):
                stop.set()
    except KeyboardInterrupt:
        print("stopping workers")
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=3.0)
        if buffered:
            ppo_update(policy, optimizer, config, buffered)
        save_checkpoint(policy, optimizer, config, updates, steps, episodes[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO through one or more STS2 bridge instances.")
    parser.add_argument("--base-url", action="append", dest="base_urls", help="Repeat for each bridge/game instance.")
    parser.add_argument("--workers", type=int, default=1, help="Use sequential localhost ports when --base-url is omitted.")
    parser.add_argument("--port-start", type=int, default=17654)
    parser.add_argument("--checkpoint", type=Path, default=Path("ppo_spire_model.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainerConfig:
    urls = tuple(url.rstrip("/") for url in args.base_urls) if args.base_urls else tuple(f"http://127.0.0.1:{args.port_start + i}" for i in range(max(1, args.workers)))
    return TrainerConfig(
        base_urls=urls,
        checkpoint=args.checkpoint,
        resume=args.resume,
        device=args.device,
        rollout_steps=max(1, args.rollout_steps),
        max_episodes=max(0, args.max_episodes),
        max_updates=max(0, args.max_updates),
        poll_seconds=max(0.01, args.poll_seconds),
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    train(config_from_args(build_parser().parse_args()))
