from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

GAME_EXE = "SlayTheSpire2.exe"
DEFAULT_PORT = 17654


def wait_for_health(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start multiple isolated STS2 processes, one bridge port per process."
    )
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--port-start", type=int, default=DEFAULT_PORT)
    parser.add_argument("--health-timeout", type=float, default=60.0)
    parser.add_argument("--enable-writes", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()

    game_dir = args.game_dir.resolve()
    exe = game_dir / GAME_EXE
    if not exe.exists():
        raise SystemExit(f"Game executable not found: {exe}")
    if args.instances < 1:
        raise SystemExit("--instances must be at least 1")

    children: list[subprocess.Popen] = []
    try:
        for index in range(args.instances):
            port = args.port_start + index
            env = os.environ.copy()
            env["STS2_BRIDGE_PORT"] = str(port)
            env["STS2_BRIDGE_HOST"] = "127.0.0.1"
            env["STS2_BRIDGE_ENABLE_WRITES"] = "true" if args.enable_writes else "false"
            command = [str(exe), *args.extra_arg]
            print(f"starting instance {index + 1}/{args.instances} on bridge port {port}", flush=True)
            child = subprocess.Popen(command, cwd=game_dir, env=env)
            children.append(child)
            if wait_for_health(port, args.health_timeout):
                print(f"ready: http://127.0.0.1:{port}", flush=True)
            else:
                print(f"warning: bridge did not become healthy on port {port}", flush=True)

        print("all instances started; press Ctrl+C to stop them", flush=True)
        while True:
            exited = [child for child in children if child.poll() is not None]
            if exited:
                print(f"{len(exited)} game instance(s) exited", flush=True)
                return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    raise SystemExit(main())
