"""
start.py — запускає всі сервери і main.py.

Порядок:
  8901 — SearchMCP   (mcp_servers/search_mcp.py)
  8902 — ReportMCP   (mcp_servers/report_mcp.py)
  8903 — ACPServer   (acp_server.py)

Якщо сервер вже запущений — пропускає.
Якщо не запущений — стартує і чекає готовності.
Після того як всі живі — запускає main.py.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
VENV = PROJECT_DIR.parent / ".venv" / "Scripts"
PYTHON = str(VENV / "python.exe")
FASTMCP = str(VENV / "fastmcp.exe")

# ---------------------------------------------------------------------------
# Server definitions
# ---------------------------------------------------------------------------
SERVERS = [
    {
        "name": "SearchMCP",
        "port": 8901,
        "cmd": [
            FASTMCP, "run", "mcp_servers/search_mcp.py:mcp_server",
            "--transport", "http", "--port", "8901", "--host", "127.0.0.1",
        ],
        "startup_timeout": 60,
    },
    {
        "name": "ReportMCP",
        "port": 8902,
        "cmd": [
            FASTMCP, "run", "mcp_servers/report_mcp.py:mcp_server",
            "--transport", "http", "--port", "8902", "--host", "127.0.0.1",
        ],
        "startup_timeout": 30,
    },
    {
        "name": "ACPServer",
        "port": 8903,
        "cmd": [PYTHON, "acp_server.py"],
        "startup_timeout": 60,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def wait_for_port(name: str, port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        if is_port_open(port):
            return True
        time.sleep(2)
        dots += 1
        print(f"  waiting for {name}{'.' * dots}", end="\r")
    print()
    return False


def check_all() -> dict[str, bool]:
    return {s["name"]: is_port_open(s["port"]) for s in SERVERS}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(PROJECT_DIR)
    print("=" * 55)
    print("  Startup check")
    print("=" * 55)

    status = check_all()
    for name, alive in status.items():
        state = "running" if alive else "not running"
        print(f"  {name:12s} -> {state}")

    print()

    started_procs: list[tuple[dict, subprocess.Popen]] = []

    for server in SERVERS:
        name = server["name"]
        if status[name]:
            print(f"[OK]    {name} already running on :{server['port']}")
            continue

        print(f"[START] {name} on :{server['port']} ...")
        proc = subprocess.Popen(
            server["cmd"],
            cwd=PROJECT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started_procs.append((server, proc))

    if started_procs:
        print()
        all_ready = True
        for server, proc in started_procs:
            name = server["name"]
            ready = wait_for_port(name, server["port"], server["startup_timeout"])
            if ready:
                print(f"[OK]    {name} ready on :{server['port']}")
            else:
                print(f"[ERROR] {name} did not start within {server['startup_timeout']}s")
                all_ready = False

        if not all_ready:
            print("\n[ERROR] Some servers failed to start. Aborting.")
            for _, proc in started_procs:
                proc.terminate()
            sys.exit(1)

    print()
    print("=" * 55)
    print("  All servers running — starting main.py")
    print("=" * 55)
    print()

    try:
        result = subprocess.run([PYTHON, "main.py"], cwd=PROJECT_DIR)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for server, proc in started_procs:
            print(f"[STOP]  {server['name']}")
            proc.terminate()


if __name__ == "__main__":
    main()
