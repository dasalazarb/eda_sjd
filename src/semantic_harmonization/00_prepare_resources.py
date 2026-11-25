"""Prepare authorized local resources and public model/vocabulary snapshots."""
from pathlib import Path
import fcntl
import os
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "semantic_harmonization"

from .resources import prepare
from .settings import load_config


def main() -> None:
    """Prepare resources from YAML without opening study data."""
    if len(sys.argv) != 1:
        raise SystemExit("Configure this command through YAML.")
    os.umask(0o077)
    cfg = load_config()
    root = Path(cfg["paths"]["knowledge_dir"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = prepare(cfg)
    print("Resource manifest written.")
    for name, item in manifest["resources"].items():
        if item["status"] != "available":
            print(f"{name}: unavailable. Configure an authorized local resource or prepare the public snapshot.")


if __name__ == "__main__":
    main()
