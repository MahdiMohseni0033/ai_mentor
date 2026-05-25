"""Verify the LLM, embedding model, and Chroma index are all working.

    python scripts/run_healthcheck.py

Exits non-zero if any check fails, so it can gate the rest of the workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.healthcheck import run_all


def main() -> None:
    print("Running health checks (this loads the embedding model — ~10s)...\n")
    results = run_all()
    all_ok = True
    for name, (ok, detail) in results.items():
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {name:10} {detail}")
        all_ok = all_ok and ok

    print("\nAll systems go." if all_ok else "\nSome checks failed — see messages above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
