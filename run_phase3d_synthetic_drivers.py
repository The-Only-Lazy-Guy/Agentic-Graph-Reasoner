"""Persist deterministic Phase 3D synthetic plan-tree sessions."""
from __future__ import annotations

from reasoning.adaptive_planning_examples import persist_synthetic_plan_sessions


def main() -> None:
    paths = persist_synthetic_plan_sessions()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
