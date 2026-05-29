"""DispatchMissNudge — surface unknown-procedure mentions to the model.

Fires on `post_dispatch`. Scans the most recent raw model output for
apply/invoke/using phrases that mention procedure names; checks each
against MetaContext.procedure_names. Any name mentioned but unknown
gets an INFO signal listing the available procedures, so the model
can correct itself in the next iteration.

Conservative scan: only the patterns that the dispatcher itself
recognizes (apply, invoke, using_the, create_new, apply_intent).
Avoids flagging free-text mentions ("VerifyShortestPath is a useful
tool") as misses.

once_per_session per (mentioned_name) so the same typo doesn't
re-fire every iteration.

CAVEAT FOR FUTURE AUTHORS: this predicate reads raw model output,
which means false positives are possible. If the model says
"I'll apply common sense to..." or "I'll apply mathematical rigor
to..." those names won't match procedure_names → signal fires →
the prompt suggests procedures the model didn't actually want. The
signal text is non-prescriptive ("if you meant to invoke one of...")
to avoid misleading the model.
"""
from __future__ import annotations

import re
from typing import List, Set

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


# Same patterns the dispatcher uses for top-level scan, kept in sync.
# Capturing just the name; ignoring args.
_PROC_NAME = r"(?P<name>[A-Za-z][A-Za-z0-9_]*)"

_PATTERNS = [
    re.compile(rf"\bapply\s+{_PROC_NAME}\s+to\s+", re.IGNORECASE),
    re.compile(rf"\bI(?:'?ll|\s+will)\s+(?:now\s+)?apply\s+{_PROC_NAME}", re.IGNORECASE),
    re.compile(rf"\binvoke\s+{_PROC_NAME}", re.IGNORECASE),
    re.compile(rf"\busing\s+the\s+{_PROC_NAME}\s+procedure", re.IGNORECASE),
    re.compile(rf"\bcreate\s+a?\s*new\s+{_PROC_NAME}\s+(?:object|instance)", re.IGNORECASE),
]


def _detect_dispatch_misses(ctx: MetaContext) -> List[Signal]:
    if not ctx.raw_outputs:
        return []

    last_output = ctx.raw_outputs[-1]
    # Set of known procedure names (case-insensitive)
    known_lower = {n.lower() for n in ctx.procedure_names}

    # Names mentioned via dispatcher-style phrases
    mentioned: Set[str] = set()
    for pattern in _PATTERNS:
        for m in pattern.finditer(last_output):
            mentioned.add(m.group("name"))

    # Names mentioned that aren't known
    misses = {name for name in mentioned if name.lower() not in known_lower}

    if not misses:
        return []

    available_str = ", ".join(sorted(ctx.procedure_names)) or "(no procedures available)"

    signals: List[Signal] = []
    for name in sorted(misses):
        signals.append(Signal(
            id=f"dispatch_miss_{name.lower()}",
            type="dispatch_miss",
            severity="info",
            message=(
                f"You referenced \"{name}\" with a procedure-invocation phrase, "
                f"but no procedure by that name is registered. If you meant to "
                f"invoke one of: {available_str} — use the correct name. "
                f"If you meant the name as free-text prose, you can ignore "
                f"this notice."
            ),
            emitted_at_step=ctx.current_iteration,
            emitted_by="dispatch_miss_nudge",
            metadata={
                "mentioned_name": name,
                "available_names": sorted(ctx.procedure_names),
            },
            sticky=False,
            once=True,
        ))
    return signals


def build_dispatch_miss_nudge() -> MetaProcedure:
    return MetaProcedure(
        id="meta_dispatch_miss_nudge",
        name="DispatchMissNudge",
        purpose=(
            "Notify the model when it uses procedure-invocation phrasing for a "
            "name not in the procedure index, listing the available names."
        ),
        fires_on="post_dispatch",
        predicate=_detect_dispatch_misses,
        once_per_session=True,
        priority=40,
    )
