"""
MoE Expert Memory Manager for GLM-4.7-Flash (DeepSeekV2 architecture).

Status: scaffolding only. The profiler/cache primitives below are wired into
`MoEAwareController` as a proxy around any Answerer-v2 controller, but the
real routing hooks into llama.cpp are NOT implemented yet. `record_routing`
is only exercised by the Zipf simulator at the bottom of this file.

Architecture (planned):
  VRAM tier: hot expert pool (top-K per layer)
  RAM tier:  cold expert pool (remaining experts)
  LRU eviction: promotes/demotes based on recent routing patterns

DeepSeekV2 expert layout:
  64 experts total, top-4 routed per token, 1 shared expert
  46 MoE layers (47 total - 1 leading dense)
  Per expert: 3 weight tensors (ffn_gate, ffn_up, ffn_down)
"""

from __future__ import annotations
import json
import os
import time
import math
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants (from GLM-4.7-Flash GGUF metadata)
# ---------------------------------------------------------------------------

N_EXPERTS = 64
N_EXPERTS_USED = 4  # top-4 routing
N_EXPERTS_SHARED = 1
N_LAYERS = 47
N_MOE_LAYERS = 46  # 47 - 1 leading dense
HIDDEN_DIM = 2048
FFN_DIM = 1536  # per expert

# Memory estimates (at Q2_K quant).
# Q2_K is ~2.5–3 bits per weight; 0.20 of fp16 is a deliberate over-estimate
# so VRAM/RAM budgets quoted below stay on the safe side.
BYTES_PER_FP16_PARAM = 2
Q2_K_COMPRESSION_RATIO = 0.20
PARAMS_PER_EXPERT = (HIDDEN_DIM * FFN_DIM) * 2 + (FFN_DIM * HIDDEN_DIM)  # gate + up + down
BYTES_PER_EXPERT_Q2K = int(PARAMS_PER_EXPERT * Q2_K_COMPRESSION_RATIO)
MB_PER_EXPERT_Q2K = BYTES_PER_EXPERT_Q2K / (1024 * 1024)


@dataclass
class ExpertKey:
    """Unique key for an expert in a specific layer."""
    layer: int
    expert_id: int
    
    def __hash__(self):
        return hash((self.layer, self.expert_id))


@dataclass
class ExpertAccess:
    """Record of a single expert access during inference."""
    token_index: int
    expert_key: ExpertKey
    routing_weight: float = 0.0
    timestamp: float = 0.0


class ExpertCache:
    """Per-layer LRU cache for MoE expert weights.
    
    Each layer has its own pool of hot experts in VRAM.
    Cold experts live in RAM and are promoted on access.
    
    miss → RAM load (cold→hot promotion)
    hit  → LRU reorder only
    
    With per-layer pools: each layer caches H experts of 64.
    Top-4 active per token → max 4 misses per layer if all 4 are cold.
    H=8 means at most 4 misses per layer at steady state.
    """
    
    def __init__(
        self,
        n_experts: int = N_EXPERTS,
        n_layers: int = N_MOE_LAYERS,
        expert_size_bytes: int = BYTES_PER_EXPERT_Q2K,
        hot_per_layer: int = 8,  # cache 8 experts per MoE layer
    ):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.expert_size = expert_size_bytes
        self.hot_per_layer = hot_per_layer
        
        # Per-layer LRU pools
        self.layer_pools: List[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(n_layers)
        ]
        
        # Stats
        self.hits: int = 0
        self.misses: int = 0
        
        # Temperature tracking per layer
        self._temperatures: List[Counter] = [Counter() for _ in range(n_layers)]
        self._total_accesses: int = 0
    
    def access(self, layer: int, expert_id: int) -> bool:
        """Returns True if hot (VRAM hit)."""
        if layer >= self.n_layers:
            return False
        
        self._temperatures[layer][expert_id] += 1
        self._total_accesses += 1
        
        pool = self.layer_pools[layer]
        is_hit = expert_id in pool
        
        if is_hit:
            self.hits += 1
            pool.move_to_end(expert_id)
        else:
            self.misses += 1
            # Evict LRU if pool full
            if len(pool) >= self.hot_per_layer:
                pool.popitem(last=False)
            pool[expert_id] = None
        
        return is_hit
    
    def get_hot_experts(self, layer: int) -> List[int]:
        """Get hot expert IDs for a given layer."""
        return list(self.layer_pools[layer].keys()) if layer < self.n_layers else []
    
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        n_cold_total = self.n_experts * self.n_layers
        n_hot_total = self.hot_per_layer * self.n_layers
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "hot_slots_total": n_hot_total,
            "cold_slots_total": n_cold_total - n_hot_total,
            "hot_vram_gb": n_hot_total * self.expert_size / (1024**3),
            "cold_ram_gb": (n_cold_total - n_hot_total) * self.expert_size / (1024**3),
            "hot_per_layer": self.hot_per_layer,
        }


class ExpertRouterProfiler:
    """Captures and analyzes expert routing patterns.
    
    Works by hooking into the answerer inference loop:
    1. Records which tokens route to which experts
    2. Computes expert usage heatmap
    3. Identifies hot/cold expert sets
    4. Detects routing pattern shifts
    """
    
    def __init__(self, n_experts: int = N_EXPERTS, n_layers: int = N_MOE_LAYERS):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.layer_expert_counts: List[Counter] = [Counter() for _ in range(n_layers)]
        self.total_tokens: int = 0
        self.routing_entropy: List[float] = []
        
    def record_routing(self, layer: int, expert_ids: List[int], weights: List[float]) -> None:
        """Record which experts were activated for a token.
        
        Args:
            layer: MoE layer index (0-45)
            expert_ids: list of expert IDs (4 of 64)
            weights: routing weights for each expert
        """
        if layer >= self.n_layers:
            return
        for eid, w in zip(expert_ids, weights):
            self.layer_expert_counts[layer][eid] += 1
        self.total_tokens += 1
    
    def get_heatmap(self, top_k: int = 10) -> Dict[str, Any]:
        """Compute expert usage heatmap."""
        total_per_layer = [sum(c.values()) for c in self.layer_expert_counts]
        
        # Overall expert frequency
        global_counts = Counter()
        for c in self.layer_expert_counts:
            global_counts.update(c)
        
        total_all = sum(global_counts.values())
        
        # Top-k hottest experts
        hottest = global_counts.most_common(top_k)
        
        # Compute per-layer concentration (Gini-like)
        concentration = []
        for c in self.layer_expert_counts:
            vals = list(c.values())
            if vals:
                vals.sort(reverse=True)
                # Fraction of accesses in top-4 experts per layer
                top4_frac = sum(vals[:4]) / sum(vals) if sum(vals) > 0 else 0
                concentration.append(top4_frac)
        
        return {
            "total_tokens": self.total_tokens,
            "hottest_experts": [(eid, cnt, cnt/total_all*100) for eid, cnt in hottest],
            "per_layer_concentration": {
                "mean": sum(concentration) / len(concentration) if concentration else 0,
                "min": min(concentration) if concentration else 0,
                "max": max(concentration) if concentration else 0,
            },
            "top_k_access_percent": sum(cnt for _, cnt in hottest) / total_all * 100 if total_all > 0 else 0,
        }
    
    def compute_effective_hot_set(self, coverage_target: float = 0.9) -> Set[int]:
        """Compute minimal expert set that achieves coverage_target of total accesses.
        
        Returns the set of expert IDs that should be kept hot in VRAM.
        """
        global_counts = Counter()
        for c in self.layer_expert_counts:
            global_counts.update(c)
        
        total = sum(global_counts.values())
        target = total * coverage_target
        
        hot_experts = set()
        running = 0
        for eid, cnt in global_counts.most_common():
            hot_experts.add(eid)
            running += cnt
            if running >= target:
                break
        
        return hot_experts
    
    def summary(self) -> str:
        heat = self.get_heatmap()
        lines = [
            f"ExpertRouterProfiler summary:",
            f"  Tokens profiled: {heat['total_tokens']}",
            f"  Top-10 expert access: {heat['top_k_access_percent']:.1f}% of total",
            f"  Per-layer concentration (top-4): {heat['per_layer_concentration']['mean']:.1%} mean",
        ]
        if heat['hottest_experts']:
            lines.append(f"  Hottest experts: {[f'e{eid}({pct:.0f}%)' for eid, _, pct in heat['hottest_experts'][:5]]}")
        
        # Effective hot set
        for target in [0.5, 0.8, 0.9]:
            hot = self.compute_effective_hot_set(target)
            lines.append(f"  Experts needed for {target:.0%} coverage: {len(hot)}/{self.n_experts} ({len(hot)/self.n_experts*100:.0f}%)")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answerer Integration
# ---------------------------------------------------------------------------

class MoEAwareController:
    """Duck-typed proxy around an Answerer-v2 controller.

    Carries an `ExpertRouterProfiler` and `ExpertCache` so future llama.cpp
    routing hooks can record per-step expert activations. Currently a pure
    pass-through; the cache/profiler are exposed for callers to drive
    externally.
    """

    def __init__(self, base_controller, cache: Optional[ExpertCache] = None):
        self.base = base_controller
        self.profiler = ExpertRouterProfiler()
        self.cache = cache or ExpertCache()

    def choose_action(self, question, session, step, max_steps):
        return self.base.choose_action(question, session, step, max_steps)
    
    def note_result(self, action, result):
        if hasattr(self.base, 'note_result'):
            self.base.note_result(action, result)
    
    def build_answer(self, question, session):
        return self.base.build_answer(question, session)
    
    @property
    def name(self):
        return f"MoE-{self.base.name}"
    
    def get_profiler_summary(self) -> str:
        return self.profiler.summary()


# ---------------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------------

def estimate_architecture() -> Dict[str, Any]:
    """Print MoE architecture estimates for GLM-4.7-Flash."""
    total_expert_params = PARAMS_PER_EXPERT * N_EXPERTS * N_MOE_LAYERS
    active_per_token = PARAMS_PER_EXPERT * (N_EXPERTS_USED + N_EXPERTS_SHARED) * N_MOE_LAYERS
    
    return {
        "architecture": "DeepSeekV2 MoE",
        "total_params": "~4.7B",
        "active_per_token": f"{active_per_token / 1e9:.2f}B",
        "n_experts": N_EXPERTS,
        "experts_active_per_token": N_EXPERTS_USED,
        "n_moe_layers": N_MOE_LAYERS,
        "per_expert_params": f"{PARAMS_PER_EXPERT / 1e6:.1f}M",
        "per_expert_q2k": f"{MB_PER_EXPERT_Q2K:.1f} MB",
        "total_expert_fp16_gb": f"{total_expert_params * 2 / 1e9:.1f} GB",
        "active_ratio": f"{N_EXPERTS_USED / N_EXPERTS * 100:.1f}%",
        "cold_ratio": f"{(N_EXPERTS - N_EXPERTS_USED) / N_EXPERTS * 100:.1f}%",
        "cold_potential_gb": f"{(N_EXPERTS - N_EXPERTS_USED) * MB_PER_EXPERT_Q2K * N_MOE_LAYERS / 1024:.1f} GB",
        "vram_savings_with_paging": f"{(N_EXPERTS - N_EXPERTS_USED) / N_EXPERTS * 100:.0f}% of expert memory",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GLM-4.7-Flash MoE Expert Cache Manager")
    print("=" * 60)
    
    arch = estimate_architecture()
    for k, v in arch.items():
        print(f"  {k}: {v}")
    
    print("\n--- Expert Cache Simulation (per-layer, hot=8/64) ---")
    cache = ExpertCache(hot_per_layer=8)  # 8 hot experts per MoE layer
    
    import random
    rng = random.Random(42)
    
    # Simulate Zipf-like expert distribution per layer
    for _ in range(20000):
        layer = rng.randint(0, N_MOE_LAYERS - 1)
        
        # DeepSeekV2 top-4 routing: 4 experts active per token per layer
        # With Zipf distribution: 70% from top-6, 25% from next-10, 5% from rest
        r = rng.random()
        if r < 0.70:
            eid = rng.randint(0, 5)      # top-6 hot
        elif r < 0.95:
            eid = rng.randint(6, 15)     # next-10 warm
        else:
            eid = rng.randint(16, 63)    # cold tail
        
        cache.access(layer, eid)
    
    s = cache.stats()
    print(f"  Per-layer hot pool: {s['hot_per_layer']}/64 experts")
    print(f"  Total hot slots:    {s['hot_slots_total']:,}")
    print(f"  Hit rate:           {s['hit_rate']*100:.1f}%")
    print(f"  VRAM used:          {s['hot_vram_gb']:.2f} GB")
    print(f"  RAM used:           {s['cold_ram_gb']:.2f} GB")
    print(f"  VRAM saved vs full: {s['cold_ram_gb']:.2f} GB ({s['cold_slots_total']/(s['hot_slots_total']+s['cold_slots_total'])*100:.0f}%)")
    
    # The key insight: 5/8 of active experts are warm at steady state
    # (4 active per token, 8 cached → 8/4=2x overprovisioning covers shifts)
    print(f"\n  With top-4 routing: expected misses per token = 4 × (1 - {s['hit_rate']*100:.0f}%) = {4*(1-s['hit_rate']):.1f}")
    print(f"  RAM bandwidth per miss: {MB_PER_EXPERT_Q2K:.1f} MB per expert × 46 layers = {MB_PER_EXPERT_Q2K*46:.0f} MB (pessimistic)")
    print(f"  Realistic: only hot layers need swap, ~{MB_PER_EXPERT_Q2K:.1f} MB per miss")
    
    print("\nDone.")
