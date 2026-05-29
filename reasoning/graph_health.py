"""Graph health monitor — structural metrics for small-world validation.

Computes metrics that indicate whether the knowledge graph maintains
desirable structural properties (connectivity, short paths, clustering)
as edits accumulate. Used by both inline and offline editors to gate
edits based on structural impact.

Key metrics:
  - connected_components: number + largest component fraction
  - avg_degree: mean edges per node
  - clustering_coefficient: fraction of closed triangles (high = good)
  - approx_diameter: sampled BFS max distance (short = good)
  - hub_reachability: % of nodes reachable from any hub within 3 hops
  - orphan_count: nodes with zero edges (bad — unreachable knowledge)
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from graph_core import MemoryGraph


@dataclass
class GraphHealthReport:
    node_count: int = 0
    edge_count: int = 0
    connected_components: int = 0
    largest_component_frac: float = 0.0
    orphan_count: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    clustering_coefficient: float = 0.0
    approx_diameter: int = 0
    avg_path_length: float = 0.0
    hub_reachability_3hop: float = 0.0
    hub_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def health_score(self) -> float:
        """Composite score 0-1. Higher = healthier small-world structure.

        Components:
          - connectivity (0.3): 1.0 if single component, penalize fragmentation
          - clustering (0.2): raw clustering coefficient (0-1)
          - reachability (0.3): hub reachability within 3 hops
          - no-orphans (0.2): 1.0 - orphan_fraction
        """
        connectivity = self.largest_component_frac if self.node_count > 0 else 0.0
        clustering = min(1.0, self.clustering_coefficient)
        reachability = self.hub_reachability_3hop
        orphan_frac = (self.orphan_count / self.node_count) if self.node_count > 0 else 0.0
        no_orphans = 1.0 - orphan_frac
        return (
            0.3 * connectivity +
            0.2 * clustering +
            0.3 * reachability +
            0.2 * no_orphans
        )


@dataclass
class HealthDelta:
    """Difference between two health reports."""
    before: GraphHealthReport
    after: GraphHealthReport
    score_delta: float = 0.0
    component_delta: int = 0
    orphan_delta: int = 0
    clustering_delta: float = 0.0
    reachability_delta: float = 0.0
    verdict: str = ""  # "healthy" | "degraded" | "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_delta": round(self.score_delta, 4),
            "component_delta": self.component_delta,
            "orphan_delta": self.orphan_delta,
            "clustering_delta": round(self.clustering_delta, 4),
            "reachability_delta": round(self.reachability_delta, 4),
            "verdict": self.verdict,
            "before_score": round(self.before.health_score, 4),
            "after_score": round(self.after.health_score, 4),
        }


def compute_health(graph: MemoryGraph, *, sample_bfs: int = 50) -> GraphHealthReport:
    """Compute structural health metrics for the graph."""
    report = GraphHealthReport()
    nodes = graph.nodes
    edges = graph.edges
    report.node_count = len(nodes)
    report.edge_count = len(edges)

    if not nodes:
        return report

    # Build adjacency list (undirected for structural analysis)
    adj: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    for e in edges:
        if e.src in adj and e.dst in adj:
            adj[e.src].add(e.dst)
            adj[e.dst].add(e.src)

    # Degree stats
    degrees = [len(adj[nid]) for nid in nodes]
    report.avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
    report.max_degree = max(degrees) if degrees else 0
    report.orphan_count = sum(1 for d in degrees if d == 0)

    # Connected components (BFS)
    visited: Set[str] = set()
    components: List[int] = []
    for start in nodes:
        if start in visited:
            continue
        comp_size = 0
        queue = deque([start])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            comp_size += 1
            for nb in adj[nid]:
                if nb not in visited:
                    queue.append(nb)
        components.append(comp_size)
    report.connected_components = len(components)
    report.largest_component_frac = (
        max(components) / report.node_count if components else 0.0
    )

    # Clustering coefficient (average local)
    cc_sum = 0.0
    cc_count = 0
    for nid in nodes:
        neighbors = adj[nid]
        k = len(neighbors)
        if k < 2:
            continue
        triangles = 0
        nb_list = list(neighbors)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                if nb_list[j] in adj[nb_list[i]]:
                    triangles += 1
        possible = k * (k - 1) / 2
        cc_sum += triangles / possible if possible > 0 else 0
        cc_count += 1
    report.clustering_coefficient = cc_sum / cc_count if cc_count > 0 else 0.0

    # Approximate diameter + avg path length (sampled BFS from random nodes)
    all_node_ids = list(nodes.keys())
    sample_sources = random.sample(all_node_ids, min(sample_bfs, len(all_node_ids)))
    max_dist = 0
    total_dist = 0
    dist_count = 0
    for src in sample_sources:
        dist = _bfs_distances(src, adj)
        for d in dist.values():
            if d > 0:
                total_dist += d
                dist_count += 1
                if d > max_dist:
                    max_dist = d
    report.approx_diameter = max_dist
    report.avg_path_length = total_dist / dist_count if dist_count > 0 else 0.0

    # Hub reachability: % of nodes reachable from ANY hub within 3 hops
    hub_ids = [nid for nid, n in nodes.items() if n.node_type == "hub"]
    report.hub_count = len(hub_ids)
    reachable_from_hubs: Set[str] = set()
    for hub in hub_ids:
        dist = _bfs_distances(hub, adj, max_depth=3)
        reachable_from_hubs.update(dist.keys())
    report.hub_reachability_3hop = (
        len(reachable_from_hubs) / report.node_count if report.node_count > 0 else 0.0
    )

    return report


def compare_health(
    before: GraphHealthReport,
    after: GraphHealthReport,
    *,
    degradation_threshold: float = -0.02,
) -> HealthDelta:
    """Compare two health reports and produce a verdict."""
    delta = HealthDelta(
        before=before,
        after=after,
        score_delta=after.health_score - before.health_score,
        component_delta=after.connected_components - before.connected_components,
        orphan_delta=after.orphan_count - before.orphan_count,
        clustering_delta=after.clustering_coefficient - before.clustering_coefficient,
        reachability_delta=after.hub_reachability_3hop - before.hub_reachability_3hop,
    )
    if delta.score_delta < degradation_threshold:
        delta.verdict = "degraded"
    elif delta.score_delta > 0.005:
        delta.verdict = "healthy"
    else:
        delta.verdict = "neutral"
    return delta


def _bfs_distances(
    source: str,
    adj: Dict[str, Set[str]],
    max_depth: int = 999,
) -> Dict[str, int]:
    """BFS from source, return {node_id: distance}."""
    dist: Dict[str, int] = {source: 0}
    queue = deque([source])
    while queue:
        nid = queue.popleft()
        d = dist[nid]
        if d >= max_depth:
            continue
        for nb in adj.get(nid, ()):
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return dist
